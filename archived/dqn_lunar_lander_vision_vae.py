import os, json
import gym
import numpy as np
import tensorflow as tf
from tqdm import trange
import cv2 as cv

# from tensorflow.keras.mixed_precision import experimental as mixed_precision
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_policy(policy)

from dqn_agent import DqnAgentVisionVQVae

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth( physical_devices[0], True )

print(tf.__version__)
print("GPU Available: ", tf.test.is_gpu_available())

class Obj(object):
    def __init__(self):
        pass

args = Obj()

args.environment = "LunarLander-v2"
args.normalize_obs = False
args.noise  = 'normal'
args.gamma = 0.99
args.tau = 5e-3
args.noise_scale = 0.2
args.batch_size = 128
args.epochs = None
args.epoch_cycles = 20
args.rollout_steps = 100
args.T = 50
args.start_timesteps = 10
args.buffer_size = 5000
args.num_steps = 2000000
args.experiment = 's256x128_bs1024x256_adam2e7_per_normal_nonstep'
args.load = True

env = gym.make( args.environment )
args.action_dim = env.action_space.n
args.state_dim = env.observation_space.shape[0]

dqn = DqnAgentVisionVQVae( args.state_dim, args.action_dim, args.buffer_size, args.batch_size, args.experiment, 256, 128, priorized_exp = True, gn = True )

base_dir = os.getcwd() + '/models/' + args.environment + '_' + args.experiment + '/'
run_number = 0
while os.path.exists(base_dir + str(run_number)): run_number += 1

# if args.load: dqn.restore_training( base_dir + 'training/' )

os.makedirs( base_dir + str(run_number) )

state = env.reset()
state = cv.resize( env.render(mode="rgb_array"), ( 96, 96 ) )
results_dict = {
    'train_rewards': [],
    'eval_rewards': [],
    'actor_losses': [],
    'value_losses': [],
    'critic_losses': []
}
episode_steps, episode_rewards = 0, 0 # total steps and rewards for each episode

num_steps = args.num_steps
if num_steps is not None:
    nb_epochs = int(num_steps) // (args.epoch_cycles * args.rollout_steps)
else:
    nb_epochs = 500

"""
training loop
"""
average_rewards = [ 0 ] * 100
count = 0
total_steps = 0
train_steps = 0

bar = trange(nb_epochs)
for epoch in bar:
    for cycle in range(args.epoch_cycles):
        for rollout in range(args.rollout_steps):
            """
            Get an action from neural network and run it in the environment
            """
            if total_steps < args.start_timesteps:
                action = env.action_space.sample()
            else:
                action = dqn.act( tf.convert_to_tensor([state], dtype=tf.float32), eps = 0.1 )
            
            # remove the batch_size dimension if batch_size == 1
            ns, reward, is_terminal, _ = env.step(action)
            next_state = cv.resize( env.render(mode="rgb_array"), ( 96, 96 ) )
            reward /= 10.0
            next_state, reward = np.float32(next_state), np.float32(reward)
            
            dqn.step( state, action, reward, next_state, is_terminal )
            episode_rewards += reward

            # check if game is terminated to decide how to update state, episode_steps,
            # episode_rewards
            if is_terminal:
                state = np.float32(env.reset())
                state = cv.resize( env.render(mode="rgb_array"), ( 96, 96 ) )
                results_dict['train_rewards'].append(
                    (total_steps, episode_rewards)
                )
                episode_steps = 0
                episode_rewards = 0
            else:
                state = next_state
                episode_steps += 1

            total_steps += 1

            average_rewards.pop( 0 )
            average_rewards.append( reward )

            bar.set_description('average_rewards: {} - Steps: {} - TSteps: {}'.format( np.mean( average_rewards ), total_steps, train_steps ) )
            bar.refresh() # to show immediately the update

            if len(dqn.memory) >= args.batch_size:
                dqn.learn_policy()
                train_steps += 1

        # # train
        if len( dqn.memory ) >= 32:
            for _ in range(args.T):
                dqn.learn_encoder()

        dqn.save_training( base_dir + 'training/' )

dqn.save( base_dir + str( run_number ) )
        
with open('results.txt', 'w') as file:
    file.write(json.dumps(results_dict))

#