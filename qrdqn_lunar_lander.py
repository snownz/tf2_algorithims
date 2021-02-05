import os, json
import gym
import numpy as np
import tensorflow as tf
from tqdm import trange

from dqn_agent import QRDqnAgent, NQRDqnAgent, AGNNQRDqnAgent

print(tf.__version__)
print("GPU Available: ", tf.test.is_gpu_available())

#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth( physical_devices[0], True )

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
args.batch_size = 1024 * 1
args.epochs = None
args.epoch_cycles = 20
args.rollout_steps = 1000
args.start_timesteps = 10
args.buffer_size = 30000
args.num_steps = 2000000
# args.experiment = 'qrdqn_8atoms_s256x128_bs9216_adam2e4_noper_normal_nonstep_l2loss'
# args.experiment = 'agnostic_qrdqn_8atoms_s256x128_bs9216_adam2e4_noper_normal_nonstep_l2loss'
args.experiment = 'agnostic_out_qrdqn_8atoms_s256x128_bs9216_adam2e4_noper_normal_nonstep_l2loss'
args.load = True
args.train = False

env = gym.make( args.environment )
args.action_dim = env.action_space.n
args.state_dim = env.observation_space.shape[0]

# dqn = NQRDqnAgent( args.state_dim, args.action_dim, args.buffer_size, args.batch_size, args.experiment, 256, 128, 8, args.train, priorized_exp = False, gn = False )
dqn = AGNNQRDqnAgent( args.state_dim, args.action_dim, args.buffer_size, args.batch_size, args.experiment, 256, 128, 8, args.train, priorized_exp = False, gn = False )

base_dir = os.getcwd() + '/models/' + args.environment + '_' + args.experiment + '/'
run_number = 0
while os.path.exists(base_dir + str(run_number)): run_number += 1

if args.load: dqn.restore_training( base_dir + 'training/' )

os.makedirs( base_dir + str(run_number) )

state = env.reset()
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
                action = dqn.act( tf.convert_to_tensor([state], dtype=tf.float32), total_steps, args.train )
            
            # remove the batch_size dimension if batch_size == 1
            next_state, reward, is_terminal, _ = env.step(action)
            reward /= 10.0
            next_state, reward = np.float32(next_state), np.float32(reward)
            dqn.step( state, action, reward, next_state, is_terminal )
            episode_rewards += reward

            # check if game is terminated to decide how to update state, episode_steps,
            # episode_rewards
            #env.render()
            if is_terminal:
                state = np.float32(env.reset())                
                episode_steps = 0
                episode_rewards = 0
            else:
                state = next_state
                episode_steps += 1

            if not args.train:
                env.render()

            total_steps += 1

            average_rewards.pop( 0 )
            average_rewards.append( reward )

            bar.set_description('average_rewards: {} - Steps: {} - TSteps: {}'.format( np.mean( average_rewards ), total_steps, train_steps ) )
            bar.refresh() # to show immediately the update

            if len(dqn.memory) >= args.batch_size and args.train:
                dqn.learn()
                train_steps += 1

        dqn.save_training( base_dir + 'training/' )
