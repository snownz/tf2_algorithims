import os, json
import gym
import numpy as np
import tensorflow as tf
from tqdm import trange

from dqn_agent import NQRDqnAgent

print(tf.__version__)
print("GPU Available: ", tf.test.is_gpu_available())

class Obj(object):
    def __init__(self):
        pass

args = Obj()

args.environment = "LunarLander-v2"
args.batch_size = 2048
args.epochs = None
args.epoch_cycles = 2
args.rollout_steps = 1000
args.buffer_size = 30000
args.num_steps = 2000000
args.experiment = 'nqrdqn_8atoms_s256x128_bs4096_adam2e4_noper_normal_nonstep_l2loss_v0'
args.load = False
args.train = True

env = gym.make( args.environment )
args.action_dim = env.action_space.n
args.state_dim = env.observation_space.shape[0]

dqn = NQRDqnAgent( args.state_dim, args.action_dim, args.buffer_size, args.batch_size, args.experiment, 128, 64, 8, args.train, tau = 1e-3 )

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
count = 0
total_steps = 0
train_steps = dqn.t_step

bar = trange(1000000)
for epoch in bar:
    for cycle in range(args.epoch_cycles):
        for rollout in range(args.rollout_steps):
            """
            Get an action from neural network and run it in the environment
            """
            action = dqn.act( tf.convert_to_tensor( [ state ], dtype = tf.float32 ), total_steps, args.train )
            
            # remove the batch_size dimension if batch_size == 1
            next_state, reward, is_terminal, _ = env.step(action)
            reward /= 10.0
            next_state, reward = np.float32(next_state), np.float32(reward)
            
            dqn.step( state, action, reward, next_state, is_terminal )
            episode_rewards += reward

            # check if game is terminated to decide how to update state, episode_steps,
            # episode_rewards
            if is_terminal:
                state = np.float32(env.reset())                
                episode_steps = 0
                episode_rewards = 0
            else:
                state = next_state
                episode_steps += 1

            if not args.train:
                # if total_steps%10==0:
                env.render()

            total_steps += 1

            bar.set_description('Steps: {} - TSteps: {} - Esteps: {}'.format( total_steps, train_steps, episode_steps ) )
            bar.refresh() # to show immediately the update

        if len(dqn.memory) >= args.batch_size * 4 and args.train:
            for _ in range( 20 ):
                dqn.learn()
                train_steps += 1

        if args.train:
            dqn.save_training( base_dir + 'training/' )
