import os, json
import gym
import numpy as np
import tensorflow as tf
from tqdm import trange
import cv2 as cv
import matplotlib.pyplot as plt

from dqn_agent import NQRPPOAgent

print(tf.__version__)
print("GPU Available: ", tf.test.is_gpu_available())

class Obj(object):
    def __init__(self):
        pass

args = Obj()

args.environment = "LunarLander-v2"
args.batch_size = 1024
args.epochs = 1000000
args.rollout_steps = 300
args.buffer_size = 100000
args.num_steps = 2000000
args.update_interval = 128
args.experiment = 'nqrppo_8atoms_s256x128_bs1024_adam2e4_noper_normal_nonstep_l2loss_v2'
args.load = False
args.train = True

env = gym.make( args.environment )
args.action_dim = env.action_space.n
args.state_dim = env.observation_space.shape[0]

ppo = NQRPPOAgent( args.state_dim, args.action_dim, args.buffer_size, args.batch_size, args.experiment, 256, 128, 8, 64, args.train, priorized_exp = False )

base_dir = os.getcwd() + '/models/' + args.environment + '_' + args.experiment + '/'
run_number = 0
while os.path.exists(base_dir + str(run_number)): run_number += 1

if args.load: ppo.restore_training( base_dir + 'training/' )

os.makedirs( base_dir + str(run_number) )

state = env.reset()
episode_steps, episode_rewards = 0, 0 # total steps and rewards for each episode

"""
training loop
"""
count = 0
total_ep = 0
total_steps = 0
train_steps = ppo.t_step
state_batch = []
action_batch = []
reward_batch = []
next_state_batch = []
old_policy_batch = []
dones_batch = []
episode_reward, is_terminal = 0, False
bar = trange(args.epochs)
state = env.reset()
for epoch in bar:
    while True:

        """
        Get an action from neural network and run it in the environment
        """
        action, logits = ppo.act( tf.convert_to_tensor( [ state ], dtype = tf.float32 ) )
        
        next_state, reward, is_terminal, info = env.step( action )
        reward /= 10.0
        next_state, reward = np.float32( next_state ), np.float32( reward )

        state = np.reshape( state, [ args.state_dim ] )
        action = np.reshape( action, [ 1, 1 ] )
        next_state = np.reshape( next_state, [ args.state_dim ] )
        reward = np.reshape( reward, [ 1, 1 ] )
        done = np.reshape( is_terminal, [ 1, 1 ] )
        
        state_batch.append( np.expand_dims( state, 0 ) )
        action_batch.append( action )
        next_state_batch.append( np.expand_dims( next_state, 0 ) )
        reward_batch.append( reward )
        dones_batch.append( done )
        old_policy_batch.append( logits )
        
        if is_terminal:
            total_ep += 1
            state = np.float32( env.reset() )
        else: 
            state = next_state

        if ( len(state_batch) >= args.update_interval or is_terminal ) and args.train:

            states = tf.convert_to_tensor( ppo.list_to_batch( state_batch ) )
            next_states = tf.convert_to_tensor( ppo.list_to_batch( next_state_batch ) )
            actions = tf.convert_to_tensor( ppo.list_to_batch( action_batch ) )
            rewards = tf.convert_to_tensor( ppo.list_to_batch( reward_batch ) )
            dones = tf.convert_to_tensor( ppo.list_to_batch( dones_batch ) )
            old_policys = tf.convert_to_tensor( ppo.list_to_batch( old_policy_batch ) )

            ppo.learn( states, next_states, actions, rewards, dones, old_policys, is_terminal )

            state_batch = []
            next_state_batch = []
            action_batch = []
            td_target_batch = []
            advatnage_batch = []
            reward_batch = []
            dones_batch = []
            old_policy_batch = []

            train_steps += 1

        if not args.train:
            env.render()

        total_steps += 1

        bar.set_description('Eps: {} - Steps: {} - TSteps: {}'.format( total_ep, total_steps, train_steps ) )
        bar.refresh() # to show immediately the update
    
    if args.train and total_steps%1000 == 0:
        ppo.save_training( base_dir + 'training/' )
