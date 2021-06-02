import os, json
import gym
import numpy as np
import tensorflow as tf
from tqdm import trange
import time

from dqn_agent import RecurrentQRDqnAgent

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
args.batch_size = 1024
args.epochs = None
args.epoch_cycles = 2000
args.rollout_steps = 1000
args.sequence = 16
args.buffer_size = 15000
args.num_steps = 2000000

args.experiment = 'rqrdqn_gru_8atoms_s256x128x128_bs32_seq32_adam2e4_noper_normal_nonstep_sumhalf'

args.load = False
args.train = True

env = gym.make( args.environment )
args.action_dim = env.action_space.n
args.state_dim = env.observation_space.shape[0]

dqn = RecurrentQRDqnAgent( args.state_dim, args.action_dim, args.buffer_size, args.batch_size, 
                           args.sequence, args.experiment, 256, 128, 16, 8, args.sequence, 'g',
                           priorized_exp = False, reduce = 'sum_half', train = args.train )

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
count_down = 64
total_steps = 0
train_steps = 0

p_state = dqn.reset(1)
states = list( np.zeros( [ args.sequence, args.state_dim ] ) )
states_ = list( np.zeros( [ args.sequence, args.state_dim ] ) )
rewards = list( np.zeros( [ args.sequence ] ) )
actions = list( np.zeros( [ args.sequence ] ) )
dones = list( np.zeros( [ args.sequence ] ) )
prevs = p_state

bar = trange(nb_epochs)
for epoch in bar:
    for cycle in range(args.epoch_cycles):
        for rollout in range(args.rollout_steps):
            
            """
            Get an action from neural network and run it in the environment
            """           
            action, p_state = dqn.act( tf.convert_to_tensor( [[state]], dtype=tf.float32 ), 
                                        prevs, total_steps, args.train )
            
            # remove the batch_size dimension if batch_size == 1
            next_state, reward, is_terminal, _ = env.step(action)
            reward /= 10.0
            next_state, reward = np.float32(next_state), np.float32(reward)
            
            states.pop( 0 )
            states_.pop( 0 )
            rewards.pop( 0 )
            actions.pop( 0 )
            dones.pop( 0 )

            states.append( state )
            states_.append( next_state )
            rewards.append( reward )
            actions.append( action )
            dones.append( is_terminal )

            count_down -= 1
            if args.train and ( count_down <= 0 or is_terminal ):
                dqn.step( np.array( states ), np.array( actions ), np.array( rewards ), np.array( states_ ), np.array( dones ), prevs.numpy()[0] )
            
            episode_rewards += reward
            # p_state = dqn.reset(1)

            # check if game is terminated to decide how to update state, episode_steps,
            # episode_rewards
            if is_terminal:

                p_state = dqn.reset(1)                
                states = list( np.zeros( [ args.sequence, args.state_dim ] ) )
                states_ = list( np.zeros( [ args.sequence, args.state_dim ] ) )
                rewards = list( np.zeros( [ args.sequence ] ) )
                actions = list( np.zeros( [ args.sequence ] ) )
                dones = list( np.zeros( [ args.sequence ] ) )

                count_down = 64
                
                state = np.float32(env.reset())                
                episode_steps = 0
                episode_rewards = 0

            else:

                state = next_state
                episode_steps += 1

            if not args.train:
                env.render()

            prevs = p_state

            if len(dqn.memory) >= args.batch_size * 4 and args.train:
                dqn.learn()
                train_steps += 1

            total_steps += 1

            bar.set_description('Steps: {} - TSteps: {} - Buffer: {}'.format( total_steps, train_steps, len(dqn.memory) ) )
            bar.refresh() # to show immediately the update

        # if len(dqn.memory) >= args.batch_size * 4 and args.train:
            
        #     for _ in range( args.rollout_steps // 2 ):
        #         dqn.learn()
        #         train_steps += 1
            
        #     if train_steps % 10000 == 0:
        #         dqn.save_training( base_dir + 'training/' )
