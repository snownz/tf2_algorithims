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

def test_reward(env, agent):
    total_reward = 0
    state = env.reset()
    done = False
    while not done:
        action = agent.act_max( tf.convert_to_tensor( [ state ], dtype = tf.float32 ) )
        next_state, reward, done, _ = env.step( action )
        state = next_state
        total_reward += reward
    return total_reward

def preprocess1(states, actions, rewards, done, values, gamma):
    g = 0
    lmbda = 0.95
    returns = []
    for i in reversed(range(len(rewards))):
       delta = rewards[i] + gamma * values[i + 1] * done[i] - values[i]
       g = delta + gamma * lmbda * dones[i] * g
       returns.append(g + values[i])

    returns.reverse()
    adv = np.array(returns, dtype=np.float32) - values[:-1]
    adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-10)
    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.int32)
    returns = np.array(returns, dtype=np.float32)
    return states, actions, returns, adv

steps = 1000
ep_reward = []
total_avgr = []
target = False 
best_reward = 0
avg_rewards_list = []

for s in range( steps ):
    
    if target == True:
        break
    
    done = False
    state = env.reset()
    all_aloss = []
    all_closs = []
    rewards = []
    states = []
    next_states = []
    actions = []
    probs = []
    dones = []
    values = []

    print("Episod: {}".format(s))
    for e in range(128):

        action, logits = ppo.act( tf.convert_to_tensor( [ state ], dtype = tf.float32 ) )
        value = ppo.critic( tf.convert_to_tensor( [ state ], dtype = tf.float32 ), tf.convert_to_tensor( [ action ], dtype = tf.int32 ) )
        next_state, reward, done, _ = env.step( action )
        
        dones.append( 1-done )
        rewards.append( reward )
        states.append( state )
        next_states.append( next_state )
        actions.append( action )
        probs.append( logits[0] )
        values.append( value )

        state = np.copy( next_state )

        if done:
            env.reset()

    action, _ = ppo.act( tf.convert_to_tensor( [ state ], dtype = tf.float32 ) )
    value = ppo.critic( tf.convert_to_tensor( [ state ], dtype = tf.float32 ), tf.convert_to_tensor( [ action ], dtype = tf.int32 ) )
    values.append( value )

    states, actions,returns, adv = preprocess1( states, actions, rewards, dones, values, 1 )

    for epocs in range(10):
        ppo.learn( tf.convert_to_tensor( states, dtype = tf.float32 ), 
                   tf.convert_to_tensor( next_states, dtype = tf.float32 ), 
                   tf.convert_to_tensor( actions, dtype = tf.int32 ), 
                   tf.convert_to_tensor( adv, dtype = tf.float32 ), 
                   tf.convert_to_tensor( probs, dtype = tf.float32 ), 
                   tf.convert_to_tensor( returns, dtype = tf.float32 ), 
                   tf.convert_to_tensor( dones, dtype = tf.float32 ),
                   tf.convert_to_tensor( rewards, dtype = tf.float32 ) )

    avg_reward = np.mean( [ test_reward( env, ppo ) for _ in range(5) ] )

    print( 'Total Reward: {}'.format( avg_reward ) )

    avg_rewards_list.append(avg_reward)

    if avg_reward > best_reward:

        print( 'Best Reward: {}'.format( avg_reward ) )

        best_reward = avg_reward

    if best_reward == 200:
        target = True

    env.reset()

env.close()
