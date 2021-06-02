import os, json
import gym
import numpy as np
import tensorflow as tf
from tqdm import trange

from dqn_agent import MNQRDqnAgent

print(tf.__version__)
print("GPU Available: ", tf.test.is_gpu_available())

class Obj(object):
    def __init__(self):
        pass

args = Obj()

args.environment = "LunarLander-v2"
args.batch_size = 512
args.epochs = None
args.epoch_cycles = 20
args.rollout_steps = 2000
args.buffer_size = 1000000
args.num_steps = 2000000
args.experiment = 'mnqrdqn_8atoms_s256x128_bs1024_adam2e4_noper_normal_nonstep_l2loss'
args.load = True
args.train = False

env = gym.make( args.environment )
args.action_dim = env.action_space.n
args.state_dim = env.observation_space.shape[0]

dqn = MNQRDqnAgent( args.state_dim, args.action_dim, args.buffer_size, args.batch_size, args.experiment, 256, 128, 8, args.train )

base_dir = os.getcwd() + '/models/' + args.environment + '_' + args.experiment + '/'
run_number = 0
while os.path.exists(base_dir + str(run_number)): run_number += 1

if args.load: dqn.restore_training( base_dir + 'training/' )

os.makedirs( base_dir + str(run_number) )

def reset(env, agent, args):
    s = env.reset()
    _s = np.zeros_like( s )
    w, M = agent.reset()
    pm = np.zeros_like( M )
    a = np.zeros( args.action_dim )
    r = np.zeros( 16 )
    return s, _s, w, M, pm, a, r

S, _S, W, M, PM, A, R = reset( env, dqn, args )

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
train_steps = 0

bar = trange(nb_epochs)
for epoch in bar:
    for cycle in range(args.epoch_cycles):
        for rollout in range(args.rollout_steps):
            """
            Get an action from neural network and run it in the environment
            """
            a, action, m, pm, w = dqn.act( 
                tf.convert_to_tensor( [ S ], dtype = tf.float32 ),
                tf.convert_to_tensor( [ _S ], dtype = tf.float32 ), 
                tf.convert_to_tensor( [ M ], dtype = tf.float32 ), 
                tf.convert_to_tensor( [ PM ], dtype = tf.float32 ), 
                tf.convert_to_tensor( [ W ], dtype = tf.float32 ), 
                tf.convert_to_tensor( [ A ], dtype = tf.float32 ), 
                tf.convert_to_tensor( [ R ], dtype = tf.float32 ),
                total_steps, args.train )
            
            # remove the batch_size dimension if batch_size == 1
            next_state, reward, is_terminal, _ = env.step(action)
            reward /= 10.0
            next_state, reward = np.float32(next_state), np.float32(reward)
            
            dqn.step( S, action, reward, next_state, is_terminal, _S, M, PM, W, A, R )
            
            episode_rewards += reward

            if is_terminal:
                S, _S, W, M, PM, A, R = reset( env, dqn, args )
                episode_steps = 0
                episode_rewards = 0
            else:
                _S = S
                S = next_state
                # _, _, W, M, PM, A, R = reset( env, dqn, args )
                # M = m
                # PM = pm
                # W = w
                # A = a
                # R = np.random.normal( reward, 0.01, size = 16 )
                # episode_steps += 1

            if not args.train:
                env.render()

            total_steps += 1

            bar.set_description('Steps: {} - TSteps: {} - Esteps: {}'.format( total_steps, train_steps, episode_steps ) )
            bar.refresh() # to show immediately the update

        if args.train:
            if len(dqn.memory) >= args.batch_size * 4:
                for _ in range(10):
                    dqn.learn()
                    train_steps += 1
            dqn.save_training( base_dir + 'training/' )
