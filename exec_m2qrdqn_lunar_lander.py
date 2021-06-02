import os, json
import gym
import numpy as np
import tensorflow as tf
from tqdm import trange
import cv2 as cv
import matplotlib.pyplot as plt

from dqn_agent import M2NQRDqnAgent

print(tf.__version__)
print("GPU Available: ", tf.test.is_gpu_available())

class Obj(object):
    def __init__(self):
        pass

args = Obj()

args.environment = "LunarLander-v2"
args.batch_size = 128
args.sequence = 32 + 2
args.epochs = None
args.epoch_cycles = 1000
args.rollout_steps = 600
args.buffer_size = 10000
args.num_steps = 2000000
args.experiment = '2mnqrdqn_bs64_ss16_adam2e4_noper_nonstep_l2loss'
args.load = True
args.train = False

env = gym.make( args.environment )
args.action_dim = env.action_space.n
args.state_dim = env.observation_space.shape[0]

dqn = M2NQRDqnAgent( args.experiment, args.state_dim, 
                     32, 32, 
                     32, 32, 8, args.sequence, 2, 4, 4, 1, int( args.rollout_steps * 0.7 ),
                     32, 32,
                     args.action_dim, 32, 8,
                     args.train, args.batch_size, args.sequence, args.buffer_size )

base_dir = os.getcwd() + '/models/' + args.environment + '_' + args.experiment + '/'
run_number = 0
while os.path.exists(base_dir + str(run_number)): run_number += 1

if args.load: dqn.restore_training( base_dir + 'training/' )

os.makedirs( base_dir + str(run_number) )

def reset(env, agent, args):
    s = env.reset()
    _s = np.zeros_like( s )
    m, u, l, wp, wr, ww, tw = agent.reset()
    a = 0
    r = 0
    d = 0
    return s, _s, m, u, l, wp, wr, ww, a, r, d, None, tw

def norm(img):
    mx = img.max()
    mn = img.min()
    v = ( ( img - mn ) / ( mx - mn ) ) * 255
    return cv.applyColorMap( v.astype(np.uint8), cv.COLORMAP_JET )

S, _S, M, U, L, WP, WR, WW, A, R, D, P, TW = reset( env, dqn, args )

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
total_steps = 323735
global_steps = 0
train_steps = dqn.t_step

p_state = []
states = list( np.zeros( [ args.sequence, args.state_dim ] ) )
states_ = list( np.zeros( [ args.sequence, args.state_dim ] ) )
_states = list( np.zeros( [ args.sequence, args.state_dim ] ) )
rewards = list( np.zeros( [ args.sequence ] ) )
actions = list( np.zeros( [ args.sequence ] ) )
tws = list( np.zeros( [ args.sequence, 32*2 ] ) )
dones = list( np.zeros( [ args.sequence ] ) )
ssteps = list( np.zeros( [ args.sequence, ] ) )

fig, ax = plt.subplots(1)
fig.suptitle('Action Rewards Dopamine')
bar = trange(nb_epochs)
for epoch in bar:
    for cycle in range(args.epoch_cycles):
        for rollout in range(args.rollout_steps):
            """
            Get an action from neural network and run it in the environment
            """

            if len(p_state) > 0: pst = tf.concat( p_state, axis = -2 )
            else: pst = None
            

            ac, z, c, gpresent, gparams = dqn.act( 
                                                   tf.convert_to_tensor( [ [S] ], dtype = tf.float32 ),
                                                   tf.convert_to_tensor( [ [_S] ], dtype = tf.float32 ),
                                                   tf.convert_to_tensor( [ [A] ], dtype = tf.float32 )[:,tf.newaxis,0], 
                                                   tf.convert_to_tensor( [ [R] ], dtype = tf.float32 )[:,tf.newaxis,0],
                                                   tf.convert_to_tensor( [ [D] ], dtype = tf.float32 )[:,tf.newaxis,0],
                                                   TW[:,tf.newaxis,:],
                                                   [ M, U, L, WP, WR, WW ], pst, 
                                                   tf.convert_to_tensor( [ [ global_steps ] ] ), 
                                                   total_steps, args.train )
            
            # remove the batch_size dimension if batch_size == 1
            next_state, reward, is_terminal, _ = env.step( ac )
            reward /= 10.0
            next_state, reward = np.float32(next_state), np.float32(reward)

            global_steps += 1

            if not args.train:

                im1 = norm( cv.resize( M.numpy()[0] * 255, ( 200, 200 ) ) )
                im2 = norm( cv.resize( L.numpy()[0] * 255, ( 200, 200 ) ) )
                im3 = norm( cv.resize( U.numpy()[0], ( 200, 200 ) ) )
                im5 = norm( cv.resize( WR.numpy()[0] * 255, ( 200, 200 ) ) )
                im6 = norm( cv.resize( WW.numpy()[0] * 255, ( 200, 200 ) ) )

                ax.clear()
                ax.plot( np.arange( args.action_dim ), z )
                fig.canvas.draw_idle()

                # convert canvas to image
                img1 = np.frombuffer( fig.canvas.tostring_rgb(), dtype=np.uint8 )
                img1 = img1.reshape(fig.canvas.get_width_height()[::-1] + (3,))

                l1 = np.hstack( [ im1, im3 ] )
                l2 = np.hstack( [ im2, cv.resize( img1, ( 200, 200 ) ) ] )
                l3 = np.hstack( [ im5, im6 ] )

                im = np.vstack( [ l1, l2, l3 ] )

                cv.imshow( 'stats', im )
                cv.waitKey(1)

                env.render()

            if args.train:

                states.pop( 0 )
                states_.pop( 0 )
                _states.pop( 0 )
                rewards.pop( 0 )
                actions.pop( 0 )
                dones.pop( 0 )
                ssteps.pop( 0 )
                tws.pop( 0 )

                states.append( S )
                states_.append( next_state )
                _states.append( _S )
                rewards.append( reward )
                actions.append( ac )
                dones.append( is_terminal )
                ssteps.append( episode_steps )
                tws.append( TW.numpy()[0,:] )
                
                dqn.step( np.array(states), np.array(actions), np.array(rewards), np.array(states_), 
                          np.array(dones), np.array(_states), np.array(tws), M.numpy()[0], U.numpy()[0], L.numpy()[0], 
                          WP.numpy()[0], WR.numpy()[0], WW.numpy()[0], np.array( [ global_steps ] ) )
                
            episode_rewards += reward
            _S = S
            
            if is_terminal:
                S = env.reset()
                episode_steps = 0
                episode_rewards = 0
            else:
                episode_steps += 1
                S = next_state

            M, U, L, WP, WR, WW = gparams
            A = ac
            R = reward
            TW = tf.concat( [ gparams[1], gparams[5] ], axis = 1 )[:,:,0]
            
            if len( p_state ) == args.sequence:
                p_state.pop( 0 )
            p_state.append( gpresent )

            total_steps += 1
            
            bar.set_description('Steps: {} - Memory: {:.4f} - Usage: {:.4f} - Linkage: {:.4f} - Wp: {:.4f} - Read: {:.4f} - Write: {:.4f}'.format( 
                dqn.qnetwork_local.gm.memory.memory.alpha( global_steps ) , np.mean(M), np.mean(U), np.mean(L), np.mean(WP), np.mean(WR), np.mean(WW) ) )
            bar.refresh() # to show immediately the update

        if args.train:

            if len(dqn.memory) >= args.batch_size * 4:
                for _ in range( 20 ):
                    dqn.learn()
                    train_steps += 1

            global_steps = 0
            S, _S, M, U, L, WP, WR, WW, A, R, D, P, TW = reset( env, dqn, args )
            dqn.save_training( base_dir + 'training/' )
