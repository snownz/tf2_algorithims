import os, gc, threading, logging, time
import tensorflow as tf
import cv2 as cv
import numpy as np
from functools import partial
import json
from random import randint, random
from time import sleep
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import signal
import gc
from a2c_2D_simple import EncoderModel, ActorModel, CriticModel, A2CTrain, A2CGaeTrain
from random import sample, choice, randint
from threading import Thread
from queue import Queue, Empty
from envs import create_atari_train_env

MOVIMENTS = [
    ['NOOP'],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['A'],
    ['A'],
    ['A'],
    ['A'],
    ['A'],
    ['A'],
    ['A'],
    ['A'],
    ['A'],
    ['A'],
    ['A'],
    ['A'],
    ['A'],
]

g_states = []
g_actions = []
g_rewards = []
g_next_states = []
g_values = []
g_old_log_policies = [] 
g_R = []
g_adv = []
g_dones = []

def render(_env, env, a, ac, fig, ax, name):

    ax.clear()
    ax.bar(np.arange(len(MOVIMENTS)), ac, color='red')

    fig.canvas.draw_idle()

    # convert canvas to image
    img1 = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img1  = img1.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # img is rgb, convert to opencv's default bgr
    img1 = cv.resize( cv.cvtColor(img1,cv.COLOR_RGB2BGR), (500,500) )
    img2 = cv.resize( _env, (500,500) ) * 255.0

    font = cv.FONT_HERSHEY_SIMPLEX
    org = (50, 200)
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2
    env = cv.resize( env, (500,500) )
    env = cv.putText( env, str(a), org, font, fontScale, color, thickness, cv.LINE_AA)

    #img = np.hstack( ( env[:,:,::-1], img1 ) )
    img = np.hstack( ( env[:,:,::-1], img1, img2.astype(np.uint8)[:,:,::-1] ) )

    cv.imshow(name,img)
    cv.waitKey(1)

def ac_env_runner_explorer(model, env, num_local_steps):

    curr_states, state_ = env.reset()
    curr_episode = 0
    c = 0

    fig, axs = plt.subplots(1)
    fig.suptitle('Actions Probability')
    while True:

        curr_episode += 1        
        total_reward = 0

        rollout = MemoryRollout()

        for v in range( num_local_steps ):

            action, p, log_policy, value = model.get_values( curr_states )

            next_state, state_, reward, done, info = env.step( int(action) )
            total_reward += reward

            render( curr_states, state_, action, p, fig, axs, model.name )

            curr_tuple = [ curr_states, next_state, log_policy, action, value, reward, done ]

            curr_states = next_state

            if done: model.set_board( 'reward', total_reward )

            rollout.add(*curr_tuple)

            c += 1

            yield rollout
        
        with model.summary_writer.as_default():
            tf.summary.scalar('reward', model.get_board('reward'), step=model.steps)
        model.steps += 1

class MemoryRollout(object):
   
    def __init__(self):
        
        self.states = []
        self._states = []
        self.log_polices = []
        self.actions = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.terminal = False

    def add(self, curr_states, next_state, log_policy, action, value, reward, done):

        self.states += [curr_states]
        self._states += [next_state]
        self.log_polices += [log_policy]
        self.actions += [action]
        self.values += [value]
        self.rewards += [reward]
        self.dones += [done]
        self.terminal = done

    def extend(self, rollout):

        self.states.extend( rollout.states )
        self._states.extend( rollout._states )
        self.log_polices.extend( rollout.log_polices )
        self.actions.extend( rollout.actions )
        self.values.extend( rollout.values )
        self.rewards.extend( rollout.rewards )
        self.dones.extend( rollout.dones )
        self.terminal = rollout.terminal

    def size(self):
        return len( self.states )

class ExplorerWorker:

    def __init__(self, name, actions):

        self.e_model = EncoderModel()
        self.act_model = ActorModel( actions )
        self.c_model = CriticModel()
        self.t_model = A2CTrain(self.e_model, self.act_model, self.c_model, name, lr=2e-5)
        self.name = name
        self.steps = 0
        
        self.log_dir = 'logs/{}'.format(name)
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)
        self.boards = {
            'reward': tf.keras.metrics.Mean('reward_board', dtype=tf.float32),
            'actor_loss': tf.keras.metrics.Mean('train_loss', dtype=tf.float32),
            'critic_loss': tf.keras.metrics.Mean('train_loss_c', dtype=tf.float32),
        }

    def set_weights(self, model):
        self.e_model.update_model( model.e_model )
        self.act_model.update_model( model.act_model )
        self.c_model.update_model( model.c_model )

    def get_weights(self):
        return self.e_model.trainable_variables + self.act_model.trainable_variables + self.c_model.trainable_variables

    def set_board(self, name, value):
        self.boards[name](value)
    
    def get_board(self, name):
        return self.boards[name].result()

    def get_values(self, state):

        embeding = self.e_model( tf.convert_to_tensor( state[None, ...], dtype = tf.float32 ) )
        action, p, log_policy = self.act_model.get_action( embeding )
        value = np.asarray( self.c_model( embeding ) )[0][0]

        return action, p, log_policy, value

    def train(self, states, actions, rewards, next_states, dones):
        self.t_model.train( states, actions, rewards, next_states, dones )

    #def update_model(self, cg, ag, eg):
    #    self.t_model.apply_grads( cg, ag, eg )

size = ( 84, 84 )

print("Creating envs")
env = create_atari_train_env( 'Pong-v0', MOVIMENTS )

MOVIMENTS = [ x for x in range( env.action_space.n ) ]

print("Crating Models")
worker = ExplorerWorker( 'atari_worker', len(MOVIMENTS) )

print("Starting")
curr_states, state_ = env.reset()
curr_episode = 0
c = 0
gamma = .99
tau = 1.0

fig, axs = plt.subplots(1)
fig.suptitle('Actions Probability')
while True:

    curr_episode += 1        
    total_reward = 0

    rollout = MemoryRollout()

    for v in range( 100 ):

        action, p, log_policy, value = worker.get_values( curr_states )

        next_state, state_, reward, done, info = env.step( int(action) )
        total_reward += reward

        if v%2 == 0:
            render( curr_states, state_, action, p, fig, axs, worker.name )

        curr_tuple = [ curr_states, next_state, log_policy, action, value, reward, done ]

        curr_states = next_state

        rollout.add(*curr_tuple)

        c += 1

    states = rollout.states[:-1]
    _states = rollout._states[:-1]
    log_policys = rollout.log_polices[:-1]
    actions = rollout.actions[:-1]
    values = rollout.values[:-1]
    rewards = rollout.rewards[:-1]
    dones = rollout.dones[:-1]

    next_value = values[-1]

    gae = 0
    R = []
    for value, reward, done in list(zip(values, rewards, dones))[::-1]:
        gae = gae * gamma * tau
        gae = gae + reward + gamma * next_value * ( 1 - done ) - value
        next_value = value
        R.append( gae + value )
    R = R[::-1]

    R = np.asarray( R ).squeeze()
    adv = R - values

    worker.train( np.asarray( states ), 
                  np.asarray( actions ), 
                  np.asarray( rewards ), 
                  np.asarray( _states ),
                  np.asarray( dones ) )
        
    del rollout
    del states
    del _states
    del log_policy
    del action
    del value
    del reward
    del R
    del adv
    del done

    gc.collect()    
    
    worker.steps += 1