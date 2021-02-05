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
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, SIMPLE_MOVEMENT, RIGHT_ONLY
import gc
from a2c_2D_simple import EncoderModel, ActorModel, CriticModel, A2CTrain, A2CGaeTrain
from super_mario_env import create_train_env
from random import sample, choice

def render(_env, env, a, ac, fig, ax):

    ax.clear()
    ax.bar(np.arange(len(RIGHT_ONLY)), ac, color='red')

    fig.canvas.draw_idle()

    # convert canvas to image
    img1 = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img1  = img1.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # img is rgb, convert to opencv's default bgr
    img1 = cv.resize( cv.cvtColor(img1,cv.COLOR_RGB2BGR), (500,500) )
    img21 = cv.resize( cv.cvtColor(_env[:,:,0], cv.COLOR_GRAY2BGR), (250,250) ) * 255.0
    img22 = cv.resize( cv.cvtColor(_env[:,:,1], cv.COLOR_GRAY2BGR), (250,250) ) * 255.0
    img23 = cv.resize( cv.cvtColor(_env[:,:,2], cv.COLOR_GRAY2BGR), (250,250) ) * 255.0
    img24 = cv.resize( cv.cvtColor(_env[:,:,3], cv.COLOR_GRAY2BGR), (250,250) ) * 255.0

    font = cv.FONT_HERSHEY_SIMPLEX
    org = (50, 200)
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2
    env = cv.resize( env, (500,500) )
    env = cv.putText( env, str(a), org, font, fontScale, color, thickness, cv.LINE_AA)

    img2 = np.vstack( ( np.hstack( ( img21, img22 ) ), np.hstack( ( img23, img24 ) ) ) ).astype(np.uint8)

    img = np.hstack( ( env[:,:,::-1], img1, img2 ) )

    cv.imshow("game",img)
    cv.waitKey(1)

size = ( 84, 84 )
bs = 1024
buffer_memory = 5000
t_steps = 1000
n_episodes = 100000
num_local_steps = 100
num_epochs = 5
gamma = 0.9
tau = 1.0

env = create_train_env( 1, 1, RIGHT_ONLY, random=True )

e_model = EncoderModel()
act_model = ActorModel( env.action_space.n )
c_model = CriticModel()
t_model = A2CGaeTrain( e_model, act_model, c_model )

fig, axs = plt.subplots(1)
fig.suptitle('Vertically stacked subplots')

curr_states, state_ = env.reset()
curr_episode = 0
ct = 0
while True:
    curr_episode += 1
    actions = []
    values = []
    states = []
    _states = []
    rewards = []
    dones = []
    old_log_policies = []
    total_reward = 0

    for _ in range( num_local_steps ):

        states.append( curr_states )

        embeding = e_model( tf.convert_to_tensor( curr_states[None, ...], dtype = tf.float32 ) )

        action, p, log_policy = act_model.get_action( embeding )
        value = np.asarray( c_model( embeding ) )[0][0]

        render( curr_states, state_, action, p, fig, axs )

        state, state_, reward, done, info = env.step( action )
        total_reward += reward
        curr_states = state

        old_log_policies.append( log_policy )
        values.append( value )
        rewards.append( reward )
        actions.append( action )
        dones.append( done )
        _states.append( curr_states )

        if done:
            t_model.reward_board(total_reward)
    
    old_log_policies = np.asarray(old_log_policies)
    values = np.asarray(values)
    states = np.asarray(states)
    actions = np.asarray(actions)
    rewards = np.asarray(rewards)
    _states = np.asarray(_states)
    dones = np.asarray(dones)
    
    print('Training')

    t_model.train( states, actions, rewards, _states, values, old_log_policies, dones )

    states = []
    actions = []
    rewards = []
    _states = []
    dones = []

    with t_model.train_summary_writer.as_default():
        tf.summary.scalar('actor loss', t_model.train_loss.result(), step=ct)
        tf.summary.scalar('critic loss', t_model.train_loss_c.result(), step=ct)
        tf.summary.scalar('reward', t_model.reward_board.result(), step=ct)

    ct+=1

    gc.collect()


