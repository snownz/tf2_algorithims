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
from random import sample, choice, randint
from threading import Thread
from queue import Queue, Empty

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

    cv.imshow(name,img)
    cv.waitKey(1)

def render_save(_env, env, a, ac, name, i):

    fig, ax = plt.subplots(1)
    fig.suptitle('Vertically stacked subplots')
    
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

    if not os.path.isdir( 'results/{}'.format(name) ):
        os.mkdir( 'results/{}'.format(name) )
    
    cv.imwrite( 'results/{}/{}.jpg'.format(name, i),img)

def ac_env_runner_explorer(model, env, num_local_steps):

    curr_states, state_ = env.reset()
    curr_episode = 0
    c = 0
    while True:

        curr_episode += 1        
        total_reward = 0

        rollout = MemoryRollout()

        for v in range( num_local_steps ):

            action, p, log_policy, value = model.get_values( curr_states )

            next_state, state_, reward, done, info = env.step( action )
            total_reward += reward

            curr_tuple = [ curr_states, next_state, log_policy, action, value, reward, done ]

            curr_states = next_state

            if done: model.set_board( 'reward', total_reward )

            rollout.add(*curr_tuple)

            c += 1

            yield rollout
        
        with model.summary_writer.as_default():
            tf.summary.scalar('reward', model.get_board('reward'), step=model.steps)
        model.steps += 1

def ac_env_runner_eval(model, env, num_local_steps):

    curr_states, state_ = env.reset()

    fig, axs = plt.subplots(1)
    fig.suptitle('Vertically stacked subplots')
    
    while True:
      
        total_reward = 0

        for _ in range( num_local_steps ):

            action, p, log_policy, value = model.get_values( curr_states )

            render( curr_states, state_, action, p, fig, axs, model.name )

            next_state, state_, reward, done, info = env.step( action )
            total_reward += reward

            curr_states = next_state

            if done: model.set_board( 'reward', total_reward )
        
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
        self.t_model = A2CGaeTrain(self.e_model, self.act_model, self.c_model, name, lr=2e-6)
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

    def train(self, states, actions, rewards, next_states, values, old_log_policies, R, adv, dones):
        return self.t_model.get_gradients( states, actions, rewards, next_states, values, old_log_policies, R, adv, dones )

    def update_model(self, cg, ag, eg):
        self.t_model.apply_grads( cg, ag, eg )

class TotalWorker:

    def __init__(self, name, actions):

        self.e_model = EncoderModel()
        self.act_model = ActorModel( actions )
        self.c_model = CriticModel()
        self.t_model = A2CGaeTrain(self.e_model, self.act_model, self.c_model, name, lr=2e-4)
        self.name = name
        self.steps = 0
        
        self.log_dir = 'logs/{}'.format(name)
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)
        self.boards = {
            'reward': tf.keras.metrics.Mean('reward_board', dtype=tf.float32),
            'actor_loss': tf.keras.metrics.Mean('train_loss', dtype=tf.float32),
            'critic_loss': tf.keras.metrics.Mean('train_loss_c', dtype=tf.float32)
        }
        
    def set_board(self, name, value):
        self.boards[name](value)
    
    def get_board(self, name):
        return self.boards[name].result()

    def get_values(self, state):
        embeding = self.e_model( tf.convert_to_tensor( state[None, ...], dtype = tf.float32 ) )
        action, p, log_policy = self.act_model.get_action( embeding )
        value = np.asarray( self.c_model( embeding ) )[0][0]
        return action, p, log_policy, value

    def update_model(self, cg, ag, eg):
        self.t_model.apply_grads( cg, ag, eg )

    def get_weights(self):
        return self.e_model.trainable_variables + self.act_model.trainable_variables + self.c_model.trainable_variables

class Worker(Thread):

    def __init__(self, model, env, num_local_steps):

        threading.Thread.__init__(self)
        self.model = model
        self.queue = Queue(5)
        self.num_local_steps = num_local_steps
        self.env = env

    def start_runner(self):
        self.start()

    def run(self):
        self._run()

    def _run(self):

        if type(self.model) is ExplorerWorker:

            rollout_provider = ac_env_runner_explorer( self.model, self.env, self.num_local_steps )            
            while True: 
                self.queue.put( next( rollout_provider ), timeout = 600.0 )        
        else:
            while True: ac_env_runner_eval( self.env, self.model, self.num_local_steps )

class Experiencer(Thread):

    def __init__(self, worker, tmodel):

        threading.Thread.__init__(self)
        self.worker = worker
        self.tmodel = tmodel
        self.gamma = 0.99
        self.tau = 1.0

    def start_runner(self):
        self.start()

    def run(self):
        self._run()

    def _run(self):

        while True:
            time.sleep(2)
            try:
                model = self.worker.model
                state, _state, log_policy, action, value, reward, R, adv, done = self._process()

                l1, l2, cg, ag, eg, polycy = model.train( np.asarray( state ), 
                                                  np.asarray( action ), 
                                                  np.asarray( reward ), 
                                                  np.asarray( _state ), 
                                                  np.asarray( value ), 
                                                  np.asarray( log_policy ), 
                                                  np.asarray( R ), 
                                                  np.asarray( adv ), 
                                                  np.asarray( done ) )
        
                model.set_board('actor_loss', l1)
                model.set_board('critic_loss', l2)

                self.tmodel.update_model( cg, ag, eg )
                # model.update_model( cg, ag, eg )
                model.set_weights( self.tmodel )

                with model.summary_writer.as_default():

                    tf.summary.scalar('actor_loss', model.get_board('actor_loss'), step=model.steps)
                    tf.summary.scalar('critic_loss', model.get_board('critic_loss'), step=model.steps)
                                        
                    for v in model.get_weights():
                        tf.summary.histogram( v.name, v, step=model.steps)
                    
                    tf.summary.histogram( '{}_policy'.format( model.name ), polycy, step=model.steps)

                with self.tmodel.summary_writer.as_default():                                        
                    for v in self.tmodel.get_weights():
                        tf.summary.histogram( v.name, v, step=model.steps)                

                self.tmodel.steps += 1
                model.steps += 1

                gc.collect()

            except: 
                continue

    def _pull_batch_from_queue(self):
        """
        Take a rollout from the queue of the thread runner.
        """
        rollout = self.worker.queue.get( timeout = 600.0 )
        while rollout.size() < 256:
            try:
                rollout.extend( self.worker.queue.get_nowait() )
            except Empty:
                pass
        print(rollout.size())
        return rollout

    def _process(self):
        
        rollout = self._pull_batch_from_queue()
        
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
            gae = gae * self.gamma * self.tau
            gae = gae + reward + self.gamma * next_value * ( 1 - done ) - value
            next_value = value
            R.append( gae + value )
        R = R[::-1]

        R = np.asarray( R ).squeeze()
        adv = R - values

        return states, _states, log_policys, actions, values, rewards, R, adv, dones

class A3C:

    def __init__( self, env, envs, num_workers, num_actions ):

        self.trainer = TotalWorker( 'omega',  num_actions )

        self.workers = [ ExplorerWorker( 'worker_{}'.format( x ), num_actions ) for x in range( num_workers ) ]
        self.explorers = [ Worker( w, ev, 100 ) for ev, w in zip( envs, self.workers ) ]
        self.experiences = [ Experiencer( w, self.trainer ) for w in self.explorers ]        
        self.global_runner = Worker( env, self.trainer, 100 )

    def start(self, run_global=False):
        if run_global: self.global_runner.start_runner()
        for e, exp in zip( self.explorers, self.experiences ): 
            e.start_runner()
            exp.start_runner()

size = ( 84, 84 )
num_workers = 6

print("Creating envs")
envs = [ create_train_env( randint(1,8), randint(1,4), RIGHT_ONLY, random=False ) for x in range( num_workers ) ]
env = create_train_env( 1, 1, RIGHT_ONLY, random=False )

print("Crating Models")
a3c = A3C( env, envs, num_workers, env.action_space.n )

print("Starting")
a3c.start(True)