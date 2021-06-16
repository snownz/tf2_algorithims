# Multi-Environments 
# From https://squadrick.dev/journal/efficient-multi-gym-environments.html

import pickle
import cloudpickle
import numpy as np
import gym
from gym import spaces
import cv2 as cv
from collections import namedtuple
from threading import Thread, Lock
from multiprocessing import Process, Pipe

def make_mp_envs(env_ids, seed, start_idx = 0):
    def make_env(rank, id_env):
        def fn():
            env = gym.make( id_env )
            env.seed( seed + rank )
            return env
        return fn
    return SubprocVecEnv( [ make_env( i + start_idx, id_env ) for i, id_env in enumerate( env_ids ) ] )

def make_mps_envs(env_ids, num_env, seed, start_idx = 0):
    def make_env(rank, id_env):
        def fn():
            env = MultiEnv( id_env, num_env )
            env.seed( seed + rank )
            return env
        return fn
    return SubprocVecEnv( [ make_env( i + start_idx, id_env ) for i, id_env in enumerate( env_ids ) ] )

def worker(remote, parent_remote, env_fn):
    parent_remote.close()
    env = env_fn()
    while True:
        cmd, data = remote.recv()
		
        if cmd == 'step':
            ob, reward, done, info = env.step( data )
            if done:
                ob = env.reset()
            remote.send( ( ob, reward, done, info ) )

        elif cmd == 'render':
            remote.send(env.render())

        elif cmd == 'close':
            remote.close()
            break

        elif cmd == 'sample':
            ac = env.action_space.sample()
            remote.send( ( ac ) )
            break

        elif cmd == 'reset':
            ob = env.reset()
            remote.send( ( ob ) )
            break

        else:
            raise "NotImplentedError"


class MultiEnv:

    def __init__(self, envs, maxlen, gm_m, gm_n):
        
        self.envs = envs
        self.maxlen = maxlen
        self.gm_m = gm_m
        self.gm_n = gm_n
        for env in envs:
            env['env'] = gym.make( env['name'] )            
            try:
                env['state_dim'] = env['env'].observation_space.shape[0]
            except:
                env['state_dim'] = env['env'].observation_space.n
            env['action_dim'] = env['env'].action_space.n
            env['values'] = namedtuple( 'transition', [ 'state', 'p_state', 'past_state', 'past_memory', 'g_memory', 'g_usage', 'g_l', 'g_wp', 'g_wr', 'g_ww' ] )
            self.reset_all( env )

    def reset_all(self, env):

        self.reset_env( env )

        # state values
        env['values'].p_state = np.zeros_like( env['values'].state )

        # global memory values
        env['values'].g_memory = np.zeros( [ self.gm_m, self.gm_n ], dtype = np.float32 )
        env['values'].g_usage  = np.zeros( [ self.gm_m, 1 ],         dtype = np.float32 )
        env['values'].g_l      = np.zeros( [ self.gm_m, self.gm_m ], dtype = np.float32 )
        env['values'].g_wp     = np.zeros( [ self.gm_m, 1 ],         dtype = np.float32 )
        env['values'].g_wr     = np.zeros( [ self.gm_m, 3 ],         dtype = np.float32 )
        env['values'].g_ww     = np.zeros( [ self.gm_m, 1 ],         dtype = np.float32 )

    def reset_env(self, env):

        env['to_done'] = False
        env['step'] = 0

        # buffer to store sequences of events
        env['buffer_s']  = list( np.zeros( [ self.maxlen, env['state_dim'] ] ) )
        env['buffer_s_'] = list( np.zeros( [ self.maxlen, env['state_dim'] ] ) )
        env['buffer_a']  = list( np.zeros( [ self.maxlen, ] ) )
        env['buffer_r']  = list( np.zeros( [ self.maxlen, ] ) )
        env['buffer_d']  = list( np.zeros( [ self.maxlen, ] ) )
        
        # state values
        env['values'].state = env['st_func']( env['env'].reset() )
        env['values'].past_state = []

    def step_bufeer(self, env, ob, ac, rew, done):

        if env['step'] >= self.maxlen:

            env['buffer_s'].pop(0)
            env['buffer_s_'].pop(0)
            env['buffer_a'].pop(0)
            env['buffer_r'].pop(0)
            env['buffer_d'].pop(0)

            env['buffer_s'].append( env['values'].state )
            env['buffer_s_'].append( ob )
            env['buffer_a'].append( ac )
            env['buffer_r'].append( rew )
            env['buffer_d'].append( done )

        else:

            env['buffer_s'][env['step']] = env['values'].state
            env['buffer_s_'][env['step']] = ob
            env['buffer_a'][env['step']] = ac
            env['buffer_r'][env['step']] = rew
            env['buffer_d'][env['step']] = done

    def step(self, actions):
        
        for env in self.envs:
                        
            a = actions[env['id']]
                
            ob, rew, done, info = env['env'].step( env['ac_func']( a[0] ) )
            rew = env['rw_func']( rew, env['step'], done )
            ob = env['st_func']( ob )

            self.step_bufeer( env, ob, a[0], rew, done )

            # store preview state
            env['values'].p_state = np.copy( env['values'].state )
            
            # change current state to next state and update recurrent params
            if done:
                env['to_done'] = True
                env['values'].past_state = []
            else:
                self.control_pstate( a[1], env['values'].past_state )
                env['values'].state = ob

            env['step'] += 1
            	
    def update(self):
        for env in self.envs:
            if env['to_done']:
                self.reset_env(env)
    
    def seed(self, seed):
        for env in self.envs:
            for e in env['env']:
                e.seed( seed )
    
    def control_pstate(self, pstate, storage):
        if len( storage ) >= ( self.maxlen - 1 ):
                storage.pop(0)
        storage.append( pstate )

    def reset(self):
        for env in self.envs:
            self.reset_all(env)

    def render(self):
        for env in self.envs:
            env['env'].render()


class SubprocVecEnv():

    def __init__(self, env_fns):
        		
        self.waiting = False
        self.closed = False
		
        no_of_envs = len( env_fns )
		
        self.remotes, self.work_remotes = zip( *[ Pipe() for _ in range( no_of_envs ) ] )
        self.ps = []

        for wrk, rem, fn in zip(self.work_remotes, self.remotes, env_fns):
            proc = Process( target = worker, args = ( wrk, rem, CloudpickleWrapper( fn ) ) )
            self.ps.append( proc )

        for p in self.ps:
            p.daemon = True
            p.start()

        for remote in self.work_remotes:
            remote.close()
    
    def step_async(self, actions):

        if self.waiting:
            raise 'AlreadySteppingError'
        self.waiting = True

        for remote, action in zip( self.remotes, actions ):
            remote.send( ( 'step', action ) )
	
    def step_wait(self):
        
        if not self.waiting:
            raise 'NotSteppingError'

        self.waiting = False

        results = [ remote.recv() for remote in self.remotes ]
        obs, rews, dones, infos = zip( *results )
        return np.stack( obs ), np.stack( rews ), np.stack( dones ), infos
	
    def step(self, actions):
        
        self.step_async( actions )
        return self.step_wait()
	
    def reset(self):

        for remote in self.remotes:
            remote.send( ( 'reset', None ) )

        return [ remote.recv() for remote in self.remotes ]

    def sample(self):

        for remote in self.remotes:
            remote.send( ( 'sample', None ) )

        return [ remote.recv() for remote in self.remotes ]
	
    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send( ( 'close', None ) )
        for p in self.ps:
            p.join()
        self.closed = True


class CloudpickleWrapper(object):
	
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        return cloudpickle.dumps( self.x )

    def __setstate__(self, ob):
        self.x = pickle.loads( ob )
	
    def __call__(self):
        return self.x()

class Environment(Process):

    def __init__(self, env_idx, child_conn, env_name, state_size, action_size, visualize=False, warper=None):
        
        super(Environment, self).__init__()
        self.env = gym.make(env_name)
        if not warper is None:
            self.env = warper( self.env )
        self.is_render = visualize
        self.env_idx = env_idx
        self.child_conn = child_conn
        self.state_size = state_size
        self.action_size = action_size

    def run(self):

        super(Environment, self).run()
        state = self.env.reset()
        state = np.reshape(state, [1, self.state_size])
        self.child_conn.send(state)
        
        while True:
            action = self.child_conn.recv()
            if self.is_render and self.env_idx == 0:
                self.env.render()

            state, reward, done, info = self.env.step(action)
            state = np.reshape(state, [1, self.state_size])
            # reward *= 100

            if done:
                state = self.env.reset()
                state = np.reshape(state, [1, self.state_size])

            self.child_conn.send([state, reward, done, info])


class VisionEnvironment(Process):

    def __init__(self, env_idx, child_conn, env_name, action_size, visualize=False):
        
        super(VisionEnvironment, self).__init__()
        self.env = ImageEnv( gym.make(env_name) )
        self.is_render = visualize
        self.env_idx = env_idx
        self.child_conn = child_conn
        self.state_size = list( self.env.observation_space.shape )
        self.action_size = action_size

    def run(self):

        super(VisionEnvironment, self).run()
        si, sv = self.env.reset()
        si = np.reshape(si, [1] + list( self.state_size ) )
        self.child_conn.send((si, sv))
        
        while True:
            action = self.child_conn.recv()
            state, reward, done, info = self.env.step(action)

            si = np.reshape(state[0], [1] + list( self.state_size ) )
            sv = state[1]
            # reward *= 100

            if done:
                si, sv = self.env.reset()
                si = np.reshape(si, [1] + list( self.state_size ) )

            self.child_conn.send([(si, sv), reward, done, info])


class ImageEnv(gym.Wrapper):
    
    def __init__(self, env):

        gym.Wrapper.__init__(self, env)
        self.observation_space = spaces.Box( low=0, high=255, shape = ( 128 , 128, 3 ), dtype = env.observation_space.dtype )
        env.viewer = None

    def reset(self):
        ob_v = self.env.reset()
        ob = self.env.render( mode = "rgb_array" )
        ob = ( cv.resize( ob, ( 128, 128 ) ) / 256.0 ).astype( np.float32 )
        return ( ob, ob_v )

    def step(self, action):
        ob_v, reward, done, info = self.env.step(action)
        ob1 = self.env.render( mode = "rgb_array" )
        ob2 = ( cv.resize( ob1, ( 128, 128 ) ) / 256.0 ).astype( np.float32 )
        return ( ob2, ob_v ), reward, done, info


class miniGridEnv(gym.Wrapper):
    
    def __init__(self, env):

        gym.Wrapper.__init__(self, env)
        env.viewer = None
        self.observation_space = spaces.Box( low=0, high=255, shape = [ 147 ] )

    def reset(self):
        ob = self.env.reset()
        return ob['image'].astype(np.float32).flatten()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        return ob['image'].astype(np.float32).flatten(), reward, done, info


class floatEnv(gym.Wrapper):
    
    def __init__(self, env):

        gym.Wrapper.__init__(self, env)
        env.viewer = None

    def reset(self):
        ob = self.env.reset()
        return ob.astype(np.float32)

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        return ob.astype(np.float32), reward, done, info