import random
from matplotlib.animation import AVConvBase

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from collections import namedtuple, deque
import pandas as pd

class ReplayContinuosBuffer:
  
    def __init__(self, obs_dim, act_dim, size):
        self.transitions = namedtuple('transition', ['s', 'a', 'r', 'sp', 'it'])
        # (this_state, this_action, this_reward, next_state, this_is_terminal)

        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros([size, 1], dtype=np.float32)
        self.done_buf = np.zeros([size, 1], dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.size_list = range(self.size)

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        prev_size = self.size
        self.size = min(self.size+1, self.max_size)
        if prev_size < self.size:
            self.size_list = range(self.size)

    def sample_batch(self, batch_size=32):
        idxs = random.sample(self.size_list, batch_size)
        self.transitions.s = tf.convert_to_tensor(self.obs1_buf[idxs])
        self.transitions.a = tf.convert_to_tensor(self.acts_buf[idxs])
        self.transitions.r = tf.convert_to_tensor(self.rews_buf[idxs])
        self.transitions.sp = tf.convert_to_tensor(self.obs2_buf[idxs])
        self.transitions.it = tf.convert_to_tensor(self.done_buf[idxs])
        return self.transitions


class ReplayDiscreteBuffer(object):

    def __init__(self, obs_dim, size):
        self.transitions = namedtuple('transition', ['s', 'a', 'r', 'sp', 'it'])
        # (this_state, this_action, this_reward, next_state, this_is_terminal)

        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, ], dtype=np.float32)
        self.rews_buf = np.zeros([size, ], dtype=np.float32)
        self.done_buf = np.zeros([size, ], dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.size_list = range(self.size)

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        prev_size = self.size
        self.size = min(self.size+1, self.max_size)
        if prev_size < self.size:
            self.size_list = range(self.size)

    def storem(self, obss, acts, rews, next_obss, dones):
        for obs, act, rew, next_obs, done in zip( obss, acts, rews, next_obss, dones ):
            self.store( obs, act, rew, next_obs, done )

    def sample_batch(self, batch_size=32):
        idxs = random.sample(self.size_list, batch_size)
        self.transitions.s = tf.convert_to_tensor(self.obs1_buf[idxs])
        self.transitions.a = tf.convert_to_tensor(self.acts_buf[idxs], dtype = tf.int32)
        self.transitions.r = tf.convert_to_tensor(self.rews_buf[idxs])
        self.transitions.sp = tf.convert_to_tensor(self.obs2_buf[idxs])
        self.transitions.it = tf.convert_to_tensor(self.done_buf[idxs])
        return self.transitions

    def __len__(self):
        return self.size


class ReplayDiscreteBufferPandas(object):

    def __init__(self, max_size):

        self.transitions = namedtuple('transition', ['s', 'a', 'r', 'sp', 'it'])

        di = {
            'state': [],
            'state_': [],
            'acts': [],
            'rewards': [],
            'dones': []
        }
        self.df = pd.DataFrame.from_dict( di )
        self.max_size = max_size

    def store(self, obs, act, rew, next_obs, done):

        row = { 'state': obs, 'state_': next_obs, 'acts': act, 'rewards': rew, 'dones': done}

        if len(self) >= ( self.max_size + ( 0.3 * self.max_size ) ):
            self.df = self.df.iloc[int( 0.3 * self.max_size ):]
            
        self.df = self.df.append( row, ignore_index = True, sort = False )
        
    def storem(self, obss, acts, rews, next_obss, dones):
        for obs, act, rew, next_obs, done in zip( obss, acts, rews, next_obss, dones ):
            self.store( obs, act, rew, next_obs, done )

    def sample_batch(self, batch_size=32):

        batch = self.df.sample( n = batch_size ).values
        
        self.transitions.s  = tf.convert_to_tensor( np.asarray( list( batch[:,0] ) ) )
        self.transitions.sp = tf.convert_to_tensor( np.asarray( list( batch[:,1] ) ) )
        self.transitions.a  = tf.convert_to_tensor( np.asarray( list( batch[:,2] ) ), dtype = tf.int32)
        self.transitions.r  = tf.convert_to_tensor( np.asarray( list( batch[:,3] ) ) )
        self.transitions.it = tf.convert_to_tensor( np.asarray( list( batch[:,4] ) ) )
        return self.transitions

    def __len__(self):
        return self.df.count()[0]



class ReplayDiscreteNTMBuffer(object):

    def __init__(self, obs_dim, size):
        self.transitions = namedtuple('transition', ['s', 'a', 'r', 'sp', 'it', 's_', 'M', 'MP', 'w', 'av', 'rd'])
        # (this_state, this_action, this_reward, next_state, this_is_terminal)

        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, ], dtype=np.float32)
        self.rews_buf = np.zeros([size, ], dtype=np.float32)
        self.done_buf = np.zeros([size, ], dtype=np.float32)
        
        self.obs3_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.memory_buf = np.zeros([size, 128, 20], dtype=np.float32)
        self.pmemory_buf = np.zeros([size, 128, 20], dtype=np.float32)
        self.wgh_buf = np.zeros([size, 128], dtype=np.float32)
        self.acval_buf = np.zeros([size, 4], dtype=np.float32)
        self.rd_buf = np.zeros([size, 16], dtype=np.float32)

        self.ptr, self.size, self.max_size = 0, 0, size
        self.size_list = range(self.size)

    def store(self, obs, act, rew, next_obs, done, _state, M_t, p_M_t, w_t_1, av, rd):
        
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done

        self.obs3_buf[self.ptr] = _state
        self.memory_buf[self.ptr] = M_t
        self.pmemory_buf[self.ptr] = p_M_t
        self.wgh_buf[self.ptr] = w_t_1
        self.acval_buf[self.ptr] = av
        self.rd_buf[self.ptr] = rd
        
        self.ptr = (self.ptr+1) % self.max_size
        prev_size = self.size
        self.size = min(self.size+1, self.max_size)
        if prev_size < self.size:
            self.size_list = range(self.size)

    def sample_batch(self, batch_size=32):
        
        idxs = random.sample(self.size_list, batch_size)
        
        self.transitions.s = tf.convert_to_tensor(self.obs1_buf[idxs])
        self.transitions.a = tf.convert_to_tensor(self.acts_buf[idxs], dtype = tf.int32)
        self.transitions.r = tf.convert_to_tensor(self.rews_buf[idxs])
        self.transitions.sp = tf.convert_to_tensor(self.obs2_buf[idxs])
        self.transitions.it = tf.convert_to_tensor(self.done_buf[idxs])

        self.transitions.s_ = tf.convert_to_tensor(self.obs3_buf[idxs])
        self.transitions.M = tf.convert_to_tensor(self.memory_buf[idxs])
        self.transitions.MP = tf.convert_to_tensor(self.pmemory_buf[idxs])
        self.transitions.w = tf.convert_to_tensor(self.wgh_buf[idxs])
        self.transitions.av = tf.convert_to_tensor(self.acval_buf[idxs])
        self.transitions.rd = tf.convert_to_tensor(self.rd_buf[idxs])
        
        return self.transitions

    def __len__(self):
        return self.size


class ReplayDiscreteDNCBuffer(object):

    def __init__(self, obs_dim, size, m, n):
        
        self.transitions = namedtuple('transition', ['s', 'a', 'r', 'sp', 'it', 's_', 'av', 'rd', 'm', 'u', 'l', 'wp', 'wr', 'ww'])

        self.obs_dim = obs_dim
        self.sz = size
        self.m = m
        self.n = n

        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, ], dtype=np.float32)
        self.rews_buf = np.zeros([size, ], dtype=np.float32)
        self.done_buf = np.zeros([size, ], dtype=np.float32)
        
        self.obs3_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acval_buf = np.zeros([size, 4], dtype=np.float32)
        self.rd_buf = np.zeros([size, 16], dtype=np.float32)

        self.m_buf = np.zeros([size, m, n], dtype=np.float32)
        self.u_buf = np.zeros([size, m, 1], dtype=np.float32)
        self.l_buf = np.zeros([size, m, m], dtype=np.float32)
        self.wp_buf = np.zeros([size, m, 1], dtype=np.float32)
        self.wr_buf = np.zeros([size, m, 3], dtype=np.float32)
        self.ww_buf = np.zeros([size, m, 1], dtype=np.float32)

        self.ptr, self.size, self.max_size = 0, 0, size
        self.size_list = range(self.size)

    def store(self, obs, act, rew, next_obs, done, _state, av, rd, M, usage, L, W_precedence, W_read, W_write):
        
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done

        self.obs3_buf[self.ptr] = _state
        self.acval_buf[self.ptr] = av
        self.rd_buf[self.ptr] = rd

        self.m_buf[self.ptr] = M
        self.u_buf[self.ptr] = usage
        self.l_buf[self.ptr] = L
        self.wp_buf[self.ptr] = W_precedence
        self.wr_buf[self.ptr] = W_read
        self.ww_buf[self.ptr] = W_write        
        
        self.ptr = (self.ptr+1) % self.max_size
        prev_size = self.size
        self.size = min(self.size+1, self.max_size)
        if prev_size < self.size:
            self.size_list = range(self.size)

    def sample_batch(self, batch_size=32):
        
        idxs = [ x for x in range( self.ptr ) ]
        
        self.transitions.s = tf.convert_to_tensor(self.obs1_buf[idxs])
        self.transitions.a = tf.convert_to_tensor(self.acts_buf[idxs], dtype = tf.int32)
        self.transitions.r = tf.convert_to_tensor(self.rews_buf[idxs])
        self.transitions.sp = tf.convert_to_tensor(self.obs2_buf[idxs])
        self.transitions.it = tf.convert_to_tensor(self.done_buf[idxs])

        self.transitions.s_ = tf.convert_to_tensor(self.obs3_buf[idxs])
        self.transitions.av = tf.convert_to_tensor(self.acval_buf[idxs])
        self.transitions.rd = tf.convert_to_tensor(self.rd_buf[idxs])

        self.transitions.m = tf.convert_to_tensor(self.m_buf[idxs])
        self.transitions.u = tf.convert_to_tensor(self.u_buf[idxs])
        self.transitions.l = tf.convert_to_tensor(self.l_buf[idxs])
        self.transitions.wp = tf.convert_to_tensor(self.wp_buf[idxs])
        self.transitions.wr = tf.convert_to_tensor(self.wr_buf[idxs])
        self.transitions.ww = tf.convert_to_tensor(self.ww_buf[idxs])
                
        return self.transitions

    def reset(self):

        del self.obs1_buf
        del self.obs2_buf
        del self.acts_buf
        del self.rews_buf
        del self.done_buf
        del self.obs3_buf
        del self.acval_buf
        del self.rd_buf
        del self.m_buf
        del self.u_buf
        del self.l_buf
        del self.wp_buf
        del self.wr_buf
        del self.ww_buf

        self.obs1_buf = np.zeros([self.sz, self.obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([self.sz, self.obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([self.sz, ], dtype=np.float32)
        self.rews_buf = np.zeros([self.sz, ], dtype=np.float32)
        self.done_buf = np.zeros([self.sz, ], dtype=np.float32)
        
        self.obs3_buf = np.zeros([self.sz, self.obs_dim], dtype=np.float32)
        self.acval_buf = np.zeros([self.sz, 4], dtype=np.float32)
        self.rd_buf = np.zeros([self.sz, 16], dtype=np.float32)

        self.m_buf = np.zeros([self.sz, self.m, self.n], dtype=np.float32)
        self.u_buf = np.zeros([self.sz, self.m, 1], dtype=np.float32)
        self.l_buf = np.zeros([self.sz, self.m, self.m], dtype=np.float32)
        self.wp_buf = np.zeros([self.sz, self.m, 1], dtype=np.float32)
        self.wr_buf = np.zeros([self.sz, self.m, 3], dtype=np.float32)
        self.ww_buf = np.zeros([self.sz, self.m, 1], dtype=np.float32)

        self.ptr, self.size, self.max_size = 0, 0, self.sz
        self.size_list = range(self.size)

    def __len__(self):
        return self.size


class ReplaySequenceDiscreteDNCBuffer(object):

    def __init__(self, obs_dim, s, size, m, n, h):
        
        self.transitions = namedtuple('transition', ['s', 'a', 'r', 'sp', 'it', 's_', 'tw', 'm', 'u', 'l', 'wp', 'wr', 'ww', 'step'])

        self.obs_dim = obs_dim
        self.sz = size
        self.m = m
        self.n = n

        self.obs1_buf = np.zeros([size, s, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, s, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, s, ], dtype=np.float32)
        self.rews_buf = np.zeros([size, s, ], dtype=np.float32)
        self.done_buf = np.zeros([size, s, ], dtype=np.float32)
        self.tw_buf = np.zeros([size, s, m*2], dtype=np.float32)
        
        self.obs3_buf = np.zeros([size, s, obs_dim], dtype=np.float32)

        self.m_buf = np.zeros([size, m, n], dtype=np.float32)
        self.u_buf = np.zeros([size, m, 1], dtype=np.float32)
        self.l_buf = np.zeros([size, m, m], dtype=np.float32)
        self.wp_buf = np.zeros([size, m, 1], dtype=np.float32)
        self.wr_buf = np.zeros([size, m, h], dtype=np.float32)
        self.ww_buf = np.zeros([size, m, 1], dtype=np.float32)


        self.step_buf = np.zeros([size, 1], dtype=np.float32)
        
        self.ptr, self.size, self.max_size = 0, 0, size
        self.size_list = range(self.size)

    def store(self, obs, act, rew, next_obs, done, _state, tw, M, usage, L, W_precedence, W_read, W_write, step):
        
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.tw_buf[self.ptr] = tw

        self.obs3_buf[self.ptr] = _state
        self.step_buf[self.ptr] = step

        self.m_buf[self.ptr] = M
        self.u_buf[self.ptr] = usage
        self.l_buf[self.ptr] = L
        self.wp_buf[self.ptr] = W_precedence
        self.wr_buf[self.ptr] = W_read
        self.ww_buf[self.ptr] = W_write
        
        self.ptr = (self.ptr+1) % self.max_size
        prev_size = self.size
        self.size = min(self.size+1, self.max_size)
        if prev_size < self.size:
            self.size_list = range(self.size)

    def sample_batch(self, batch_size=32):
        
        idxs = random.sample( self.size_list, batch_size )
        
        self.transitions.s = tf.convert_to_tensor(self.obs1_buf[idxs])
        self.transitions.a = tf.convert_to_tensor(self.acts_buf[idxs], dtype = tf.int32)
        self.transitions.r = tf.convert_to_tensor(self.rews_buf[idxs])
        self.transitions.sp = tf.convert_to_tensor(self.obs2_buf[idxs])
        self.transitions.it = tf.convert_to_tensor(self.done_buf[idxs])
        self.transitions.tw = tf.convert_to_tensor(self.tw_buf[idxs])

        self.transitions.s_ = tf.convert_to_tensor(self.obs3_buf[idxs])

        self.transitions.m = tf.convert_to_tensor(self.m_buf[idxs])
        self.transitions.u = tf.convert_to_tensor(self.u_buf[idxs])
        self.transitions.l = tf.convert_to_tensor(self.l_buf[idxs])
        self.transitions.wp = tf.convert_to_tensor(self.wp_buf[idxs])
        self.transitions.wr = tf.convert_to_tensor(self.wr_buf[idxs])
        self.transitions.ww = tf.convert_to_tensor(self.ww_buf[idxs])

        self.transitions.step = tf.convert_to_tensor(self.step_buf[idxs])
                
        return self.transitions

    def __len__(self):
        return self.size


class ReplayDiscreteSequenceBuffer(object):

    def __init__(self, obs_dim, seq_dim, size):
        self.transitions = namedtuple('transition', ['s', 'a', 'r', 'sp', 'it', 'p'])

        self.obs1_buf = np.zeros([size, seq_dim, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, seq_dim, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, seq_dim, ], dtype=np.float32)
        self.rews_buf = np.zeros([size, seq_dim, ], dtype=np.float32)
        self.done_buf = np.zeros([size, seq_dim, ], dtype=np.float32)
        self.prev_buf = np.zeros([size, 16 ], dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.size_list = range(self.size)

    def calc_multistep_return(self, n_step_buffer):
        Return = 0
        R = []
        v = n_step_buffer == 0.0
        ln = len( n_step_buffer )
        init = np.argmin( v ) if np.add.reduce( v ) < len( n_step_buffer ) else len( n_step_buffer )
        for idx in range( init, ln ):
            Return += n_step_buffer[idx] # 0.99**idx * n_step_buffer[idx]
            R.append( Return / ( ( idx - init ) + 1 ) )
        return np.array( ( init * [ 0 ] ) + R )

    def store(self, obs, act, rew, next_obs, done, prev):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew + 0.1 * self.calc_multistep_return( rew )
        self.done_buf[self.ptr] = done
        self.prev_buf[self.ptr] = prev
        self.ptr = (self.ptr+1) % self.max_size
        prev_size = self.size
        self.size = min(self.size+1, self.max_size)
        if prev_size < self.size:
            self.size_list = range(self.size)

    def sample_batch(self, batch_size=32):
        idxs = random.sample(self.size_list, batch_size)
        self.transitions.s = tf.convert_to_tensor(self.obs1_buf[idxs])
        self.transitions.a = tf.convert_to_tensor(self.acts_buf[idxs], dtype = tf.int32)
        self.transitions.r = tf.convert_to_tensor(self.rews_buf[idxs])
        self.transitions.sp = tf.convert_to_tensor(self.obs2_buf[idxs])
        self.transitions.it = tf.convert_to_tensor(self.done_buf[idxs])
        self.transitions.pr = tf.convert_to_tensor(self.prev_buf[idxs])
        return self.transitions

    def __len__(self):
        return self.size


class ReplayDiscreteSTransformerBuffer(object):

    def __init__(self, obs_dim, seq_dim, size):
        self.transitions = namedtuple('transition', ['s', 'a', 'r', 'sp', 'it', 'm'])

        self.obs1_buf = np.zeros([size, seq_dim, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, seq_dim, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, seq_dim, ], dtype=np.float32)
        self.rews_buf = np.zeros([size, seq_dim, ], dtype=np.float32)
        self.done_buf = np.zeros([size, seq_dim, ], dtype=np.float32)
        self.mask_buf = np.zeros([size, seq_dim ], dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.size_list = range(self.size)

    def calc_multistep_return(self, n_step_buffer, mask):
        Return = 0
        R = []
        ln = np.argmin( mask )
        init = 0
        for idx in range( init, ln ):
            Return += n_step_buffer[idx]
            R.append( Return / ( ( idx - init ) + 1 ) )
        return np.array( R + ( ( len( mask ) - ln ) * [ 0 ] ) )

    def store(self, obs, act, rew, next_obs, done, mask):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew + 0.1 * self.calc_multistep_return( rew, mask )
        self.done_buf[self.ptr] = done
        self.mask_buf[self.ptr] = mask
        self.ptr = (self.ptr+1) % self.max_size
        prev_size = self.size
        self.size = min(self.size+1, self.max_size)
        if prev_size < self.size:
            self.size_list = range(self.size)

    def storem(self, obss, acts, rews, next_obss, dones):
        for obs, act, rew, next_obs, done in zip( obss, acts, rews, next_obss, dones ):
            self.store( obs[0], act[0], rew, next_obs, done )

    def sample_batch(self, batch_size=32):
        idxs = random.sample(self.size_list, batch_size)
        self.transitions.s = tf.convert_to_tensor(self.obs1_buf[idxs])
        self.transitions.a = tf.convert_to_tensor(self.acts_buf[idxs], dtype = tf.int32)
        self.transitions.r = tf.convert_to_tensor(self.rews_buf[idxs])
        self.transitions.sp = tf.convert_to_tensor(self.obs2_buf[idxs])
        self.transitions.it = tf.convert_to_tensor(self.done_buf[idxs])
        self.transitions.m = tf.convert_to_tensor(self.mask_buf[idxs])
        return self.transitions

    def __len__(self):
        return self.size


class PrioritizedReplay(object):
    """
    Proportional Prioritization
    """
    def __init__(self, capacity, batch_size, seed, gamma=0.99, tau=1, n_step=1, alpha=0.6, beta_start = 0.4, beta_frames=100000, parallel_env=4):
        
        self.transitions = namedtuple('transition', ['s', 'a', 'r', 'sp', 'it'])
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1 #for beta calculation
        self.batch_size = batch_size
        self.capacity   = capacity
        self.buffer     = []
        self.pos        = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.seed = np.random.seed(seed)
        self.n_step = n_step
        self.parallel_env = parallel_env
        self.n_step_buffer = [deque(maxlen=self.n_step) for i in range(parallel_env)]
        self.iter_ = 0
        self.gamma = gamma
        self.tau = tau

    def calc_multistep_return(self,n_step_buffer):
        Return = 0
        for idx in range(self.n_step):
            Return += self.gamma**idx * n_step_buffer[idx][2]
        return n_step_buffer[0][0], n_step_buffer[0][1], Return, n_step_buffer[-1][3], n_step_buffer[-1][4]

    def calc_nstep(self,n_step_buffer):
        Return = 0
        for idx in range( self.n_step ):
            Return = Return * self.gamma * self.tau
            Return = ( Return + n_step_buffer[idx][2] + self.gamma * n_step_buffer[-1][2] ) * ( 1 - n_step_buffer[idx][4] ) #- n_step_buffer[idx][5]
        
        return n_step_buffer[0][0], n_step_buffer[0][1], Return, n_step_buffer[-1][3], n_step_buffer[-1][4]

    def gae(self,n_step_buffer):
        Return = 0
        R = []
        for idx in reversed( range( len( n_step_buffer ) ) ):
            Return = Return * self.gamma * self.tau
            Return = Return + n_step_buffer[idx][2] + self.gamma * n_step_buffer[-1][2] #- n_step_buffer[idx][5]
            R.append( Return / ( len( n_step_buffer ) + 1 - idx ) )
        return R[::-1]
    
    def beta_by_frame(self, frame_idx):
        """
        Linearly increases beta from beta_start to 1 over time from 1 to beta_frames.
        
        3.4 ANNEALING THE BIAS (Paper: PER)
        We therefore exploit the flexibility of annealing the amount of importance-sampling
        correction over time, by defining a schedule on the exponent 
        that reaches 1 only at the end of learning. In practice, we linearly anneal from its initial value 0 to 1
        """
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)
    
    def store_gae(self, state, action, reward, next_state, done):
        if self.iter_ == self.parallel_env:
            self.iter_ = 0
        assert state.ndim == next_state.ndim
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        
        # gae calc
        self.n_step_buffer[self.iter_].append((state, action, reward, next_state, done))
        if done:
            
            rewards = self.gae( self.n_step_buffer[self.iter_] )

            for v, r in zip( self.n_step_buffer[self.iter_], rewards ):

                max_prio = self.priorities.max() if self.buffer else 1.0 # gives max priority if buffer is not empty else 1

                if len(self.buffer) < self.capacity:
                    self.buffer.append((v[0], v[1], r, v[3], v[4]))
                else:
                    # puts the new data on the position of the oldes since it circles via pos variable
                    # since if len(buffer) == capacity -> pos == 0 -> oldest memory (at least for the first round?) 
                    self.buffer[self.pos] = (v[0], v[1], r, v[3], v[4])

                self.priorities[self.pos] = max_prio
                self.pos = (self.pos + 1) % self.capacity # lets the pos circle in the ranges of capacity if pos+1 > cap --> new posi = 0
            
            self.n_step_buffer[self.iter_] = []
            self.iter_ += 1

    def store_nsteps(self, state, action, reward, next_state, done):
        if self.iter_ == self.parallel_env:
            self.iter_ = 0
        assert state.ndim == next_state.ndim
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        
        #n_step calc
        self.n_step_buffer[self.iter_].append((state, action, reward, next_state, done))
        if len(self.n_step_buffer[self.iter_]) == self.n_step:
            state, action, reward, next_state, done = self.calc_nstep(self.n_step_buffer[self.iter_])

        max_prio = self.priorities.max() if self.buffer else 1.0 # gives max priority if buffer is not empty else 1
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            # puts the new data on the position of the oldes since it circles via pos variable
            # since if len(buffer) == capacity -> pos == 0 -> oldest memory (at least for the first round?) 
            self.buffer[self.pos] = (state, action, reward, next_state, done)
                 
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity # lets the pos circle in the ranges of capacity if pos+1 > cap --> new posi = 0
        self.iter_ += 1

    def store(self, state, action, reward, next_state, done):
        if self.iter_ == self.parallel_env:
            self.iter_ = 0
        assert state.ndim == next_state.ndim
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        
        # n_step calc
        self.n_step_buffer[self.iter_].append((state, action, reward, next_state, done))
        if len(self.n_step_buffer[self.iter_]) == self.n_step:
            state, action, reward, next_state, done = self.calc_multistep_return(self.n_step_buffer[self.iter_])

        max_prio = self.priorities.max() if self.buffer else 1.0 # gives max priority if buffer is not empty else 1

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            # puts the new data on the position of the oldes since it circles via pos variable
            # since if len(buffer) == capacity -> pos == 0 -> oldest memory (at least for the first round?) 
            self.buffer[self.pos] = (state, action, reward, next_state, done) 
        
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity # lets the pos circle in the ranges of capacity if pos+1 > cap --> new posi = 0
        self.iter_ += 1

    def sample_batch(self, batch):
        N = len(self.buffer)
        if N == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
            
        # calc P = p^a/sum(p^a)
        probs  = prios ** self.alpha
        P = probs/probs.sum()
        
        #gets the indices depending on the probability p
        indices = np.random.choice(N, self.batch_size, p=P) 
        samples = [self.buffer[idx] for idx in indices]
        
        beta = self.beta_by_frame(self.frame)
        self.frame+=1
                
        #Compute importance-sampling weight
        weights  = (N * P[indices]) ** (-beta)
        # normalize weights
        weights /= weights.max() 
        weights  = np.array(weights, dtype=np.float32) 
        
        states, actions, rewards, next_states, dones = zip(*samples)

        self.transitions.s = tf.convert_to_tensor(np.concatenate(states))
        self.transitions.a = tf.convert_to_tensor(actions)
        self.transitions.r = tf.convert_to_tensor(rewards)
        self.transitions.sp = tf.convert_to_tensor(np.concatenate(next_states))
        self.transitions.it = tf.convert_to_tensor(dones)
        
        return self.transitions, indices, tf.convert_to_tensor( weights )
    
    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio 

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayExtra(object):
    """
    Proportional Prioritization
    """
    def __init__(self, capacity, batch_size, seed, gamma=0.99, n_step=1, alpha=0.6, beta_start = 0.4, beta_frames=100000, parallel_env=4):
        
        self.transitions = namedtuple('transition', ['s', 'a', 'r', 'sp', 'it', 'sts'])
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1 #for beta calculation
        self.batch_size = batch_size
        self.capacity   = capacity
        self.buffer     = []
        self.pos        = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.seed = np.random.seed(seed)
        self.n_step = n_step
        self.parallel_env = parallel_env
        self.n_step_buffer = [deque(maxlen=self.n_step) for i in range(parallel_env)]
        self.iter_ = 0
        self.gamma = gamma

    def calc_multistep_return(self,n_step_buffer):
        Return = 0
        for idx in range(self.n_step):
            Return += self.gamma**idx * n_step_buffer[idx][2]
        
        return n_step_buffer[0][0], n_step_buffer[0][1], Return, n_step_buffer[-1][3], n_step_buffer[-1][4], n_step_buffer[-1][5]
    
    def beta_by_frame(self, frame_idx):
        """
        Linearly increases beta from beta_start to 1 over time from 1 to beta_frames.
        
        3.4 ANNEALING THE BIAS (Paper: PER)
        We therefore exploit the flexibility of annealing the amount of importance-sampling
        correction over time, by defining a schedule on the exponent 
        that reaches 1 only at the end of learning. In practice, we linearly anneal from its initial value 0 to 1
        """
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)
    
    def store(self, state, action, reward, next_state, done, st):
        if self.iter_ == self.parallel_env:
            self.iter_ = 0
        assert state.ndim == next_state.ndim
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        st = np.expand_dims(st, 0)
        
        # n_step calc
        self.n_step_buffer[self.iter_].append((state, action, reward, next_state, done, st))
        if len(self.n_step_buffer[self.iter_]) == self.n_step:
            state, action, reward, next_state, done, st = self.calc_multistep_return(self.n_step_buffer[self.iter_])

        max_prio = self.priorities.max() if self.buffer else 1.0 # gives max priority if buffer is not empty else 1

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done, st))
        else:
            # puts the new data on the position of the oldes since it circles via pos variable
            # since if len(buffer) == capacity -> pos == 0 -> oldest memory (at least for the first round?) 
            self.buffer[self.pos] = (state, action, reward, next_state, done, st) 
        
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity # lets the pos circle in the ranges of capacity if pos+1 > cap --> new posi = 0
        self.iter_ += 1
        
    def sample_batch(self, batch):
        N = len(self.buffer)
        if N == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
            
        # calc P = p^a/sum(p^a)
        probs  = prios ** self.alpha
        P = probs/probs.sum()
        
        #gets the indices depending on the probability p
        indices = np.random.choice(N, self.batch_size, p=P) 
        samples = [self.buffer[idx] for idx in indices]
        
        beta = self.beta_by_frame(self.frame)
        self.frame+=1
                
        #Compute importance-sampling weight
        weights  = (N * P[indices]) ** (-beta)
        # normalize weights
        weights /= weights.max() 
        weights  = np.array(weights, dtype=np.float32) 
        
        states, actions, rewards, next_states, dones, sts = zip(*samples)

        self.transitions.s = tf.convert_to_tensor(np.concatenate(states))
        self.transitions.a = tf.convert_to_tensor(actions)
        self.transitions.r = tf.convert_to_tensor(rewards)
        self.transitions.sp = tf.convert_to_tensor(np.concatenate(next_states))
        self.transitions.it = tf.convert_to_tensor(dones)
        self.transitions.sts = tf.convert_to_tensor(sts)
        
        return self.transitions, indices, tf.convert_to_tensor( weights )
    
    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio 

    def __len__(self):
        return len(self.buffer)


def update(target, source, rate):
    """
    update function.
    when tau = 1, then it's just assignment, i.e. hard update
    Args:
        target (tf.Module): target model
        source (tf.Module): source model
    """
    target_params = target.trainable_variables
    source_params = source.trainable_variables
    for t, s in zip(target_params, source_params):
        t.assign(t * (1.0 - rate) + s * rate)


def normalize(x, stats):
    if stats is None:
        return x
    return (
        (x - tf.Variable(stats.mean, dtype=tf.float32)) /
        tf.math.sqrt(tf.Variable(stats.var, dtype=tf.float32))
    )


def denormalize(x, stats):
    if stats is None:
        return x
    return x * stats.std + stats.mean