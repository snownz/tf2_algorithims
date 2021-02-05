import tensorflow as tf
from tensorboard.plugins.mesh import summary as mesh_summary
from helpers import ReplayDiscreteBuffer, update, ActionSampler, normalize, denormalize, PrioritizedReplay, PrioritizedReplayExtra, ReplayDiscreteSequenceBuffer
from custom_rl import QNetwork, QNetworkVision, Vae, VQVae, QAttnNetwork, VQ, QRNetwork, IQNetwork, C51Network, RecurrentQRNetwork, NALUQRNetwork, AGNNALUQRNetwork, AGNNALUQRNetwork2
from custom_rl import RecurrentNALUQRNetwork
from ann_utils import gather
import random, math
import numpy as np
from utils import save_model, load_model, save_checkpoint, restore_checkpoint

'''
Simple Deep Q-Learning
'''
class DqnAgent():

    def __init__(self, state_size, action_size, 
                 buffer_size, batch_size, name,
                 f1, f2,
                 gamma=0.99, tau=1e-3, priorized_exp=False, gn=False):
        
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size

        self.gamma = gamma
        self.tau = tau
        self.priorized_exp = priorized_exp
        self.gn = gn

        # Q-Network
        self.qnetwork_local = QNetwork( state_size, action_size, '{}_local'.format( name ), f1, f2, train = True )
        self.qnetwork_target = QNetwork (state_size, action_size, '{}_target'.format( name ), f1, f2 )

        self.qnetwork_local( tf.zeros( [ self.batch_size, self.state_size ] ) )
        self.qnetwork_target( tf.zeros( [ self.batch_size, self.state_size ] ) )

        # Replay memory
        if self.priorized_exp:
            self.memory = PrioritizedReplay( buffer_size, batch_size, 999, parallel_env = 1, n_step = 1 )
        else:
            self.memory = ReplayDiscreteBuffer( state_size, buffer_size )
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.store( state, action, reward, next_state, done )
        
    def act(self, state, eps=0.):
                
        action_values = self.qnetwork_local( state )

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax( action_values.numpy() )
        else:
            return random.choice( np.arange( self.action_size ) )

    def learn(self):
        
        if self.priorized_exp:
            transitions, idx, w = self.memory.sample_batch( self.batch_size )
        else:
            transitions = self.memory.sample_batch( self.batch_size )
            w = tf.ones_like( transitions.r )
        state_batch = transitions.s
        action_batch = transitions.a
        reward_batch = transitions.r
        next_state_batch = transitions.sp
        terminal_mask = transitions.it

        td_error = self.qnetwork_local.train( state_batch, next_state_batch, reward_batch, action_batch, terminal_mask, self.gamma, self.qnetwork_target, w, self.gn )

        # ------------------- update target network ------------------- #
        update( self.qnetwork_target, self.qnetwork_local, self.tau )

        if self.priorized_exp:
            self.memory.update_priorities( idx, abs( td_error.numpy() ) )

        with self.qnetwork_local.train_summary_writer.as_default():
            
            tf.summary.scalar( 'dqn loss', self.qnetwork_local.train_loss.result(), step = self.t_step )
            tf.summary.scalar( 'dqn reward', self.qnetwork_local.reward.result(), step = self.t_step )
            tf.summary.scalar( 'learning rate', self.qnetwork_local.lr_scheduler(self.t_step), step = self.t_step )
            
            tf.summary.histogram( 'l1', self.qnetwork_local.fc1.weights[0], step = self.t_step )
            tf.summary.histogram( 'l2', self.qnetwork_local.fc2.weights[0], step = self.t_step )
            tf.summary.histogram( 'l3', self.qnetwork_local.fc3.weights[0], step = self.t_step )

            for i, a in enumerate( self.qnetwork_local.actions ):
                tf.summary.scalar( 'act_{}'.format(i), a.result(), step = self.t_step )


        self.t_step += 1

    def load(self, dir):

        self.qnetwork_local = load_model( dir + '/local/' )
        self.qnetwork_target = load_model( dir + '/target/' )

    def save(self, dir):

        save_model( self.qnetwork_local, dir + '/local/' )
        save_model( self.qnetwork_target, dir + '/target/' )

    def save_training(self, directory):

        save_checkpoint( self.qnetwork_local, directory + 'local', self.t_step )
        save_checkpoint( self.qnetwork_target, directory + 'target', self.t_step )

    def restore_training(self, directory):

        self.t_step = restore_checkpoint( self.qnetwork_local, directory + 'local' )
        restore_checkpoint( self.qnetwork_target, directory + 'target' )


'''
Distributional RL C51 - Q
'''
class C51DqnAgent():

    def __init__(self, state_size, action_size, 
                 buffer_size, batch_size, name,
                 f1, f2, atoms, v_min, v_max,
                 gamma=0.99, tau=1e-3, priorized_exp=False, gn=False):
        
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.atoms = atoms
        self.v_max = v_max
        self.v_min = v_min
        self.delta_z = float( self.v_max - self.v_min ) / ( self.atoms - 1 )
        self.z = [ self.v_min + i * self.delta_z for i in range( self.atoms ) ]

        self.gamma = gamma
        self.tau = tau
        self.priorized_exp = priorized_exp
        self.gn = gn

        # Q-Network
        self.qnetwork_local = C51Network( state_size, action_size, '{}_local'.format( name ), f1, f2, atoms, self.z, train = True )
        self.qnetwork_target = C51Network (state_size, action_size, '{}_target'.format( name ), f1, f2, atoms, self.z, )

        self.qnetwork_local( tf.zeros( [ self.batch_size, self.state_size ] ) )
        self.qnetwork_target( tf.zeros( [ self.batch_size, self.state_size ] ) )

        # Replay memory
        if self.priorized_exp:
            self.memory = PrioritizedReplay( buffer_size, batch_size, 999, parallel_env = 1, n_step = 1 )
        else:
            self.memory = ReplayDiscreteBuffer( state_size, buffer_size )
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.store( state, action, reward, next_state, done )
    
    def get_optimal_action(self, state):

        z = self.qnetwork_local(state)
        z_concat = np.vstack( z )
        q = np.sum( np.multiply( z_concat, np.array( self.z ) ), axis = 1 )
        return np.argmax( q )
    
    def act(self, state, ep, train):
                
        eps = 1. / ((ep / 10) + 1)
        if np.random.rand() < eps and train:
            return np.random.randint( 0, self.action_size )
        else:
            return self.get_optimal_action( state )

    def learn(self):
        
        if self.priorized_exp:
            transitions, idx, w = self.memory.sample_batch( self.batch_size )
        else:
            transitions = self.memory.sample_batch( self.batch_size )
            w = tf.ones_like( transitions.r )

        state_batch = transitions.s
        action_batch = transitions.a
        reward_batch = tf.cast( transitions.r, tf.float32 )
        next_state_batch = transitions.sp
        terminal_mask = tf.cast( transitions.it, tf.float32 )

        z = self.qnetwork_local( next_state_batch )
        z_ = self.qnetwork_target( next_state_batch )
        z_concat = np.vstack( z )
        q = np.sum( np.multiply( z_concat, np.array( self.z ) ), axis = 1 )
        q = q.reshape( ( self.batch_size, self.action_size ), order = 'F' )
        next_actions = np.argmax( q, axis = 1 )
        m_prob = [ np.zeros( ( self.batch_size, self.atoms ) ) for _ in range( self.action_size ) ]

        for i in range( self.batch_size ):
            if terminal_mask[i]:
                Tz = min( self.v_max, max( self.v_min, reward_batch[i] ) )
                bj = ( Tz - self.v_min ) / self.delta_z
                l, u = math.floor( bj ), math.ceil( bj )
                m_prob[ action_batch[ i ] ][ i ][ int(l) ] += ( u - bj )
                m_prob[ action_batch[ i ] ][ i ][ int(u) ] += ( bj - l )
            else:
                for j in range(self.atoms):
                    Tz = min( self.v_max, max( self.v_min, reward_batch[i] + self.gamma * self.z[j] ) )
                    bj = ( Tz - self.v_min ) / self.delta_z
                    l, u = math.floor(bj), math.ceil(bj)
                    m_prob[ action_batch[ i ] ][ i ][ int( l ) ] += z_[ next_actions[ i ] ][ i ][ j ] * ( u - bj )
                    m_prob[ action_batch[ i ] ][ i ][ int( u ) ] += z_[ next_actions[ i ] ][ i ][ j ] * ( bj - l )
                        
        td_error = self.qnetwork_local.train( state_batch, m_prob, w )
        self.qnetwork_local.reward(tf.reduce_mean(reward_batch))

        # ------------------- update target network ------------------- #
        update( self.qnetwork_target, self.qnetwork_local, self.tau )

        if self.priorized_exp:
            self.memory.update_priorities( idx, abs( td_error.numpy() ) )

        with self.qnetwork_local.train_summary_writer.as_default():
            
            tf.summary.scalar( 'dqn loss', self.qnetwork_local.train_loss.result(), step = self.t_step )
            tf.summary.scalar( 'dqn reward', self.qnetwork_local.reward.result(), step = self.t_step )
            
            tf.summary.histogram( 'l1', self.qnetwork_local.fc1.weights[0], step = self.t_step )
            tf.summary.histogram( 'l2', self.qnetwork_local.fc2.weights[0], step = self.t_step )
            tf.summary.histogram( 'l3', self.qnetwork_local.fc3.weights[0], step = self.t_step )

        self.t_step += 1

    def save_training(self, directory):

        save_checkpoint( self.qnetwork_local, directory + 'local', self.t_step )
        save_checkpoint( self.qnetwork_target, directory + 'target', self.t_step )

    def restore_training(self, directory):

        self.t_step = restore_checkpoint( self.qnetwork_local, directory + 'local' )
        restore_checkpoint( self.qnetwork_target, directory + 'target' )


'''
Distributional Quantile Regression
'''
class QRDqnAgent():

    def __init__(self, state_size, action_size, 
                 buffer_size, batch_size, name,
                 f1, f2, atoms,
                 gamma=0.99, tau=1e-3, priorized_exp=False, gn=False):
        
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.atoms = atoms

        self.gamma = gamma
        self.tau = tau
        self.priorized_exp = priorized_exp
        self.gn = gn

        # Q-Network
        self.qnetwork_local = QRNetwork( state_size, action_size, '{}_local'.format( name ), f1, f2, atoms, train = True )
        self.qnetwork_target = QRNetwork (state_size, action_size, '{}_target'.format( name ), f1, f2, atoms )

        self.qnetwork_local( tf.zeros( [ self.batch_size, self.state_size ] ) )
        self.qnetwork_target( tf.zeros( [ self.batch_size, self.state_size ] ) )

        # Replay memory
        if self.priorized_exp:
            self.memory = PrioritizedReplay( buffer_size, batch_size, 999, parallel_env = 1, n_step = 1 )
        else:
            self.memory = ReplayDiscreteBuffer( state_size, buffer_size )
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.store( state, action, reward, next_state, done )
    
    def get_optimal_action(self, state):
        z = self.qnetwork_local(state).numpy()[0]
        q = np.mean( z, axis = 1 )
        return np.argmax( q )
    
    def act(self, state, ep, train):
                
        eps = 1. / ((ep / 10000) + 1)
        if np.random.rand() < eps and train:
            return np.random.randint( 0, self.action_size )
        else:
            return self.get_optimal_action( state )

    def learn(self):
        
        if self.priorized_exp:
            transitions, idx, w = self.memory.sample_batch( self.batch_size )
        else:
            transitions = self.memory.sample_batch( self.batch_size )
            w = tf.ones_like( transitions.r )

        state_batch = transitions.s
        action_batch = tf.one_hot( transitions.a, self.action_size, dtype = tf.float32 )
        reward_batch = tf.cast( transitions.r, tf.float32 )
        next_state_batch = transitions.sp
        terminal_mask = tf.cast( transitions.it, tf.float32 )

        q = self.qnetwork_target( next_state_batch )
        next_actions = np.argmax( np.mean( q, axis = 2 ), axis = 1 )
        
        one_hot_actions = tf.one_hot( next_actions, self.action_size, dtype = tf.float32 )
        q_selected = tf.reduce_sum( one_hot_actions[:,:,tf.newaxis] * q, axis = 1 )
        theta = ( 
                    ( terminal_mask[:,tf.newaxis] * ( tf.ones( self.atoms ) * reward_batch[:,tf.newaxis] ) ) + 
                    ( ( 1 - terminal_mask )[:,tf.newaxis] * ( reward_batch[:,tf.newaxis] + self.gamma * q_selected ) ) 
                )
                
        td_error, th = self.qnetwork_local.train( state_batch, theta, action_batch, w, self.t_step, self.batch_size )
        self.qnetwork_local.reward(tf.reduce_mean(reward_batch))

        # ------------------- update target network ------------------- #
        update( self.qnetwork_target, self.qnetwork_local, self.tau )

        if self.priorized_exp:
            self.memory.update_priorities( idx, abs( td_error.numpy() ) )

        with self.qnetwork_local.train_summary_writer.as_default():
            
            tf.summary.scalar( 'l2 loss', self.qnetwork_local.train_l2_loss.result(), step = self.t_step )
            tf.summary.scalar( 'dqn loss', self.qnetwork_local.train_loss.result(), step = self.t_step )
            tf.summary.scalar( 'dqn reward', self.qnetwork_local.reward.result(), step = self.t_step )
            
            tf.summary.histogram( 'theta out', th, step = self.t_step )
            tf.summary.histogram( 'theta target', tf.reduce_mean( theta, axis = -1 ), step = self.t_step )

            tf.summary.histogram( 'l1', self.qnetwork_local.fc1.weights[0], step = self.t_step )
            tf.summary.histogram( 'l2', self.qnetwork_local.fc2.weights[0], step = self.t_step )
            tf.summary.histogram( 'l3', self.qnetwork_local.fc3.weights[0], step = self.t_step )

        self.t_step += 1

    def save_training(self, directory):

        save_checkpoint( self.qnetwork_local, directory + 'local', self.t_step )
        save_checkpoint( self.qnetwork_target, directory + 'target', self.t_step )

    def restore_training(self, directory):

        self.t_step = restore_checkpoint( self.qnetwork_local, directory + 'local' )
        restore_checkpoint( self.qnetwork_target, directory + 'target' )


'''
Distributional Quantile Regression with NALU
'''
class NQRDqnAgent():

    def __init__(self, state_size, action_size,
                 buffer_size, batch_size, name,
                 f1, f2, atoms, train,
                 gamma=0.99, tau=1e-3, priorized_exp=False, gn=False):
        
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.atoms = atoms

        self.gamma = gamma
        self.tau = tau
        self.priorized_exp = priorized_exp
        self.gn = gn

        # Q-Network
        self.qnetwork_local = NALUQRNetwork( state_size, action_size, '{}_local'.format( name ), f1, f2, atoms, train = True, log = train )
        self.qnetwork_target = NALUQRNetwork (state_size, action_size, '{}_target'.format( name ), f1, f2, atoms )

        self.qnetwork_local( tf.zeros( [ self.batch_size, self.state_size ] ) )
        self.qnetwork_target( tf.zeros( [ self.batch_size, self.state_size ] ) )

        # Replay memory
        if self.priorized_exp:
            self.memory = PrioritizedReplay( buffer_size, batch_size, 999, parallel_env = 1, n_step = 1 )
        else:
            self.memory = ReplayDiscreteBuffer( state_size, buffer_size )
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.store( state, action, reward, next_state, done )
    
    def get_optimal_action(self, state):
        z = self.qnetwork_local(state).numpy()[0]
        q = np.mean( z, axis = 1 )
        return np.argmax( q )
    
    def act(self, state, ep, train):
                
        eps = 1. / ((ep / 10000) + 1)
        if np.random.rand() < eps and train:
            return np.random.randint( 0, self.action_size )
        else:
            return self.get_optimal_action( state )

    def learn(self):
        
        if self.priorized_exp:
            transitions, idx, w = self.memory.sample_batch( self.batch_size )
        else:
            transitions = self.memory.sample_batch( self.batch_size )
            w = tf.ones_like( transitions.r )

        state_batch = transitions.s
        action_batch = tf.one_hot( transitions.a, self.action_size, dtype = tf.float32 )
        reward_batch = tf.cast( transitions.r, tf.float32 )
        next_state_batch = transitions.sp
        terminal_mask = tf.cast( transitions.it, tf.float32 )

        q = self.qnetwork_target( next_state_batch )
        next_actions = np.argmax( np.mean( q, axis = 2 ), axis = 1 )
        
        one_hot_actions = tf.one_hot( next_actions, self.action_size, dtype = tf.float32 )
        q_selected = tf.reduce_sum( one_hot_actions[:,:,tf.newaxis] * q, axis = 1 )
        theta = ( 
                    ( terminal_mask[:,tf.newaxis] * ( tf.ones( self.atoms ) * reward_batch[:,tf.newaxis] ) ) + 
                    ( ( 1 - terminal_mask )[:,tf.newaxis] * ( reward_batch[:,tf.newaxis] + self.gamma * q_selected ) ) 
                )
                
        td_error, th = self.qnetwork_local.train( state_batch, theta, action_batch, w, self.t_step, self.batch_size )
        self.qnetwork_local.reward(tf.reduce_mean(reward_batch))

        # ------------------- update target network ------------------- #
        update( self.qnetwork_target, self.qnetwork_local, self.tau )

        if self.priorized_exp:
            self.memory.update_priorities( idx, abs( td_error.numpy() ) )

        with self.qnetwork_local.train_summary_writer.as_default():
            
            tf.summary.scalar( 'l2 loss', self.qnetwork_local.train_l2_loss.result(), step = self.t_step )
            tf.summary.scalar( 'dqn loss', self.qnetwork_local.train_loss.result(), step = self.t_step )
            tf.summary.scalar( 'dqn reward', self.qnetwork_local.reward.result(), step = self.t_step )
            
            tf.summary.histogram( 'theta out', th, step = self.t_step )
            tf.summary.histogram( 'theta target', tf.reduce_mean( theta, axis = -1 ), step = self.t_step )

            tf.summary.histogram( 'l1', self.qnetwork_local.fc1.weights[0], step = self.t_step )
            tf.summary.histogram( 'l2', self.qnetwork_local.fc2.weights[0], step = self.t_step )
            tf.summary.histogram( 'l3', self.qnetwork_local.fc3.weights[0], step = self.t_step )

            tf.summary.histogram( 'non_linear_X_arithimetic_l1', tf.nn.sigmoid( self.qnetwork_local.fc1.weights[4] ), step = self.t_step )
            tf.summary.histogram( 'non_linear_X_arithimetic_l2', tf.nn.sigmoid( self.qnetwork_local.fc2.weights[4] ), step = self.t_step )
            tf.summary.histogram( 'non_linear_X_arithimetic_l3', tf.nn.sigmoid( self.qnetwork_local.fc3.weights[4] ), step = self.t_step )

            img1 = tf.tile( self.qnetwork_local.fc1.weights[4], ( self.qnetwork_local.fc1.weights[4].shape.as_list()[1] // 2,1) )
            tf.summary.image( 'non_linear_X_arithimetic_l1_i', tf.nn.sigmoid( img1[tf.newaxis,...,tf.newaxis] ), step = self.t_step, max_outputs = 1 )

            img2 = tf.tile( self.qnetwork_local.fc2.weights[4], (self.qnetwork_local.fc2.weights[4].shape.as_list()[1] // 2,1) )
            tf.summary.image( 'non_linear_X_arithimetic_l2_i', tf.nn.sigmoid( img2[tf.newaxis,...,tf.newaxis] ), step = self.t_step, max_outputs = 1 )

            img3 = tf.tile( self.qnetwork_local.fc3.weights[4], (self.qnetwork_local.fc3.weights[4].shape.as_list()[1] // 2,1) )
            tf.summary.image( 'non_linear_X_arithimetic_l3_i', tf.nn.sigmoid( img3[tf.newaxis,...,tf.newaxis] ), step = self.t_step, max_outputs = 1 )

        self.t_step += 1

    def save_training(self, directory):

        save_checkpoint( self.qnetwork_local, directory + 'local', self.t_step )
        save_checkpoint( self.qnetwork_target, directory + 'target', self.t_step )

    def restore_training(self, directory):

        self.t_step = restore_checkpoint( self.qnetwork_local, directory + 'local' )
        restore_checkpoint( self.qnetwork_target, directory + 'target' )


'''
Distributional Quantile Regression with NALU
-- Agnostic input and output
'''
class AGNNQRDqnAgent():

    def __init__(self, state_size, action_size,
                 buffer_size, batch_size, name,
                 f1, f2, atoms, train,
                 gamma=0.99, tau=1e-3, priorized_exp=False, gn=False):
        
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.atoms = atoms

        self.gamma = gamma
        self.tau = tau
        self.priorized_exp = priorized_exp
        self.gn = gn

        # Q-Network
        self.qnetwork_local = AGNNALUQRNetwork2( state_size, action_size, '{}_local'.format( name ), f1, f2, atoms, train = True, log = train )
        self.qnetwork_target = AGNNALUQRNetwork2 (state_size, action_size, '{}_target'.format( name ), f1, f2, atoms )

        self.qnetwork_local( tf.zeros( [ self.batch_size, self.state_size ] ) )
        self.qnetwork_target( tf.zeros( [ self.batch_size, self.state_size ] ) )

        # Replay memory
        if self.priorized_exp:
            self.memory = PrioritizedReplay( buffer_size, batch_size, 999, parallel_env = 1, n_step = 1 )
        else:
            self.memory = ReplayDiscreteBuffer( state_size, buffer_size )
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.store( state, action, reward, next_state, done )
    
    def get_optimal_action(self, state):
        z = self.qnetwork_local(state).numpy()[0]
        q = np.mean( z, axis = 1 )
        return np.argmax( q )
    
    def act(self, state, ep, train):
                
        eps = 1. / ((ep / 10000) + 1)
        if np.random.rand() < eps and train:
            return np.random.randint( 0, self.action_size )
        else:
            return self.get_optimal_action( state )

    def learn(self):
        
        if self.priorized_exp:
            transitions, idx, w = self.memory.sample_batch( self.batch_size )
        else:
            transitions = self.memory.sample_batch( self.batch_size )
            w = tf.ones_like( transitions.r )

        state_batch = transitions.s
        action_batch = tf.one_hot( transitions.a, self.action_size, dtype = tf.float32 )
        reward_batch = tf.cast( transitions.r, tf.float32 )
        next_state_batch = transitions.sp
        terminal_mask = tf.cast( transitions.it, tf.float32 )

        q = self.qnetwork_target( next_state_batch )
        next_actions = np.argmax( np.mean( q, axis = 2 ), axis = 1 )
        
        one_hot_actions = tf.one_hot( next_actions, self.action_size, dtype = tf.float32 )
        q_selected = tf.reduce_sum( one_hot_actions[:,:,tf.newaxis] * q, axis = 1 )
        theta = ( 
                    ( terminal_mask[:,tf.newaxis] * ( tf.ones( self.atoms ) * reward_batch[:,tf.newaxis] ) ) + 
                    ( ( 1 - terminal_mask )[:,tf.newaxis] * ( reward_batch[:,tf.newaxis] + self.gamma * q_selected ) ) 
                )
                
        td_error, th = self.qnetwork_local.train( state_batch, theta, action_batch, w, self.t_step, self.batch_size )
        self.qnetwork_local.reward(tf.reduce_mean(reward_batch))

        # ------------------- update target network ------------------- #
        update( self.qnetwork_target, self.qnetwork_local, self.tau )

        if self.priorized_exp:
            self.memory.update_priorities( idx, abs( td_error.numpy() ) )

        with self.qnetwork_local.train_summary_writer.as_default():
            
            tf.summary.scalar( 'l2 loss', self.qnetwork_local.train_l2_loss.result(), step = self.t_step )
            tf.summary.scalar( 'dqn loss', self.qnetwork_local.train_loss.result(), step = self.t_step )
            tf.summary.scalar( 'dqn reward', self.qnetwork_local.reward.result(), step = self.t_step )
            tf.summary.scalar( 'dqn t reward', tf.reduce_mean(reward_batch), step = self.t_step )
            
            tf.summary.histogram( 'theta out', th, step = self.t_step )
            tf.summary.histogram( 'theta target', tf.reduce_mean( theta, axis = -1 ), step = self.t_step )

            tf.summary.histogram( 'l1', self.qnetwork_local.fc1.weights[0], step = self.t_step )
            tf.summary.histogram( 'l2', self.qnetwork_local.fc2.weights[0], step = self.t_step )
            tf.summary.histogram( 'l3', self.qnetwork_local.fc3.weights[0], step = self.t_step )

            tf.summary.histogram( 'non_linear_X_arithimetic_l1', tf.nn.sigmoid( self.qnetwork_local.fc1.weights[4] ), step = self.t_step )
            tf.summary.histogram( 'non_linear_X_arithimetic_l2', tf.nn.sigmoid( self.qnetwork_local.fc2.weights[4] ), step = self.t_step )
            tf.summary.histogram( 'non_linear_X_arithimetic_l3', tf.nn.sigmoid( self.qnetwork_local.fc3.weights[4] ), step = self.t_step )

            img1 = tf.tile( self.qnetwork_local.fc1.weights[4], ( self.qnetwork_local.fc1.weights[4].shape.as_list()[1] // 2,1) )
            tf.summary.image( 'non_linear_X_arithimetic_l1_i', tf.nn.sigmoid( img1[tf.newaxis,...,tf.newaxis] ), step = self.t_step, max_outputs = 1 )

            img2 = tf.tile( self.qnetwork_local.fc2.weights[4], (self.qnetwork_local.fc2.weights[4].shape.as_list()[1] // 2,1) )
            tf.summary.image( 'non_linear_X_arithimetic_l2_i', tf.nn.sigmoid( img2[tf.newaxis,...,tf.newaxis] ), step = self.t_step, max_outputs = 1 )

            img3 = tf.tile( self.qnetwork_local.fc3.weights[4], (self.qnetwork_local.fc3.weights[4].shape.as_list()[1] // 2,1) )
            tf.summary.image( 'non_linear_X_arithimetic_l3_i', tf.nn.sigmoid( img3[tf.newaxis,...,tf.newaxis] ), step = self.t_step, max_outputs = 1 )

        self.t_step += 1

    def save_training(self, directory):

        save_checkpoint( self.qnetwork_local, directory + 'local', self.t_step )
        save_checkpoint( self.qnetwork_target, directory + 'target', self.t_step )

    def restore_training(self, directory):

        self.t_step = restore_checkpoint( self.qnetwork_local, directory + 'local' )
        restore_checkpoint( self.qnetwork_target, directory + 'target' )


'''
Distributional Recurrent Quantile Regression
'''
class RecurrentQRDqnAgent():

    def __init__(self, state_size, action_size,
                 buffer_size, batch_size, sequence_size, name,
                 f1, f2, r, atoms, max_len, t,
                 gamma=0.99, tau=1e-3, priorized_exp=False, reduce='sum', train=False):
        
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.atoms = atoms
        self.max_len = max_len
        self.t = t

        self.gamma = gamma
        self.tau = tau
        self.priorized_exp = priorized_exp

        # Q-Network
        self.qnetwork_local = RecurrentNALUQRNetwork( state_size, action_size, '{}_local'.format( name ), 
                                                      f1, f2, r, atoms, max_len, train = True, log = train, reduce = reduce, typer = t )
        self.qnetwork_target = RecurrentNALUQRNetwork( state_size, action_size, '{}_target'.format( name ), f1, f2, r, atoms, max_len, typer = t )

        self.qnetwork_local( tf.zeros( [ self.batch_size, max_len, self.state_size ] ), self.qnetwork_local.zero_states( self.batch_size ) )
        self.qnetwork_target( tf.zeros( [ self.batch_size, max_len, self.state_size ] ), self.qnetwork_target.zero_states( self.batch_size ) )

        # Replay memory
        if self.priorized_exp:
            self.memory = PrioritizedReplay( buffer_size, batch_size, 999, parallel_env = 1, n_step = 1 )
        else:
            self.memory = ReplayDiscreteSequenceBuffer( state_size, sequence_size, buffer_size )
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done, prev):
        # Save experience in replay memory
        self.memory.store( state, action, reward, next_state, done, prev )
    
    def reset(self, bs):
        return self.qnetwork_local.zero_states(bs)

    def get_optimal_action(self, state, p_state):
        z, p = self.qnetwork_local( state, p_state )
        z = z.numpy()[0,0,...]
        if self.t != 'l': p = p[0]
        q = np.mean( z, axis = 1 )
        return np.argmax( q ), p
    
    def act(self, state, p_state, ep, train):
                
        eps = 1. / ((ep / 5000) + 1)
        if np.random.rand() < eps and train:
            return np.random.randint( 0, self.action_size ), p_state
        else:
            state = tf.concat( ( state, tf.zeros( [ 1, self.max_len - 1, self.state_size ] ) ), axis = 1 )
            return self.get_optimal_action( state, p_state )

    def learn(self):
        
        if self.priorized_exp:
            transitions, idx, w = self.memory.sample_batch( self.batch_size )
        else:
            transitions = self.memory.sample_batch( self.batch_size )
            w = tf.ones_like( transitions.r )

        state_batch = transitions.s
        action_batch = tf.one_hot( transitions.a, self.action_size, dtype = tf.float32 )
        reward_batch = tf.cast( transitions.r, tf.float32 )
        next_state_batch = transitions.sp
        terminal_mask = tf.cast( transitions.it, tf.float32 )
        prevs = transitions.pr

        q, _ = self.qnetwork_target( next_state_batch, None )
        next_actions = np.argmax( np.mean( q, axis = 3 ), axis = 2 )
        
        one_hot_actions = tf.one_hot( next_actions, self.action_size, dtype = tf.float32 )
        q_selected = tf.reduce_sum( one_hot_actions[:,:,:,tf.newaxis] * q, axis = 2 )
        theta = ( 
                    ( terminal_mask[:,:,tf.newaxis] * ( tf.ones( self.atoms ) * reward_batch[:,:,tf.newaxis] ) ) + 
                    ( ( 1 - terminal_mask )[:,:,tf.newaxis] * ( reward_batch[:,:,tf.newaxis] + self.gamma * q_selected ) ) 
                )
                
        td_error, th = self.qnetwork_local.train( state_batch, theta, action_batch, terminal_mask, prevs, 
                                                  w, self.t_step, self.batch_size )
        
        self.qnetwork_local.treward(tf.reduce_mean(reward_batch))

        # ------------------- update target network ------------------- #
        update( self.qnetwork_target, self.qnetwork_local, self.tau )

        if self.priorized_exp:
            self.memory.update_priorities( idx, abs( td_error.numpy() ) )

        with self.qnetwork_local.train_summary_writer.as_default():
            
            tf.summary.scalar( 'l2 loss', self.qnetwork_local.train_l2_loss.result(), step = self.t_step )
            tf.summary.scalar( 'dqn loss', self.qnetwork_local.train_loss.result(), step = self.t_step )
            tf.summary.scalar( 'dqn reward', self.qnetwork_local.treward.result(), step = self.t_step )
            tf.summary.scalar( 'dqn t reward', tf.reduce_mean(reward_batch), step = self.t_step )
            
            tf.summary.histogram( 'theta out', th, step = self.t_step )
            tf.summary.histogram( 'theta target', tf.reduce_mean( theta, axis = -1 ), step = self.t_step )

            tf.summary.histogram( 'l1h', self.qnetwork_local.fc1h.weights[0], step = self.t_step )
            tf.summary.histogram( 'l1s', self.qnetwork_local.fc1s.weights[0], step = self.t_step )
            tf.summary.histogram( 'l2', self.qnetwork_local.fc2.weights[0], step = self.t_step )
            tf.summary.histogram( 'l3', self.qnetwork_local.fc3.weights[0], step = self.t_step )

        self.t_step += 1

    def save_training(self, directory):

        save_checkpoint( self.qnetwork_local, directory + 'local', self.t_step )
        save_checkpoint( self.qnetwork_target, directory + 'target', self.t_step )

    def restore_training(self, directory):

        self.t_step = restore_checkpoint( self.qnetwork_local, directory + 'local' )
        restore_checkpoint( self.qnetwork_target, directory + 'target' )


'''
Distributional Intrinsic Quantile Regression
'''
class IQDqnAgent():

    def __init__(self, state_size, action_size, 
                 buffer_size, batch_size, name,
                 f1, f2, atoms, quantile_dim,
                 gamma=0.99, tau=1e-3, priorized_exp=False, gn=False):
        
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.atoms = atoms
        self.quantile_dim = quantile_dim

        self.gamma = gamma
        self.tau = tau
        self.priorized_exp = priorized_exp
        self.gn = gn

        # Q-Network
        self.qnetwork_local = IQNetwork( state_size, action_size, '{}_local'.format( name ), f1, f2, atoms, quantile_dim, train = True )
        self.qnetwork_target = IQNetwork (state_size, action_size, '{}_target'.format( name ), f1, f2, atoms, quantile_dim )

        self.qnetwork_local( tf.zeros( [ self.batch_size, self.state_size ] ) )
        self.qnetwork_target( tf.zeros( [ self.batch_size, self.state_size ] ) )

        # Replay memory
        if self.priorized_exp:
            self.memory = PrioritizedReplay( buffer_size, batch_size, 999, parallel_env = 1, n_step = 1 )
        else:
            self.memory = ReplayDiscreteBuffer( state_size, buffer_size )
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.store( state, action, reward, next_state, done )
    
    def get_optimal_action(self, state):
        z, _ = self.qnetwork_local(state)
        q = np.mean( z.numpy(), axis = 2 )[0]
        return np.argmax( q )
    
    def act(self, state, ep, train):
                
        eps = 1. / ((ep / 10) + 1)
        if np.random.rand() < eps and train:
            return np.random.randint( 0, self.action_size )
        else:
            return self.get_optimal_action( state )

    def learn(self):
        
        if self.priorized_exp:
            transitions, idx, w = self.memory.sample_batch( self.batch_size )
        else:
            transitions = self.memory.sample_batch( self.batch_size )
            w = tf.ones_like( transitions.r )

        state_batch = transitions.s
        action_batch = tf.one_hot( transitions.a, self.action_size, dtype = tf.float32 )
        reward_batch = tf.cast( transitions.r, tf.float32 )
        next_state_batch = transitions.sp
        terminal_mask = tf.cast( transitions.it, tf.float32 )

        q, _ = self.qnetwork_target( next_state_batch )
        next_actions = np.argmax( np.mean( q, axis = 2 ), axis = 1 )
       
        one_hot_actions = tf.one_hot( next_actions, self.action_size, dtype = tf.float32 )
        q_selected = tf.reduce_sum( one_hot_actions[:,:,tf.newaxis] * q, axis = 1 )
        theta = ( 
                    ( terminal_mask[:,tf.newaxis] * ( tf.ones( self.atoms ) * reward_batch[:,tf.newaxis] ) ) + 
                    ( ( 1 - terminal_mask )[:,tf.newaxis] * ( reward_batch[:,tf.newaxis] + self.gamma * q_selected ) ) 
                )
                
        td_error = self.qnetwork_local.train( state_batch, theta, action_batch, w )
        self.qnetwork_local.reward(tf.reduce_mean(reward_batch))

        # ------------------- update target network ------------------- #
        update( self.qnetwork_target, self.qnetwork_local, self.tau )

        if self.priorized_exp:
            self.memory.update_priorities( idx, abs( td_error.numpy() ) )

        with self.qnetwork_local.train_summary_writer.as_default():
            
            tf.summary.scalar( 'dqn loss', self.qnetwork_local.train_loss.result(), step = self.t_step )
            tf.summary.scalar( 'dqn reward', self.qnetwork_local.reward.result(), step = self.t_step )
            
            tf.summary.histogram( 'l1', self.qnetwork_local.fc1.weights[0], step = self.t_step )
            tf.summary.histogram( 'l2', self.qnetwork_local.fc2.weights[0], step = self.t_step )
            tf.summary.histogram( 'fc_q', self.qnetwork_local.fc_q.weights[0], step = self.t_step )
            tf.summary.histogram( 'phi', self.qnetwork_local.phi.weights[0], step = self.t_step )
            tf.summary.histogram( 'fc', self.qnetwork_local.fc.weights[0], step = self.t_step )

        self.t_step += 1

    def save_training(self, directory):

        save_checkpoint( self.qnetwork_local, directory + 'local', self.t_step )
        save_checkpoint( self.qnetwork_target, directory + 'target', self.t_step )

    def restore_training(self, directory):

        self.t_step = restore_checkpoint( self.qnetwork_local, directory + 'local' )
        restore_checkpoint( self.qnetwork_target, directory + 'target' )


'''
Simple Deep Q-Learning with Attention
'''
class DqnAttentionAgent():

    def __init__(self, state_size, action_size, 
                 buffer_size, batch_size, name,
                 f1, f2,
                 gamma=0.99, tau=1e-3, priorized_exp=False, gn=False):
        
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size

        self.gamma = gamma
        self.tau = tau
        self.priorized_exp = priorized_exp
        self.gn = gn
        
        # Q-Network
        self.qnetwork_local = QAttnNetwork( state_size, action_size, '{}_local'.format( name ), f1, f2, 8, 8, train = True )
        self.qnetwork_target = QAttnNetwork( state_size, action_size, '{}_target'.format( name ), f1, f2, 8, 8, )

        # ac = self.qnetwork_local( tf.zeros( [ self.batch_size, state_size, state_size ] ) )
        # model_graph = ac.graph
        #writer = tf.summary.FileWriter( logdir = self.qnetwork_local.log_dir, graph = model_graph )
        #writer.flush()

        self.qnetwork_local( tf.zeros( [ self.batch_size, state_size, state_size ] ) )
        self.qnetwork_target( tf.zeros( [ self.batch_size, state_size, state_size ] ) )

        # Replay memory
        if self.priorized_exp:
            self.memory = PrioritizedReplay( buffer_size, batch_size, 999, parallel_env = 1, n_step = 1 )
        else:
            self.memory = ReplayDiscreteBuffer( state_size, buffer_size )
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.store( state, action, reward, next_state, done )
        
    def act(self, state, eps=0.):
        
        state = tf.transpose( tf.reshape( tf.tile( state, [ 1, 8 ] ), [ -1, 8, 8 ] ), perm = [ 0, 2, 1 ] )
        action_values = self.qnetwork_local( state )

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax( action_values.numpy() )
        else:
            return random.choice( np.arange( self.action_size ) )

    def learn(self):
        
        if self.priorized_exp:
            transitions, idx, w = self.memory.sample_batch( self.batch_size )
        else:
            transitions = self.memory.sample_batch( self.batch_size )
            w = tf.ones_like( transitions.r )
        
        state_batch = transitions.s
        action_batch = transitions.a
        reward_batch = transitions.r
        next_state_batch = transitions.sp
        terminal_mask = transitions.it

        state_batch = tf.transpose( tf.reshape( tf.tile( state_batch, [ 1, 8 ] ), [ -1, 8, 8 ] ), perm = [ 0, 2, 1 ] )
        next_state_batch = tf.transpose( tf.reshape( tf.tile( next_state_batch, [ 1, 8 ] ), [ -1, 8, 8 ] ), perm = [ 0, 2, 1 ] )

        # if self.t_step == 0:
        #     tf.summary.trace_on( graph = True, profiler = True )

        td_error, msk = self.qnetwork_local.train( state_batch, next_state_batch, 
                                                   reward_batch, action_batch, terminal_mask, self.gamma, self.qnetwork_target, w, self.gn )

        # ------------------- update target network ------------------- #
        update( self.qnetwork_target, self.qnetwork_local, self.tau )

        if self.priorized_exp:
            self.memory.update_priorities( idx, abs( td_error.numpy() ) )

        with self.qnetwork_local.train_summary_writer.as_default():

            # if self.t_step == 0:
            #     tf.summary.trace_export( name = "graph", step = self.t_step, profiler_outdir = self.qnetwork_local.log_dir )
            #     # tf.summary.trace_off()
            
            tf.summary.scalar( 'dqn loss', self.qnetwork_local.dqn_loss.result(), step = self.t_step )
            tf.summary.scalar( 'dqn reward', self.qnetwork_local.reward.result(), step = self.t_step )
            tf.summary.scalar( 'dqn reward done', self.qnetwork_local.reward_done.result(), step = self.t_step )
            
            tf.summary.histogram( 'l1', self.qnetwork_local.fc1.weights[0], step = self.t_step )
            tf.summary.histogram( 'l2', self.qnetwork_local.fc2.weights[0], step = self.t_step )
            tf.summary.histogram( 'l3', self.qnetwork_local.fc3.weights[0], step = self.t_step )

            for im, m in enumerate( msk ):
                for x in range( 4 ):
                    v = m[:,x,:,:][:,:,:,tf.newaxis]
                    tf.summary.image( 'layer_{}_head_{}'.format( im, x ), v, step = self.t_step, max_outputs = 1 )

        self.t_step += 1

    def save_training(self, directory):

        save_checkpoint( self.qnetwork_local, directory + 'local', self.t_step )
        save_checkpoint( self.qnetwork_target, directory + 'target', self.t_step )

    def restore_training(self, directory):

        self.t_step = restore_checkpoint( self.qnetwork_local, directory + 'local' )
        restore_checkpoint( self.qnetwork_target, directory + 'target' )


'''
Simple Deep Q-Learning with conv2d
'''
class DqnAgentVision():

    def __init__(self, state_size, action_size, 
                 buffer_size, batch_size, 
                 gamma=0.99, tau=1e-3):
        
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size

        self.gamma = gamma
        self.tau = tau

        # Q-Network
        self.qnetwork_local = QNetworkVision( state_size, action_size, 'vision_local' )
        self.qnetwork_target = QNetworkVision (state_size, action_size, 'vision_target' )

        self.qnetwork_local( tf.zeros( [ self.batch_size ] + self.state_size ) )
        self.qnetwork_target( tf.zeros( [ self.batch_size ] + self.state_size ) )

        # Replay memory
        self.memory = PrioritizedReplay( buffer_size, batch_size, 999, parallel_env = 1 )
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.store( state, action, reward, next_state, done )
        
    def act(self, state, eps=0.):
                
        action_values = self.qnetwork_local( state )

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax( action_values.numpy() )
        else:
            return random.choice( np.arange( self.action_size ) )

    def learn(self):
        
        transitions, idx, w = self.memory.sample_batch( self.batch_size )
        state_batch = transitions.s
        action_batch = transitions.a
        reward_batch = transitions.r
        next_state_batch = transitions.sp
        terminal_mask = transitions.it

        td_error = self.qnetwork_local.train( state_batch, next_state_batch, reward_batch, action_batch, terminal_mask, self.gamma, self.qnetwork_target, w )

        # ------------------- update target network ------------------- #
        update( self.qnetwork_target, self.qnetwork_local, self.tau )

        self.memory.update_priorities( idx, abs( td_error.numpy() ) )

        with self.qnetwork_local.train_summary_writer.as_default():
            
            tf.summary.scalar( 'dqn loss', self.qnetwork_local.train_loss.result(), step = self.t_step )
            tf.summary.scalar( 'dqn reward', self.qnetwork_local.reward.result(), step = self.t_step )
            
            tf.summary.histogram( 'l1', self.qnetwork_local.fc1.weights[0], step = self.t_step )
            tf.summary.histogram( 'l2', self.qnetwork_local.fc2.weights[0], step = self.t_step )
            tf.summary.histogram( 'l3', self.qnetwork_local.fc3.weights[0], step = self.t_step )

            for i, a in enumerate( self.qnetwork_local.actions ):
                tf.summary.scalar( 'act_{}'.format(i), a.result(), step = self.t_step )



        self.t_step += 1


'''
Simple Deep Q-Learning with VAE
'''
class DqnAgentVisionVae():

    def __init__(self, state_size, action_size, 
                 buffer_size, batch_size, 
                 gamma=0.99, tau=1e-3):
        
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size

        self.gamma = gamma
        self.tau = tau

        # Encoder-Network
        self.encodernet = Vae( 'vae_vision' )

        # Q-Network
        self.qnetwork_local = QNetwork( state_size, action_size, 'vae_local', f1 = 32, f2 = 16 )
        self.qnetwork_target = QNetwork (state_size, action_size, 'vae_target', f1 = 32, f2 = 16 )

        # Replay memory
        self.memory = PrioritizedReplay( buffer_size, batch_size, 999, parallel_env = 1 )
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.store( state, action, reward, next_state, done )
        
    def act(self, state, eps=0.):
        
        enc = self.encodernet( state, False )
        action_values = self.qnetwork_local( enc )

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax( action_values.numpy() )
        else:
            return random.choice( np.arange( self.action_size ) )

    def learn_policy(self):
        
        transitions, idx, w = self.memory.sample_batch( self.batch_size )
        state_batch = self.encodernet( transitions.s, True )
        action_batch = transitions.a
        reward_batch = transitions.r
        next_state_batch = self.encodernet( transitions.sp, True )
        terminal_mask = transitions.it

        td_error = self.qnetwork_local.train( state_batch, next_state_batch, reward_batch, action_batch, terminal_mask, self.gamma, self.qnetwork_target, w )

        # ------------------- update target network ------------------- #
        update( self.qnetwork_target, self.qnetwork_local, self.tau )

        self.memory.update_priorities( idx, abs( td_error.numpy() ) )

        with self.qnetwork_local.train_summary_writer.as_default():
            
            tf.summary.scalar( 'dqn loss', self.qnetwork_local.train_loss.result(), step = self.t_step )
            tf.summary.scalar( 'dqn reward', self.qnetwork_local.reward.result(), step = self.t_step )
            
            tf.summary.histogram( 'l1', self.qnetwork_local.fc1.weights[0], step = self.t_step )
            tf.summary.histogram( 'l2', self.qnetwork_local.fc2.weights[0], step = self.t_step )
            tf.summary.histogram( 'l3', self.qnetwork_local.fc3.weights[0], step = self.t_step )

            for i, a in enumerate( self.qnetwork_local.actions ):
                tf.summary.scalar( 'act_{}'.format(i), a.result(), step = self.t_step )

        self.t_step += 1

    def learn_encoder(self):

        transitions, _, _ = self.memory.sample_batch( 256 )
        state_batch = transitions.s

        decoded = self.encodernet.train( state_batch )

        with self.encodernet.train_summary_writer.as_default():

            tf.summary.scalar( 'lat loss', self.encodernet.lat_loss.result(), step = self.t_step )
            tf.summary.scalar( 'rec loss', self.encodernet.rec_loss.result(), step = self.t_step )

            for i, d in enumerate( decoded ):
                tf.summary.image( 'rec_{}'.format(i), d, step = self.t_step, max_outputs = 1 )


'''
Simple Deep Q-Learning with VQ VAE
'''
class DqnAgentVisionVQVae():

    def __init__(self, state_size, action_size,
                 buffer_size, batch_size, name,
                 f1, f2,
                 gamma=0.99, tau=1e-3, priorized_exp=False, gn=False):
        
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.priorized_exp = priorized_exp
        self.gn = gn

        self.gamma = gamma
        self.tau = tau

        # Encoder-Network
        self.encodernet = VQVae( 64, 8, 0.25, 1e-10, '{}_vae_vision'.format( name ) )

        # Q-Network
        self.qnetwork_local = QAttnNetwork( state_size, action_size, '{}_vae_local'.format( name ), f1 = f1, f2 = f2, log = True )
        self.qnetwork_target = QAttnNetwork( state_size, action_size, '{}_vae_target'.format( name ), f1 = f1, f2 = f2 )

        # Replay memory
        if self.priorized_exp:
            self.memory = PrioritizedReplay( buffer_size, batch_size, 999, parallel_env = 1, n_step = 1 )
        else:
            self.memory = ReplayDiscreteBuffer( state_size, buffer_size )
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.v_step = 0

        self.current_vision_loss = 10 
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.store( state, action, reward, next_state, done )
        
    def act(self, state, eps=0.):
        
        enc = self.encodernet( state, False )
        action_values = self.qnetwork_local( enc )

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax( action_values.numpy() )
        else:
            return random.choice( np.arange( self.action_size ) )

    def learn_policy(self):
        
        transitions, idx, w = self.memory.sample_batch( self.batch_size )
        state_batch = self.encodernet( transitions.s, False )
        action_batch = transitions.a
        reward_batch = transitions.r
        next_state_batch = self.encodernet( transitions.sp, False )
        terminal_mask = transitions.it

        td_error = self.qnetwork_local.train( state_batch, next_state_batch, reward_batch, action_batch, terminal_mask, self.gamma, self.qnetwork_target, w, self.current_vision_loss )

        # ------------------- update target network ------------------- #
        update( self.qnetwork_target, self.qnetwork_local, self.tau )

        self.memory.update_priorities( idx, abs( td_error.numpy() ) )

        with self.qnetwork_local.train_summary_writer.as_default():
            
            tf.summary.scalar( 'dqn loss', self.qnetwork_local.dqn_loss.result(), step = self.t_step )
            tf.summary.scalar( 'dqn reward', self.qnetwork_local.reward.result(), step = self.t_step )
            
            tf.summary.histogram( 'l1', self.qnetwork_local.fc1.weights[0], step = self.t_step )
            tf.summary.histogram( 'l2', self.qnetwork_local.fc2.weights[0], step = self.t_step )
            tf.summary.histogram( 'l3', self.qnetwork_local.fc3.weights[0], step = self.t_step )

            for i, a in enumerate( self.qnetwork_local.actions ):
                tf.summary.scalar( 'act_{}'.format(i), a.result(), step = self.t_step )

        self.t_step += 1

    def learn_encoder(self):

        transitions, _, _ = self.memory.sample_batch( 32 )
        state_batch = transitions.s

        decoded, f, q = self.encodernet.train( state_batch )

        fx = 0.98 
        self.current_vision_loss = ( fx * self.current_vision_loss ) + ( ( 1 - fx ) * ( self.encodernet.lat_loss.result() + self.encodernet.rec_loss.result() ) )

        with self.encodernet.train_summary_writer.as_default() as sw:
            
            tf.summary.scalar( 'lat loss', self.encodernet.lat_loss.result(), step = self.v_step )
            tf.summary.scalar( 'rec loss', self.encodernet.rec_loss.result(), step = self.v_step )
            tf.summary.scalar( 'perplexity', self.encodernet.perplexity.result(), step = self.v_step )

            tf.summary.histogram( 'pre_encoding', f, step = self.v_step )
            tf.summary.histogram( 'encoded', q, step = self.v_step )

            tf.summary.histogram( 'enc1', self.encodernet.enc1.weights[0], step = self.v_step )
            tf.summary.histogram( 'enc2', self.encodernet.enc2.weights[0], step = self.v_step )
            tf.summary.histogram( 'enc3', self.encodernet.enc3.weights[0], step = self.v_step )
            tf.summary.histogram( 'pre_enc', self.encodernet.pre_embeding.weights[0], step = self.v_step )
            
            tf.summary.histogram( 'vq', self.encodernet.vq.w, step = self.v_step )
            
            tf.summary.histogram( 'pre_rec', self.encodernet.pre_rec.weights[0], step = self.v_step )
            tf.summary.histogram( 'dec1', self.encodernet.dec1.weights[0], step = self.v_step )
            tf.summary.histogram( 'dec2', self.encodernet.dec2.weights[0], step = self.v_step )
            tf.summary.histogram( 'dec3', self.encodernet.dec3.weights[0], step = self.v_step )
            tf.summary.histogram( 'rec', self.encodernet.rec.weights[0], step = self.v_step )

            tf.summary.image( 'rec_image', decoded, step = self.v_step, max_outputs = 1 )

        self.v_step += 1

            # points = self.encodernet.enc1.weights[0]            
            # summary = mesh_summary.op( 'point_cloud', vertices = points )
            # sw.add_summary( summary )

    def save_training(self, directory):

        save_checkpoint( self.qnetwork_local, directory + 'local', self.t_step )
        save_checkpoint( self.qnetwork_target, directory + 'target', self.t_step )
        save_checkpoint( self.encodernet, directory + 'encoder', self.v_step )

    def restore_training(self, directory):

        self.v_step = restore_checkpoint( self.encodernet, directory + 'encoder' )
        self.t_step = restore_checkpoint( self.qnetwork_local, directory + 'local' )
        restore_checkpoint( self.qnetwork_target, directory + 'target' )
