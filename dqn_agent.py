import tensorflow as tf
from tensorboard.plugins.mesh import summary as mesh_summary
from helpers import ReplayDiscreteBuffer, update, normalize, denormalize, PrioritizedReplay, ReplayDiscreteBufferPandas, ReplayDiscreteSequenceBuffer
from helpers import ReplayDiscreteSTransformerBuffer, ReplayDiscreteNTMBuffer, ReplaySequenceDiscreteDNCBuffer
from custom_rl import QNetwork, QNetworkVision, Vae, VQVae, QAttnNetwork, VQ, QRNetwork, IQNetwork, C51Network, RecurrentQRNetwork, NALUQRNetwork, NALUQRMultiNetwork
from custom_rl import RecurrentNALUQRNetwork, TransformerNALUQRNetwork, NALUQRPNetwork, MNALUQRNetwork, SimplePPONetwork
from ann_utils import gather
import random, math
import copy
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
        
    def act(self, state, eps=0):
                
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
                 gamma=0.99, tau=1e-3):
        
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.atoms = atoms

        self.gamma = gamma
        self.tau = tau

        # Q-Network
        self.qnetwork_local = NALUQRNetwork( state_size, action_size, '{}_local'.format( name ), f1, f2, atoms, train = True, log = train )
        self.qnetwork_target = NALUQRNetwork (state_size, action_size, '{}_target'.format( name ), f1, f2, atoms )

        self.qnetwork_local( tf.zeros( [ self.batch_size, self.state_size ] ) )
        self.qnetwork_target( tf.zeros( [ self.batch_size, self.state_size ] ) )

        # self.memory = PrioritizedReplay( buffer_size, batch_size, 999, parallel_env = 1, n_step = 1 )
        self.memory = ReplayDiscreteBufferPandas( buffer_size )
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

        update( self.qnetwork_target, self.qnetwork_local, 1.0 )
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.store( state, action, reward, next_state, done )
    
    def get_optimal_action(self, state):
        z, _ = self.qnetwork_local(state)
        z = z.numpy()[0]
        q = np.mean( z, axis = 1 )
        return np.argmax( q )
    
    def act(self, state, ep, train):
        
        state = state
        eps = 1. / ((ep / 30000) + 1)
        if np.random.rand() < eps and train:
            return np.random.randint( 0, self.action_size )
        else:
            return self.get_optimal_action( state )

    def learn(self):
        
        # transitions, idx, w = self.memory.sample_batch( self.batch_size )
        transitions, idx, w = self.memory.sample_batch( self.batch_size ), None, None

        state_batch = transitions.s
        action_batch = transitions.a
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
        
        td_error, th, tloss = self.qnetwork_local.train( state_batch, theta, action_batch, w )

        # self.memory.update_priorities( idx, abs( td_error.numpy() ) )

        # ------------------- update target network ------------------- #
        if self.t_step%5 == 0:
            update( self.qnetwork_target, self.qnetwork_local, self.tau )
        

        with self.qnetwork_local.train_summary_writer.as_default():

            self.qnetwork_local.reward( tf.reduce_mean( reward_batch ) )

            terminals_count = tf.reduce_sum( terminal_mask )
            rw0 = tf.reduce_sum( tf.cast( ( ( terminal_mask * reward_batch ) > 0 ) & ( ( terminal_mask * reward_batch ) <= 5 ), tf.float32 ) ) / terminals_count
            rw5 = tf.reduce_sum( tf.cast( ( ( terminal_mask * reward_batch ) > 5 ) & ( ( terminal_mask * reward_batch ) <= 9 ), tf.float32 ) ) / terminals_count
            rw9 = tf.reduce_sum( tf.cast( ( terminal_mask * reward_batch ) > 9, tf.float32 ) ) / terminals_count
            
            tf.summary.scalar( 'loss/dqn', self.qnetwork_local.train_loss.result(), step = self.t_step )
            tf.summary.scalar( 'loss/dqn t', tloss, step = self.t_step )
            tf.summary.scalar( 'reward/reward', self.qnetwork_local.reward.result(), step = self.t_step )
            tf.summary.scalar( 'reward/t', tf.reduce_mean(reward_batch), step = self.t_step )
            
            tf.summary.scalar( 'reward/end > 0', rw0, step = self.t_step )
            tf.summary.scalar( 'reward/end > 5', rw5, step = self.t_step )
            tf.summary.scalar( 'reward/end > 9', rw9, step = self.t_step )
            
            tf.summary.histogram( 'theta out', th, step = self.t_step )
            tf.summary.histogram( 'theta target', tf.reduce_mean( theta, axis = -1 ), step = self.t_step )

            # if self.t_step % 100 == 0:

            #     for i, var in zip( idd, self.qnetwork_local.trainable_variables ):
            #         tf.summary.histogram( 'vars/{}'.format( var.name ), var, step = self.t_step )
            #         tf.summary.scalar( 'difference/{}'.format( var.name ), tf.reduce_mean( tf.abs( i - var ) ), step = self.t_step )



        self.t_step += 1

    def save_training(self, directory):
        save_checkpoint( self.qnetwork_local, directory + 'local', self.t_step )

    def restore_training(self, directory):
        self.t_step = restore_checkpoint( self.qnetwork_local, directory + 'local' )
        update( self.qnetwork_target, self.qnetwork_local, 1.0 )


'''
Distributional Quantile Regression with NALU - Multi Agent
'''
class NQRDqnMultiAgent():

    def __init__(self, envs,
                 buffer_size, batch_size, name, maxlen,
                 f1, f2, atoms, gamma=0.99, tau=1e-3):
        
        self.batch_size = batch_size
        self.atoms = atoms
        self.maxlen = maxlen

        self.gamma = gamma
        self.tau = tau
        self.action_sizes = { x['id']: x['action_dim'] for x in envs }
        self.state_sizes = { x['id']: x['state_dim'] for x in envs }

        # Q-Network
        self.qnetwork_local = NALUQRMultiNetwork( envs, '{}_local'.format( name ), f1, f2, atoms, maxlen, train = True )
        self.qnetwork_target = NALUQRMultiNetwork( envs, '{}_target'.format( name ), f1, f2, atoms, maxlen )

        inputs = [ { 'id': x['id'], 
                     'input': tf.zeros( [ self.batch_size, 1, x['state_dim'] ] )
                    } for x in envs ]

        self.memory = {}
        for i in inputs:
            self.qnetwork_local( i['input'], i['id'] )
            self.qnetwork_target( i['input'], i['id'] )
            self.memory[i['id']] = ReplayDiscreteSTransformerBuffer( self.state_sizes[i['id']], maxlen, buffer_size )
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

        update( self.qnetwork_target, self.qnetwork_local, 1.0 )
    
    def step(self, envs):
        for e in envs:
            self.memory[e['id']].store( np.array( e['buffer_s'] ), np.array( e['buffer_a'] ), 
                                        np.array( e['buffer_r'] ), np.array( e['buffer_s_'] ), np.array( e['buffer_d'] ), np.arange( 0, self.maxlen ) <= ( e['step'] - 1 ) )
    
    def get_optimal_action(self, state, env, past):
        z, p, _ = self.qnetwork_local(state, env, past)
        z = z.numpy()[0,0,...]
        q = np.mean( z, axis = 1 )
        return np.argmax( q ), p
    
    def act(self, env, ep, train):
        
        eps = 1. / ((ep / 30000) + 1)
        acs = {}
        for e in env:

            state = e['values'].state
            past = e['values'].past_state
                
            if len(past) > 0: pst = tf.convert_to_tensor( np.concatenate( past, axis = -2 ), dtype = tf.float32 )
            else: pst = None
            
            ac, present = self.get_optimal_action( tf.convert_to_tensor( [ [ state ] ], dtype = tf.float32 ), e['id'], pst )
            
            if np.random.rand() < eps and train:
                acs[e['id']] = ( np.random.randint( 0, self.action_sizes[e['id']] ), present )
            else:
                acs[e['id']] = ( ac, present )

        return acs

    def learn(self):
        
        inputs = {}
        targets = {}
        next_states = {}
        actions = {}
        t_masks = {}

        for e in self.memory:

            transitions = self.memory[e].sample_batch( self.batch_size )

            state_batch = transitions.s
            action_batch = transitions.a
            reward_batch = tf.cast( transitions.r, tf.float32 )
            next_state_batch = transitions.sp
            terminal_mask = tf.cast( transitions.it, tf.float32 )
            mask = tf.cast( transitions.m, tf.float32 )

            q, _, _ = self.qnetwork_target( next_state_batch, e )
            next_actions = np.argmax( np.mean( q, axis = -1 ), axis = -1 )
            
            one_hot_actions = tf.one_hot( next_actions, self.action_sizes[e], dtype = tf.float32 )
            q_selected = tf.reduce_sum( tf.expand_dims( one_hot_actions, axis = -1 ) * q, axis = -2 )
            theta = ( 
                        ( tf.expand_dims( terminal_mask, axis = -1 ) * ( tf.ones( self.atoms ) * tf.expand_dims( reward_batch, axis = -1 ) ) ) + 
                        ( tf.expand_dims( 1 - terminal_mask, axis = -1 ) * ( tf.expand_dims( reward_batch, axis = -1 ) + self.gamma * q_selected ) ) 
                    )

            inputs[e] = state_batch
            targets[e] = theta
            next_states[e] = next_state_batch
            actions[e] = action_batch
            t_masks[e] = mask
                        
            self.qnetwork_local.reward[e]( tf.reduce_mean( reward_batch ) )
            with self.qnetwork_local.train_summary_writer[e].as_default():

                rw0 = tf.reduce_mean( tf.cast( ( ( terminal_mask * reward_batch ) > 0 ) & ( ( terminal_mask * reward_batch ) <= 5 ), tf.float32 ) )
                rw5 = tf.reduce_mean( tf.cast( ( ( terminal_mask * reward_batch ) > 5 ) & ( ( terminal_mask * reward_batch ) <= 9 ), tf.float32 ) )
                rw9 = tf.reduce_mean( tf.cast( ( terminal_mask * reward_batch ) > 9, tf.float32 ) )

                tf.summary.scalar( 'reward/reward', self.qnetwork_local.reward[e].result(), step = self.t_step )
                tf.summary.scalar( 'reward/t', tf.reduce_mean(reward_batch), step = self.t_step )
                tf.summary.scalar( 'reward/end > 0', rw0, step = self.t_step )
                tf.summary.scalar( 'reward/end > 5', rw5, step = self.t_step )
                tf.summary.scalar( 'reward/end > 9', rw9, step = self.t_step )

                tf.summary.histogram( 'reward/total', reward_batch, step = self.t_step )
                tf.summary.histogram( 'reward/end', rw0, step = self.t_step )
                        
        tloss, tsloss, masks, th, su_vls = self.qnetwork_local.train( inputs, targets, actions, next_states, t_masks, [ e for e in self.memory ] )

        # ------------------- update target network ------------------- #
        if self.t_step % 5 == 0:
            update( self.qnetwork_target, self.qnetwork_local, self.tau )

        for e in self.memory:
            
            with self.qnetwork_local.train_summary_writer[e].as_default():

                tf.summary.scalar( 'loss/dqn', self.qnetwork_local.train_loss[e].result(), step = self.t_step )
                tf.summary.scalar( 'loss/dqn t', tloss[e], step = self.t_step )
                tf.summary.scalar( 'loss/self t', tsloss[e], step = self.t_step )

                tf.summary.histogram( 'output/th', th[e], step = self.t_step )
                tf.summary.histogram( 'output/target', targets[e], step = self.t_step )

                if self.t_step%10 == 0:
                    msks = tf.unstack( masks[e], axis = 0 )
                    for im, m in enumerate( msks ):
                        for x in range( m.shape.as_list()[ 1 ] ):
                            v = m[:,x,:,:][:,:,:,tf.newaxis]
                            tf.summary.image( 'transform_layer_{}/head_{}'.format( im, x ), v, step = self.t_step, max_outputs = 1 )

        if self.t_step%10 == 0:
            with self.qnetwork_local.train_summary_writer_total.as_default():

                for v in self.qnetwork_local.trainable_variables:
                    tf.summary.histogram( 'variables/{}'.format( v.name ), v, step = self.t_step )

                for i, v in enumerate( su_vls[1:] ):
                    tf.summary.histogram( 'state_understanding/{}'.format( i ), v, step = self.t_step )
                tf.summary.histogram( 'state_understanding/in', su_vls[0], step = self.t_step )

        self.t_step += 1

    def save_training(self, directory):
        save_checkpoint( self.qnetwork_local, directory + 'local', self.t_step )

    def restore_training(self, directory):
        self.t_step = restore_checkpoint( self.qnetwork_local, directory + 'local' )
        update( self.qnetwork_target, self.qnetwork_local, .5 )


'''
Distributional Quantile Regression with NALU and DNC Memory
'''
class M2NQRDqnAgent():

    def __init__(self, name,
        state_dim, sf1, sf2, # state understanding
        m_hsize, m, n, max_size, num_blocks, n_read, n_att_heads, lr, decay, # global memory
        fc1, fc2, # feature creator
        action_dim, df1, atoms, # actor and critic
        train, batch_size, sequence, buffer_size,
        gamma=0.99, tau=1e-3):
        
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim
        self.atoms = atoms
        self.loss_dim = 2

        # Q-Network
        self.qnetwork_local = MNALUQRNetwork( '{}_local'.format( name ), 
                                              state_dim, sf1, sf2, 
                                              m_hsize, m, n, max_size - self.loss_dim, num_blocks, n_read, n_att_heads, lr, decay,
                                              fc1, fc2,
                                              action_dim, df1, atoms, train )
        
        self.qnetwork_target = MNALUQRNetwork( '{}_target'.format( name ), 
                                               state_dim, sf1, sf2, 
                                               m_hsize, m, n, max_size - self.loss_dim, num_blocks, n_read, n_att_heads, lr, decay,
                                               fc1, fc2,
                                               action_dim, df1, atoms, False )

        gparams = self.qnetwork_local.reset( 1 )
        x_w = tf.tile( gparams[-1][:,tf.newaxis,:], [ 1, sequence - self.loss_dim, 1 ] )
        gparams = gparams[:-1]
        
        inputs = ( tf.zeros( [ 1, sequence - self.loss_dim, state_dim ] ), tf.zeros( [ 1, sequence - self.loss_dim, state_dim ] ), 
                   tf.zeros( [ 1, sequence - self.loss_dim ] ), tf.zeros( [ 1, sequence - self.loss_dim ] ), tf.zeros( [ 1, sequence - self.loss_dim ] ), x_w, gparams, tf.zeros( [ 1, 1 ] ) )
                   
        self.qnetwork_local( *inputs )
        self.qnetwork_target( *inputs )

        update( self.qnetwork_target, self.qnetwork_local, 1.0 )

        self.memory = ReplaySequenceDiscreteDNCBuffer( state_dim, sequence, buffer_size, m, n, 4 )
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done, _state, tw, M, usage, L, W_precedence, W_read, W_write, step):
        # Save experience in replay memory
        self.memory.store( state, action, reward, next_state, done, _state, tw, M, usage, L, W_precedence, W_read, W_write, step )
    
    def reset(self):
        m, u, l, wp, wr, ww, tw = self.qnetwork_local.reset(1)
        return m, u, l, wp, wr, ww, tw

    def get_optimal_action(self, state, _state, ac, rd, dn, tw, gparams, past, step):
        ac, c, gpresent, gparams, _ = self.qnetwork_local( state, _state, ac, rd, dn, tw, gparams, step, past )
        z = ac.numpy()[0,0,...]
        q = np.mean( z, axis = 1 )
        return np.argmax( q ), z, c, gpresent, gparams
    
    def act(self, state, _state, ac, rd, dn, tw, gparams, past, step, ep, train):
                
        eps = 1. / ((ep / 5000) + 1)
        ac, z, c, gpresent, gparams = self.get_optimal_action( state, _state, ac, rd, dn, tw, gparams, past, step )
        if np.random.rand() < eps and train:
            return np.random.randint( 0, self.action_dim ), z, c, gpresent, gparams
        else:
            return ac, z, c, gpresent, gparams

    def learn(self):
        
        transitions = self.memory.sample_batch( self.batch_size )

        state_batch = transitions.s
        action_batch = transitions.a
        reward_batch = tf.cast( transitions.r, tf.float32 )
        next_state_batch = transitions.sp
        terminal_mask = tf.cast( transitions.it, tf.float32 )

        _s_batch = transitions.s_

        m_batch = transitions.m
        u_batch = transitions.u
        l_batch = transitions.l
        wp_batch = transitions.wp
        wr_batch = transitions.wr
        ww_batch = transitions.ww
        step_batch = transitions.step
        tw_batch = transitions.tw

        bs, s, *_ = state_batch.shape.as_list()

        # calcular next target alinhando o estado anterior, com isso perdemos o ultimo item da sequencia
        q, _, _, _, _ = self.qnetwork_target( next_state_batch[:,:-1], _s_batch[:,1:], action_batch[:,:-1], reward_batch[:,:-1], terminal_mask[:,:-1], tw_batch[:,:-1],
                                              [ m_batch, u_batch, l_batch, wp_batch, wr_batch, ww_batch ], step_batch + 1 )

        # adicionar um valor pra mover essas variaveis 1 T a frente
        action_batch_t  = tf.concat( [ tf.zeros([bs, 1], dtype=tf.int32), action_batch ], axis = -1 ).numpy()[:,:-1]
        reward_batch_t  = tf.concat( [ tf.zeros([bs, 1], dtype=tf.float32), reward_batch ], axis = -1 ).numpy()[:,:-1]
        terminal_mask_t = tf.concat( [ tf.zeros([bs, 1], dtype=tf.float32), terminal_mask ], axis = -1 ).numpy()[:,:-1]

        # desconsiderar o primeiro item da sequencia pois ele pode atrapalhar o calculo
        state_batch = state_batch[:,1:]
        action_batch = action_batch[:,1:]
        reward_batch = reward_batch[:,1:]
        terminal_mask = terminal_mask[:,1:]
        _s_batch = _s_batch[:,1:]
        tw_batch = tw_batch[:,1:]

        q = q[:,1:]

        action_batch_t = action_batch_t[:,1:]
        reward_batch_t = reward_batch_t[:,1:]
        terminal_mask_t = terminal_mask_t[:,1:]

        # alinhar as variaveis tirando mais 1 T do final pra alinhar com o target
        state_batch = state_batch[:,:-1]
        action_batch = action_batch[:,:-1]
        reward_batch = reward_batch[:,:-1]
        terminal_mask = terminal_mask[:,:-1]
        _s_batch = _s_batch[:,:-1]
        tw_batch = tw_batch[:,:-1]
        action_batch_t = action_batch_t[:,:-1]
        reward_batch_t = reward_batch_t[:,:-1]
        terminal_mask_t = terminal_mask_t[:,:-1]
        step_batch = step_batch + 1
        
        # calcular targets
        next_actions = np.argmax( np.mean( q, axis = 3 ), axis = 2 )
        
        one_hot_actions = tf.one_hot( next_actions, self.action_dim, dtype = tf.float32 )
        q_selected = tf.reduce_sum( one_hot_actions[:,:,:,tf.newaxis] * q, axis = 2 )

        sts = tf.stack( [ tf.range( start = step_batch[b,0], limit = step_batch[b,0] + s - self.loss_dim, dtype = tf.float32 ) for b in range( bs ) ], axis = 0 )
        
        # ponderação para atrasar aprendizado ate a memoria ter dados o suficiente
        c_rate = ( 1 - self.qnetwork_local.gm.memory.memory.alpha( sts ) )
        a_rate = ( 1 - tf.clip_by_value( tf.square( reward_batch - tf.reduce_mean( q_selected, axis = -1 ) ), 0, 1 ) )
        rc = reward_batch * c_rate
        ra = reward_batch * a_rate
        
        # q_selected = one_hot_actions[:,:,:,tf.newaxis] * q
        # q_unselected = ( 1 - one_hot_actions[:,:,:,tf.newaxis] ) * q

        # rt = one_hot_actions * tf.tile( reward_batch[:,:,tf.newaxis], [1,1,4] )
        # urt = ( 1 - one_hot_actions ) * ( -1 * tf.tile( reward_batch[:,:,tf.newaxis], [1,1,4] ) )

        # a_theta1 = ( 
        #             ( ( terminal_mask[:,:,tf.newaxis,tf.newaxis] * ( tf.ones( self.atoms ) * rt[:,:,:,tf.newaxis] ) ) ) + 
        #             ( ( ( 1 - terminal_mask )[:,:,tf.newaxis,tf.newaxis] * ( rt[:,:,:,tf.newaxis] + self.gamma * q_selected ) ) )
        #            )

        # a_theta2 = ( 
        #             ( ( terminal_mask[:,:,tf.newaxis,tf.newaxis] * ( tf.ones( self.atoms ) * urt[:,:,:,tf.newaxis] ) ) ) + 
        #             ( ( ( 1 - terminal_mask )[:,:,tf.newaxis,tf.newaxis] * ( urt[:,:,:,tf.newaxis] + self.gamma * q_unselected ) ) )
        #            )

        # a_theta = a_theta1 + a_theta2
        
        a_theta = ( 
                    ( ( terminal_mask[:,:,tf.newaxis] * ( tf.ones( self.atoms ) * reward_batch[:,:,tf.newaxis] ) ) ) + 
                    ( ( ( 1 - terminal_mask )[:,:,tf.newaxis] * ( reward_batch[:,:,tf.newaxis] + self.gamma * q_selected ) ) )
                  )

        c_theta = ( tf.ones( self.atoms ) * reward_batch[:,:,tf.newaxis] )
        # c_theta = tf.random.normal( shape = [ tf.shape( q )[0], tf.shape( q )[1], self.atoms ], mean = reward_batch[:,:,tf.newaxis], stddev = 0.001 )
        

        tloss, ath, cth, msk = self.qnetwork_local.train( state_batch, _s_batch, action_batch, tw_batch, step_batch, 
                                                          [ m_batch, u_batch, l_batch, wp_batch, wr_batch, ww_batch ], 
                                                          a_theta, c_theta, action_batch_t, reward_batch_t, terminal_mask_t )
        
        self.qnetwork_local.reward(tf.reduce_mean(reward_batch))

        # ------------------- update target network ------------------- #
        update( self.qnetwork_target, self.qnetwork_local, self.tau )

        with self.qnetwork_local.train_summary_writer.as_default():

            rw = tf.reduce_mean( tf.cast( ( terminal_mask * reward_batch ) > 0, tf.float32 ) ) 
            
            tf.summary.scalar( 'loss/l2', self.qnetwork_local.train_l2_loss.result(), step = self.t_step )
            tf.summary.scalar( 'loss/dqn a', self.qnetwork_local.a_train_loss.result(), step = self.t_step )
            tf.summary.scalar( 'loss/dqn c', self.qnetwork_local.c_train_loss.result(), step = self.t_step )
            tf.summary.scalar( 'loss/dqn t', tloss, step = self.t_step )
            
            tf.summary.scalar( 'reward/reward', self.qnetwork_local.reward.result(), step = self.t_step )
            tf.summary.scalar( 'reward/t', tf.reduce_mean(reward_batch), step = self.t_step )
            
            tf.summary.scalar( 'reward/t 0.3', tf.reduce_mean( reward_batch * tf.cast( sts <= ( .3 * 600 ), tf.float32 ) ), step = self.t_step )
            tf.summary.scalar( 'reward/t 0.5', tf.reduce_mean( reward_batch * tf.cast( ( sts > ( .3 * 600 ) ) & ( sts <= ( .5 * 600 ) ), tf.float32 ) ), step = self.t_step )
            tf.summary.scalar( 'reward/t 0.7', tf.reduce_mean( reward_batch * tf.cast( ( sts > ( .5 * 600 ) ) & ( sts <= ( .7 * 600 ) ), tf.float32 ) ), step = self.t_step )
            tf.summary.scalar( 'reward/t 0.9', tf.reduce_mean( reward_batch * tf.cast( ( sts > ( .7 * 600 ) ) & ( sts <= ( .9 * 600 ) ), tf.float32 ) ), step = self.t_step )
            tf.summary.scalar( 'reward/t >0.9', tf.reduce_mean( reward_batch * tf.cast( sts > ( .9 * 600 ), tf.float32 ) ), step = self.t_step )

            tf.summary.scalar( 'reward/t 0.3 end', tf.reduce_mean( terminal_mask * reward_batch * tf.cast( sts <= ( .3 * 600 ), tf.float32 ) ), step = self.t_step )
            tf.summary.scalar( 'reward/t 0.5 end', tf.reduce_mean( terminal_mask * reward_batch * tf.cast( ( sts > ( .3 * 600 ) ) & ( sts <= ( .5 * 600 ) ), tf.float32 ) ), step = self.t_step )
            tf.summary.scalar( 'reward/t 0.7 end', tf.reduce_mean( terminal_mask * reward_batch * tf.cast( ( sts > ( .5 * 600 ) ) & ( sts <= ( .7 * 600 ) ), tf.float32 ) ), step = self.t_step )
            tf.summary.scalar( 'reward/t 0.9 end', tf.reduce_mean( terminal_mask * reward_batch * tf.cast( ( sts > ( .7 * 600 ) ) & ( sts <= ( .9 * 600 ) ), tf.float32 ) ), step = self.t_step )
            tf.summary.scalar( 'reward/t >0.9 end', tf.reduce_mean( terminal_mask * reward_batch * tf.cast( sts > ( .9 * 600 ), tf.float32 ) ), step = self.t_step )

            tf.summary.scalar( 'reward/a', tf.reduce_mean(ra), step = self.t_step )
            tf.summary.scalar( 'reward/c', tf.reduce_mean(rc), step = self.t_step )
            tf.summary.scalar( 'reward/correct end', rw, step = self.t_step )

            tf.summary.histogram( 'reward', reward_batch, step = self.t_step )
            
            tf.summary.histogram( 'theta/out_a', ath, step = self.t_step )
            tf.summary.histogram( 'theta/target_a', tf.reduce_mean( a_theta, axis = -1 ), step = self.t_step )

            tf.summary.histogram( 'theta/out_c', cth, step = self.t_step )
            tf.summary.histogram( 'theta/target_c', tf.reduce_mean( c_theta, axis = -1 ), step = self.t_step )

            msks = tf.unstack( msk, axis = 0 )
            for im, m in enumerate( msks ):
                for x in range( m.shape.as_list()[ 1 ] ):
                    v = m[:,x,:,:][:,:,:,tf.newaxis]
                    tf.summary.image( 'transform_layer_{}/head_{}'.format( im, x ), v, step = self.t_step, max_outputs = 1 )

        self.t_step += 1

    def save_training(self, directory):

        save_checkpoint( self.qnetwork_local, directory + 'local', self.t_step )
        # save_checkpoint( self.qnetwork_target, directory + 'target', self.t_step )

    def restore_training(self, directory):

        self.t_step = restore_checkpoint( self.qnetwork_local, directory + 'local' )
        # restore_checkpoint( self.qnetwork_target, directory + 'target' )


'''
Distributional Quantile Regression AC with NALU
'''
class NQRPPOAgent():

    def __init__(self, state_size, action_size,
                 buffer_size, batch_size, name,
                 f1, f2, atoms, quantile_dim, train,
                 gamma=0.99, tau=1e-5, priorized_exp=False):
        
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.atoms = atoms

        self.gamma = gamma
        self.lmbda = 0.95
        self.tau = tau
        self.priorized_exp = priorized_exp

        # Q-Network
        self.qnetwork_local = NALUQRPNetwork( state_size, action_size, '{}_local'.format( name ), f1, f2, quantile_dim, atoms, 0.01, train = True, log = train )
        self.qnetwork_target = NALUQRPNetwork( state_size, action_size, '{}_local'.format( name ), f1, f2, quantile_dim, atoms, 0.01, train = False, log = False )

        self.qnetwork_local( tf.zeros( [ self.batch_size, self.state_size ] ) )
        self.qnetwork_target( tf.zeros( [ self.batch_size, self.state_size ] ) )
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
   
    def act(self, state):
        logits, a = self.qnetwork_local.get_action( state, True )
        return a.numpy()[0], logits.numpy()

    def act_max(self, state):
        a = self.qnetwork_local.get_real_action( state, True )
        return a.numpy()[0]

    def critic(self, state, act):
        c, _ = self.qnetwork_local.get_critic( state, act, True )
        return tf.squeeze( c, axis = 1 ).numpy()

    def learn(self, states, next_state, actions, adv, old_probs, values, dones, rewards):
        
        # q_, _ = self.qnetwork_target.get_critic( next_state, actions, False )
        
        # theta = ( 
        #             ( dones[:,tf.newaxis,tf.newaxis] * ( tf.ones( self.atoms ) * values ) ) + 
        #             ( ( 1 - dones )[:,tf.newaxis,tf.newaxis] * ( values + self.gamma * q_ ) ) 
        #         )

        theta = ( tf.ones( self.atoms ) * values )

        th, aloss, closs, _ = self.qnetwork_local.train( states, theta, actions, adv, old_probs )

        dones = tf.cast( dones, tf.float32 )

        # ------------------- update target network ------------------- #
        if self.t_step%5 == 0:
            update( self.qnetwork_target, self.qnetwork_local, self.tau )
        
        self.qnetwork_local.reward( tf.reduce_mean( rewards ) )
        with self.qnetwork_local.train_summary_writer.as_default():
            
            tf.summary.scalar( 'loss/actor loss', self.qnetwork_local.train_actor_loss.result(), step = self.t_step )
            tf.summary.scalar( 'loss/critic loss', self.qnetwork_local.train_critic_loss.result(), step = self.t_step )
            tf.summary.scalar( 'reward/reward', self.qnetwork_local.reward.result(), step = self.t_step )

            tf.summary.scalar( 'loss/actor t loss', aloss, step = self.t_step )
            tf.summary.scalar( 'loss/critic t loss', closs, step = self.t_step )
            tf.summary.scalar( 'reward/t', tf.reduce_mean( rewards ), step = self.t_step )

            tf.summary.histogram( 'theta out', th, step = self.t_step )
            tf.summary.histogram( 'theta target', tf.reduce_mean( theta, axis = -1 ), step = self.t_step )

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
Distributional Transformer Quantile Regression
'''
class TransformerQRDqnAgent():

    def __init__(self, state_size, action_size,
                 buffer_size, batch_size, sequence_size, name,
                 f1, f2, atoms, max_len,
                 gamma=0.99, tau=1e-3, priorized_exp=False, reduce='sum', train=False):
        
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.atoms = atoms
        self.max_len = max_len

        self.gamma = gamma
        self.tau = tau
        self.priorized_exp = priorized_exp

        # Q-Network
        self.qnetwork_local = TransformerNALUQRNetwork( state_size, action_size, '{}_local'.format( name ), 
                                                        f1, f2, atoms, sequence_size, max_len, train = True, log = train, reduce = reduce )
        self.qnetwork_target = TransformerNALUQRNetwork( state_size, action_size, '{}_target'.format( name ), f1, f2,  atoms, sequence_size, max_len )

        self.qnetwork_local( tf.zeros( [ self.batch_size, 1, state_size ] ) )
        self.qnetwork_target( tf.zeros( [ self.batch_size, 1, state_size ] ) )

        # Replay memory
        if self.priorized_exp:
            self.memory = PrioritizedReplay( buffer_size, batch_size, 999, parallel_env = 1, n_step = 1 )
        else:
            self.memory = ReplayDiscreteSTransformerBuffer( state_size, sequence_size, buffer_size )
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.store( state, action, reward, next_state, done )
  
    def get_optimal_action(self, state, p_state):
        z, p, _, _ = self.qnetwork_local( state, p_state )
        z = z.numpy()[0,0,...]
        q = np.mean( z, axis = 1 )
        return np.argmax( q ), p
    
    def act(self, state, p_state, ep, train):
                
        eps = ( 1. / ((ep / 50000) + 1) )
        ac, past = self.get_optimal_action( state, p_state )
        if np.random.rand() < eps and train:
            return np.random.randint( 0, self.action_size ), past
        else:
            return ac, past

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

        q, _, _, _ = self.qnetwork_target( next_state_batch, None )
        next_actions = np.argmax( np.mean( q, axis = 3 ), axis = 2 )
        
        one_hot_actions = tf.one_hot( next_actions, self.action_size, dtype = tf.float32 )
        q_selected = tf.reduce_sum( one_hot_actions[:,:,:,tf.newaxis] * q, axis = 2 )
        theta = ( 
                    ( ( terminal_mask[:,:,tf.newaxis] * ( tf.ones( self.atoms ) * reward_batch[:,:,tf.newaxis] ) ) / 1.0 ) + 
                    ( ( ( 1 - terminal_mask )[:,:,tf.newaxis] * ( reward_batch[:,:,tf.newaxis] + self.gamma * q_selected ) ) / 1.0 )
                )
                
        td_error, th, tloss, msks, ps = self.qnetwork_local.train( state_batch, theta, action_batch, reward_batch, terminal_mask, self.t_step )
        
        self.qnetwork_local.treward(tf.reduce_mean(reward_batch))

        # ------------------- update target network ------------------- #
        update( self.qnetwork_target, self.qnetwork_local, self.tau )

        if self.priorized_exp:
            self.memory.update_priorities( idx, abs( td_error.numpy() ) )

        with self.qnetwork_local.train_summary_writer.as_default():
            
            tf.summary.scalar( 'loss/l2', self.qnetwork_local.train_l2_loss.result(), step = self.t_step )
            tf.summary.scalar( 'loss/dqn', self.qnetwork_local.train_loss.result(), step = self.t_step )
            tf.summary.scalar( 'loss/dqn t', tloss, step = self.t_step )
            
            tf.summary.scalar( 'reward/reward', self.qnetwork_local.treward.result(), step = self.t_step )
            tf.summary.scalar( 'reward/t', tf.reduce_mean(reward_batch), step = self.t_step )
            
            tf.summary.scalar( 'change', tf.Variable( ps, dtype = tf.int32 ), step = self.t_step )

            # for w in self.qnetwork_local.fc1s.weights: tf.summary.histogram( 'weight_{}'.format( w.name ), w, step = self.t_step )
            # for w in self.qnetwork_local.fc1h.weights: tf.summary.histogram( 'weight_{}'.format( w.name ), w, step = self.t_step )
            # for w in self.qnetwork_local.fc2.weights: tf.summary.histogram( 'weight_{}'.format( w.name ), w, step = self.t_step )
            # for w in self.qnetwork_local.rnn.weights: tf.summary.histogram( 'weight_{}'.format( w.name ), w, step = self.t_step )
            # for w in self.qnetwork_local.fc3.weights: tf.summary.histogram( 'weight_{}'.format( w.name ), w, step = self.t_step )

            # for i, w in enumerate( ps ): tf.summary.histogram( 'output/{}'.format( i ), w, step = self.t_step )
            
            tf.summary.histogram( 'theta out', th, step = self.t_step )
            tf.summary.histogram( 'theta target', tf.reduce_mean( theta, axis = -1 ), step = self.t_step )

            for im, m in enumerate( msks ):
                for x in range( m.shape.as_list()[ 1 ] ):
                    v = m[:,x,:,:][:,:,:,tf.newaxis]
                    tf.summary.image( 'transform_layer_{}/head_{}'.format( im, x ), v, step = self.t_step, max_outputs = 1 )

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
