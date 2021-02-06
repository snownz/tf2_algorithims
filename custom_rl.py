import tensorflow as tf 
import numpy as np
import math
import numpy as no
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, Dropout, LayerNormalization, SimpleRNNCell, LSTMCell, GRUCell, RNN, Masking
from tensorflow.keras.mixed_precision import LossScaleOptimizer
from math import ceil
from ann_utils import gather, flatten, conv1d, shape_list, Adam, norm, gelu, RMS, nalu, nalu_gru_cell, transformer_layer

from functools import partial

from random import randint, random

import os

class QNetwork(tf.Module):

    def __init__(self, state_dim, action_dim, name, f1, f2, train=False):
        
        super(QNetwork, self).__init__()
        
        self.module_type = 'QNet'

        self.fc1 = Dense( f1, kernel_initializer = tf.keras.initializers.random_normal() )
        self.fc2 = Dense( f2, kernel_initializer = tf.keras.initializers.random_normal() )
        self.fc3 = Dense( action_dim, kernel_initializer = tf.keras.initializers.random_normal() )

        self.log_dir = 'logs/qnet_{}'.format(name)

        if train:
            
            self.optimizer = Adam( tf.Variable( 2e-4 ) )
            self.generator = tf.random.Generator.from_non_deterministic_state()

            run_number = 0
            while os.path.exists(self.log_dir + str(run_number)):
                run_number += 1
            self.log_dir = self.log_dir + str(run_number)
            self.train_summary_writer = tf.summary.create_file_writer(self.log_dir)
        
        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.reward = tf.keras.metrics.Mean('reward', dtype=tf.float32)
        self.actions = [ tf.keras.metrics.Mean('act_{}'.format(x), dtype=tf.float32) for x in range(action_dim) ]

    def __call__(self, states):
        """Build a network that maps state -> action values."""
        x = tf.nn.elu( self.fc1( states ) )
        x = tf.nn.elu( self.fc2( x ) )
        return self.fc3( x )

    def train(self, states, next_states, rewards, actions, dones, gamma, target_net, w, gradient_noise):
        
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = tf.reduce_max( target_net( next_states ), axis = -1 )
        
        # Compute Q targets for current states 
        Q_targets = tf.cast( rewards, tf.float32 ) + ( gamma * Q_targets_next * ( 1 - tf.cast( dones, tf.float32 ) ) )

        with tf.GradientTape() as tape:
            # Get expected Q values from local model
            Q_expected = gather( self( states ), actions )
            mse = tf.math.square( Q_expected - Q_targets ) * w
            loss = tf.reduce_mean( mse )        
        gradients = tape.gradient( loss, self.trainable_variables )
        
        if gradient_noise:
            for g in gradients:
                noise = self.generator.normal( shape = tf.shape( g ) )
                noise = ( noise - tf.reduce_min( noise ) ) / ( tf.reduce_max( noise ) - tf.reduce_min( noise ) )
                noise = tf.where( noise >= 0.75, 0, 1 )
                # noise = tf.where( noise >= 0.4, 1, noise )
                # noise = tf.where( noise <= 0.2, -1, noise )
                # noise = tf.where( ( noise > 0.2 ) & ( noise < 0.4 ), 0, noise )
                g = g * tf.cast( noise, tf.float32 )
        
        self.optimizer.apply_gradients( zip( gradients, self.trainable_variables ) )

        self.train_loss(tf.reduce_mean(loss))
        self.reward(tf.reduce_mean(rewards))

        values = self( states )
        for i, a in enumerate( self.actions ):
            a( tf.reduce_mean( gather( values, tf.repeat( i, values.shape[0] ) ) ) )

        return mse


class QRNetwork(tf.Module):

    def __init__(self, state_dim, action_dim, name, f1, f2, atoms, train=False):
        
        super(QRNetwork, self).__init__()
        
        self.module_type = 'QRNet'

        self.to_train = train
        self.atoms = atoms
        self.action_dim = action_dim
        self.tau = [ ( 2 * ( i - 1 ) + 1 ) / ( 2 * self.atoms ) for i in range( 1, self.atoms + 1 ) ]
        
        self.fc1 = Dense( f1, kernel_initializer = tf.keras.initializers.random_normal() )
        self.fc2 = Dense( f2, kernel_initializer = tf.keras.initializers.random_normal() )
        self.fc3 = Dense( action_dim * self.atoms, kernel_initializer = tf.keras.initializers.random_normal() )

        self.dp1 = tf.keras.layers.Dropout( .25 )
        self.dp2 = tf.keras.layers.Dropout( .25 )

        self.log_dir = 'logs/qrnet_{}'.format(name)

        if train:

            self.optimizer = Adam( tf.Variable( 2e-4 ) )
            run_number = 0
            while os.path.exists(self.log_dir + str(run_number)):
                run_number += 1
            self.log_dir = self.log_dir + str(run_number)
            self.train_summary_writer = tf.summary.create_file_writer(self.log_dir)
        
        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.train_l2_loss = tf.keras.metrics.Mean('train_l2_loss', dtype=tf.float32)
        self.reward = tf.keras.metrics.Mean('reward', dtype=tf.float32)
        self.actions = [ tf.keras.metrics.Mean('act_{}'.format(x), dtype=tf.float32) for x in range(action_dim) ]
    
    def __call__(self, states):
        
        """Build a network that maps state -> action values."""
        
        x = states
        x = tf.nn.elu( self.fc1( x ) )
        x = self.dp1( x )
        x = tf.nn.elu( self.fc2( x ) )
        x = self.dp2( x )
        
        return tf.reshape( self.fc3( x ), [ -1, self.action_dim, self.atoms ] )

    def quantile_huber_loss(self, target, pred, actions):
        
        pred = tf.reduce_sum( pred * tf.expand_dims( actions, -1 ), axis = 1 )
        pred_tile = tf.tile( tf.expand_dims( pred, axis = 2 ), [ 1, 1, self.atoms ] )
        
        target_tile = tf.cast( tf.tile( tf.expand_dims( target, axis = 1 ), [ 1, self.atoms, 1 ] ), tf.float32 )
        
        # huber_loss = tf.math.square( pred_tile - target_tile )
        huber_loss = tf.compat.v1.losses.huber_loss( target_tile, pred_tile, reduction = tf.keras.losses.Reduction.NONE )
        
        tau = tf.cast( tf.reshape( np.array( self.tau ), [ 1, self.atoms ] ), tf.float32 )
        inv_tau = tf.cast( 1.0 - tau, tf.float32 )
        tau = tf.cast( tf.tile( tf.expand_dims( tau, axis = 1 ), [ 1, self.atoms, 1 ] ), tf.float32 )
        inv_tau = tf.cast( tf.tile( tf.expand_dims( inv_tau, axis = 1 ), [ 1, self.atoms, 1 ] ), tf.float32 )
        
        error_loss = tf.math.subtract( target_tile, pred_tile )
        loss = tf.where( tf.less( error_loss, 0.0 ), inv_tau * huber_loss, tau * huber_loss )
        loss = tf.reduce_sum( tf.reduce_mean( loss, axis = 2 ), axis = 1 )
        
        return loss

    def train(self, states, target, actions, we, step, bs):

        with tf.GradientTape() as tape:
            theta = self( states )
            huber_loss = self.quantile_huber_loss( target, theta, actions ) * we
            loss = tf.reduce_mean( huber_loss )            
            l2_loss = ( 
                tf.nn.l2_loss( self.fc1.weights[0] )
                + tf.nn.l2_loss( self.fc1.weights[1] )
                + tf.nn.l2_loss( self.fc2.weights[0] )
                + tf.nn.l2_loss( self.fc2.weights[1] )
                + tf.nn.l2_loss( self.fc3.weights[0] )
                + tf.nn.l2_loss( self.fc3.weights[1] ) 
            )
            t_loss = loss + ( 2e-6 * l2_loss )
            
        gradients = tape.gradient( t_loss, self.trainable_variables )
        self.optimizer.apply_gradients( zip( gradients, self.trainable_variables ) )

        self.train_loss( loss )
        self.train_l2_loss( l2_loss )

        return huber_loss, tf.reduce_mean( theta, axis = -1 )


class NALUQRNetwork(tf.Module):

    def __init__(self, state_dim, action_dim, name, f1, f2, atoms, train=False, log=False):
        
        super(NALUQRNetwork, self).__init__()
        
        self.module_type = 'NQRNet'

        self.to_train = train
        self.atoms = atoms
        self.log = log
        self.action_dim = action_dim
        self.tau = [ ( 2 * ( i - 1 ) + 1 ) / ( 2 * self.atoms ) for i in range( 1, self.atoms + 1 ) ]
        
        self.fc1 = nalu( f1, activation = tf.nn.elu, kernel_initializer = tf.keras.initializers.random_normal() )
        self.fc2 = nalu( f2, activation = tf.nn.elu, kernel_initializer = tf.keras.initializers.random_normal() )
        self.fc3 = nalu( action_dim * self.atoms, kernel_initializer = tf.keras.initializers.random_normal() )

        self.dp1 = tf.keras.layers.Dropout( .25 )
        self.dp2 = tf.keras.layers.Dropout( .25 )

        self.log_dir = 'logs/nqrnet_{}'.format(name)

        if train:

            self.optimizer = Adam( tf.Variable( 2e-4 ) )
            
            if self.log:
                run_number = 0
                while os.path.exists(self.log_dir + str(run_number)):
                    run_number += 1
                self.log_dir = self.log_dir + str(run_number)
                self.train_summary_writer = tf.summary.create_file_writer(self.log_dir)
        
        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.train_l2_loss = tf.keras.metrics.Mean('train_l2_loss', dtype=tf.float32)
        self.reward = tf.keras.metrics.Mean('reward', dtype=tf.float32)
        self.actions = [ tf.keras.metrics.Mean('act_{}'.format(x), dtype=tf.float32) for x in range(action_dim) ]
    
    def __call__(self, states):
        
        """Build a network that maps state -> action values."""
        
        x = states
        x = self.fc1( x )
        if self.log: x = self.dp1( x )
        x = self.fc2( x )
        if self.log: x = self.dp2( x )
        
        return tf.reshape( self.fc3( x ), [ -1, self.action_dim, self.atoms ] )

    def quantile_huber_loss(self, target, pred, actions):
        
        pred = tf.reduce_sum( pred * tf.expand_dims( actions, -1 ), axis = 1 )
        pred_tile = tf.tile( tf.expand_dims( pred, axis = 2 ), [ 1, 1, self.atoms ] )
        
        target_tile = tf.cast( tf.tile( tf.expand_dims( target, axis = 1 ), [ 1, self.atoms, 1 ] ), tf.float32 )
        
        # huber_loss = tf.math.square( pred_tile - target_tile )
        huber_loss = tf.compat.v1.losses.huber_loss( target_tile, pred_tile, reduction = tf.keras.losses.Reduction.NONE )
        
        tau = tf.cast( tf.reshape( np.array( self.tau ), [ 1, self.atoms ] ), tf.float32 )
        inv_tau = tf.cast( 1.0 - tau, tf.float32 )
        tau = tf.cast( tf.tile( tf.expand_dims( tau, axis = 1 ), [ 1, self.atoms, 1 ] ), tf.float32 )
        inv_tau = tf.cast( tf.tile( tf.expand_dims( inv_tau, axis = 1 ), [ 1, self.atoms, 1 ] ), tf.float32 )
        
        error_loss = tf.math.subtract( target_tile, pred_tile )
        loss = tf.where( tf.less( error_loss, 0.0 ), inv_tau * huber_loss, tau * huber_loss )
        loss = tf.reduce_sum( tf.reduce_mean( loss, axis = 2 ), axis = 1 )
        
        return loss

    def train(self, states, target, actions, we, step, bs):

        with tf.GradientTape() as tape:
            theta = self( states )
            huber_loss = self.quantile_huber_loss( target, theta, actions ) * we
            loss = tf.reduce_mean( huber_loss )            
            l2_loss = ( 
                  tf.nn.l2_loss( self.fc1.weights[3] )
                + tf.nn.l2_loss( self.fc1.weights[5] )
                + tf.nn.l2_loss( self.fc2.weights[3] )
                + tf.nn.l2_loss( self.fc2.weights[5] )
                + tf.nn.l2_loss( self.fc3.weights[3] )
                + tf.nn.l2_loss( self.fc3.weights[5] ) 
            )
            t_loss = loss # + ( 2e-6 * l2_loss )
            
        gradients = tape.gradient( t_loss, self.trainable_variables )
        self.optimizer.apply_gradients( zip( gradients, self.trainable_variables ) )

        self.train_loss( loss )
        self.train_l2_loss( l2_loss )

        return huber_loss, tf.reduce_mean( theta, axis = -1 )


class AGNNALUQRNetwork2(tf.Module):

    def __init__(self, state_dim, action_dim, name, f1, f2, atoms, train=False, log=False):
        
        super(AGNNALUQRNetwork2, self).__init__()
        
        self.module_type = 'AGNNALUQRNet'

        self.to_train = train
        self.atoms = atoms
        self.log = log
        self.f1 = f1
        self.f2 = f2
        self.ag_in_size = 16
        self.ag_out_size = action_dim * self.atoms
        self.action_dim = action_dim
        self.tau = [ ( 2 * ( i - 1 ) + 1 ) / ( 2 * self.atoms ) for i in range( 1, self.atoms + 1 ) ]
        
        self.fc1 = nalu( f1, kernel_initializer = tf.keras.initializers.random_normal() )
        self.fc2 = nalu( f2, kernel_initializer = tf.keras.initializers.random_normal() )
        self.fc3 = nalu( f2, kernel_initializer = tf.keras.initializers.random_normal() )

        self.dp1 = tf.keras.layers.Dropout( .25 )
        self.dp2 = tf.keras.layers.Dropout( .25 )

        aux1 = self.ag_out_size
        aux2 = f2
        dvs = [ 0 ]
        w = np.zeros( [ f2, self.ag_out_size ] )
        for i in range( self.ag_out_size ):
            dv = aux2 / aux1
            v = ceil( dv ) if dv - int( dv ) > .5 else int( dv )
            w[ np.sum( dvs ) : np.sum( dvs ) + v, i ] = ( 1 / v )
            dvs.append( v )
            aux1 -= 1
            aux2 -= v

        self.aw = tf.cast( tf.convert_to_tensor( w ), tf.float32 )

        self.log_dir = 'logs/agnnaluqrnet_{}'.format(name)

        if train:

            self.optimizer = Adam( tf.Variable( 2e-4 ) )
            
            if self.log:
                run_number = 0
                while os.path.exists(self.log_dir + str(run_number)):
                    run_number += 1
                self.log_dir = self.log_dir + str(run_number)
                self.train_summary_writer = tf.summary.create_file_writer(self.log_dir)
        
        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.train_l2_loss = tf.keras.metrics.Mean('train_l2_loss', dtype=tf.float32)
        self.reward = tf.keras.metrics.Mean('reward', dtype=tf.float32)
        self.actions = [ tf.keras.metrics.Mean('act_{}'.format(x), dtype=tf.float32) for x in range(action_dim) ]
      
    def __call__(self, states):
        
        """Build a network that maps state -> action values."""
                
        x = states
        x = self.fc1( x )
        if self.log: x = self.dp1( x )
        x = self.fc2( x )
        if self.log: x = self.dp2( x )
        x = self.fc3( x )
        x = x @ self.aw
        
        return tf.reshape( x, [ -1, self.action_dim, self.atoms ] )

    def quantile_huber_loss(self, target, pred, actions):
        
        pred = tf.reduce_sum( pred * tf.expand_dims( actions, -1 ), axis = 1 )
        pred_tile = tf.tile( tf.expand_dims( pred, axis = 2 ), [ 1, 1, self.atoms ] )
        
        target_tile = tf.cast( tf.tile( tf.expand_dims( target, axis = 1 ), [ 1, self.atoms, 1 ] ), tf.float32 )
        
        # huber_loss = tf.math.square( pred_tile - target_tile )
        huber_loss = tf.compat.v1.losses.huber_loss( target_tile, pred_tile, reduction = tf.keras.losses.Reduction.NONE )
        
        tau = tf.cast( tf.reshape( np.array( self.tau ), [ 1, self.atoms ] ), tf.float32 )
        inv_tau = tf.cast( 1.0 - tau, tf.float32 )
        tau = tf.cast( tf.tile( tf.expand_dims( tau, axis = 1 ), [ 1, self.atoms, 1 ] ), tf.float32 )
        inv_tau = tf.cast( tf.tile( tf.expand_dims( inv_tau, axis = 1 ), [ 1, self.atoms, 1 ] ), tf.float32 )
        
        error_loss = tf.math.subtract( target_tile, pred_tile )
        loss = tf.where( tf.less( error_loss, 0.0 ), inv_tau * huber_loss, tau * huber_loss )
        loss = tf.reduce_sum( tf.reduce_mean( loss, axis = 2 ), axis = 1 )
        
        return loss

    def train(self, states, target, actions, we, step, bs):

        with tf.GradientTape() as tape:
            theta = self( states )
            huber_loss = self.quantile_huber_loss( target, theta, actions ) * we
            loss = tf.reduce_mean( huber_loss )            
            l2_loss = ( 
                  tf.nn.l2_loss( self.fc1.weights[3] )
                + tf.nn.l2_loss( self.fc1.weights[5] )
                + tf.nn.l2_loss( self.fc2.weights[3] )
                + tf.nn.l2_loss( self.fc2.weights[5] )
                + tf.nn.l2_loss( self.fc3.weights[3] )
                + tf.nn.l2_loss( self.fc3.weights[5] ) 
            )
            t_loss = loss + ( 2e-6 * l2_loss )
            
        gradients = tape.gradient( t_loss, self.trainable_variables )
        self.optimizer.apply_gradients( zip( gradients, self.trainable_variables ) )

        self.train_loss( loss )
        self.train_l2_loss( l2_loss )

        return huber_loss, tf.reduce_mean( theta, axis = -1 )


class AGNNALUQRNetwork(tf.Module):

    def __init__(self, state_dim, action_dim, name, f1, f2, atoms, train=False, log=False):
        
        super(AGNNALUQRNetwork, self).__init__()
        
        self.module_type = 'AGNNALUQRNet'

        self.to_train = train
        self.atoms = atoms
        self.log = log
        self.f1 = f1
        self.f2 = f2
        self.ag_in_size = 16
        self.action_dim = action_dim
        self.tau = [ ( 2 * ( i - 1 ) + 1 ) / ( 2 * self.atoms ) for i in range( 1, self.atoms + 1 ) ]
        
        self.fc1 = nalu( f1, kernel_initializer = tf.keras.initializers.random_normal() )
        self.fc2 = nalu( f2, kernel_initializer = tf.keras.initializers.random_normal() )
        self.fc3 = nalu( action_dim * self.atoms, kernel_initializer = tf.keras.initializers.random_normal() )

        self.dp1 = tf.keras.layers.Dropout( .25 )
        self.dp2 = tf.keras.layers.Dropout( .25 )

        self.gen = tf.random.get_global_generator()

        aux1 = state_dim
        aux2 = self.ag_in_size
        dvs = [ 0 ]
        w = np.zeros( [ state_dim, self.ag_in_size ] )
        for i in range( state_dim ):
            dv = aux2 / aux1
            v = ceil( dv ) if dv - int( dv ) > .5 else int( dv )
            w[ i, np.sum( dvs ) : np.sum( dvs ) + v ] = ( 1 / v )
            dvs.append( v )
            aux1 -= 1
            aux2 -= v

        self.w = tf.cast( tf.convert_to_tensor( w ), tf.float32 )

        self.log_dir = 'logs/agnnaluqrnet_{}'.format(name)

        if train:

            self.optimizer = Adam( tf.Variable( 2e-4 ) )
            
            if self.log:
                run_number = 0
                while os.path.exists(self.log_dir + str(run_number)):
                    run_number += 1
                self.log_dir = self.log_dir + str(run_number)
                self.train_summary_writer = tf.summary.create_file_writer(self.log_dir)
        
        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.train_l2_loss = tf.keras.metrics.Mean('train_l2_loss', dtype=tf.float32)
        self.reward = tf.keras.metrics.Mean('reward', dtype=tf.float32)
        self.actions = [ tf.keras.metrics.Mean('act_{}'.format(x), dtype=tf.float32) for x in range(action_dim) ]
      
    def __call__(self, states):
        
        """Build a network that maps state -> action values."""
                
        w_specs = tf.tile( tf.convert_to_tensor( [ states.shape[-1] / self.ag_in_size ] )[tf.newaxis, :], [ states.shape[0], 1 ] )
        x = states @ self.w
        x = tf.stop_gradient( tf.concat( [ x, w_specs ], axis = -1 ) )
        
        x = self.fc1( x )
        if self.log: x = self.dp1( x )
        x = self.fc2( x )
        if self.log: x = self.dp2( x )
        
        return tf.reshape( self.fc3( x ), [ -1, self.action_dim, self.atoms ] )

    def quantile_huber_loss(self, target, pred, actions):
        
        pred = tf.reduce_sum( pred * tf.expand_dims( actions, -1 ), axis = 1 )
        pred_tile = tf.tile( tf.expand_dims( pred, axis = 2 ), [ 1, 1, self.atoms ] )
        
        target_tile = tf.cast( tf.tile( tf.expand_dims( target, axis = 1 ), [ 1, self.atoms, 1 ] ), tf.float32 )
        
        # huber_loss = tf.math.square( pred_tile - target_tile )
        huber_loss = tf.compat.v1.losses.huber_loss( target_tile, pred_tile, reduction = tf.keras.losses.Reduction.NONE )
        
        tau = tf.cast( tf.reshape( np.array( self.tau ), [ 1, self.atoms ] ), tf.float32 )
        inv_tau = tf.cast( 1.0 - tau, tf.float32 )
        tau = tf.cast( tf.tile( tf.expand_dims( tau, axis = 1 ), [ 1, self.atoms, 1 ] ), tf.float32 )
        inv_tau = tf.cast( tf.tile( tf.expand_dims( inv_tau, axis = 1 ), [ 1, self.atoms, 1 ] ), tf.float32 )
        
        error_loss = tf.math.subtract( target_tile, pred_tile )
        loss = tf.where( tf.less( error_loss, 0.0 ), inv_tau * huber_loss, tau * huber_loss )
        loss = tf.reduce_sum( tf.reduce_mean( loss, axis = 2 ), axis = 1 )
        
        return loss

    def train(self, states, target, actions, we, step, bs):

        with tf.GradientTape() as tape:
            theta = self( states )
            huber_loss = self.quantile_huber_loss( target, theta, actions ) * we
            loss = tf.reduce_mean( huber_loss )            
            l2_loss = ( 
                  tf.nn.l2_loss( self.fc1.weights[3] )
                + tf.nn.l2_loss( self.fc1.weights[5] )
                + tf.nn.l2_loss( self.fc2.weights[3] )
                + tf.nn.l2_loss( self.fc2.weights[5] )
                + tf.nn.l2_loss( self.fc3.weights[3] )
                + tf.nn.l2_loss( self.fc3.weights[5] ) 
            )
            t_loss = loss + ( 2e-6 * l2_loss )
            
        gradients = tape.gradient( t_loss, self.trainable_variables )
        self.optimizer.apply_gradients( zip( gradients, self.trainable_variables ) )

        self.train_loss( loss )
        self.train_l2_loss( l2_loss )

        return huber_loss, tf.reduce_mean( theta, axis = -1 )


class RecurrentQRNetwork(tf.Module):

    def __init__(self, state_dim, action_dim, name, f1, f2, r, atoms, max_len, 
                typer='s', train=False, log=False, reduce='sum'):
        
        super(RecurrentQRNetwork, self).__init__()
        
        self.module_type = 'RQRNet'

        self.to_train = train
        self.log = log
        self.atoms = atoms
        self.r = r
        self.f2 = f2
        self.typer = typer
        self.action_dim = action_dim
        self.tau = [ ( 2 * ( i - 1 ) + 1 ) / ( 2 * self.atoms ) for i in range( 1, self.atoms + 1 ) ]
                                
        self.fc1 = Dense( f1, kernel_initializer = tf.keras.initializers.orthogonal() )
        self.fc2 = Dense( f2, kernel_initializer = tf.keras.initializers.orthogonal() )
        self.fc3 = Dense( action_dim * self.atoms, kernel_initializer = tf.keras.initializers.orthogonal() )

        if typer == 's':
            self.fcr = SimpleRNNCell( r, activation = tf.nn.tanh, dropout = 0 if log else .25 )
        elif typer == 'g':
            self.fcr = GRUCell( r, activation = tf.nn.tanh, dropout = 0 if log else .25 )
        elif typer == 'l':
            self.fcr = LSTMCell( r, activation = tf.nn.tanh, dropout = 0 if log else .25 )
           
        self.rnn = RNN( self.fcr, return_sequences = True, return_state = True, unroll = True )

        self.dp1 = tf.keras.layers.Dropout( .25 )
        self.dp2 = tf.keras.layers.Dropout( .25 )

        self.log_dir = 'logs/rqrnet_{}'.format(name)

        if train:
            
            self.optimizer = Adam( tf.Variable( 2e-4 ) )
            
            if reduce == 'sum':
                self.train = self.train_all_states
                self.reduce = lambda huber_loss : tf.reduce_mean( tf.reduce_sum( huber_loss, axis = 1 ) )                
            
            elif reduce == 'mean':
                self.train = self.train_all_states
                self.reduce = lambda huber_loss : tf.reduce_mean( tf.reduce_mean( huber_loss, axis = 1 ) )
            
            elif reduce == 'sum_p':
                self.train = self.train_all_states
                p = [ ( 2 * ( i - 1 ) + 1 ) / ( 2 * max_len ) for i in range( 1, max_len + 1 ) ]
                self.reduce = lambda huber_loss : tf.reduce_mean( tf.reduce_sum( huber_loss * p, axis = 1 ) )

            elif reduce == 'sum_half':
                self.train = self.train_all_states
                p = [ int( i >= max_len // 2 ) for i in range( 0, max_len ) ]
                self.reduce = lambda huber_loss : tf.reduce_mean( tf.reduce_sum( huber_loss * p, axis = 1 ) )

            elif reduce == 'mean_half':
                self.train = self.train_all_states
                p = [ int( i >= max_len // 2 ) for i in range( 0, max_len ) ]
                self.reduce = lambda huber_loss : tf.reduce_mean( tf.reduce_mean( huber_loss * p, axis = 1 ) )
            
            elif reduce == 'sum_end':
                self.train = self.train_terminal
                self.reduce_done = lambda huber_loss, terminal : tf.reduce_mean( tf.reduce_sum( huber_loss * terminal, axis = 1 ) )                        
            
            else:
                self.train = self.train_end_state
            
            if log:
                run_number = 0
                while os.path.exists(self.log_dir + str(run_number)):
                    run_number += 1
                self.log_dir = self.log_dir + str(run_number)
                self.train_summary_writer = tf.summary.create_file_writer(self.log_dir)
        
        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.train_l2_loss = tf.keras.metrics.Mean('train_l2_loss', dtype=tf.float32)
        self.ereward = tf.keras.metrics.Mean('end reward', dtype=tf.float32)
        self.treward = tf.keras.metrics.Mean('total reward', dtype=tf.float32)
    
    def random_states(self, bs):

        if self.typer == 's': return tf.random.normal( [ bs, self.r ] )
        if self.typer == 'g': return tf.random.normal( [ bs, self.r ] )
        if self.typer == 'l': return ( tf.random.normal( [ bs, self.r ] ), tf.random.normal( [ bs, self.r ] ) )

    def zero_states(self, bs):

        if self.typer == 's': return tf.zeros( [ bs, self.r ] )
        if self.typer == 'g': return tf.zeros( [ bs, self.r ] )
        if self.typer == 'l': return ( tf.zeros( [ bs, self.r ] ), tf.zeros( [ bs, self.r ] ) )

    def encode(self, state):
        x = state
        x = tf.nn.elu( self.fc1( x ) )
        if self.log: x = self.dp1( x )
        x = tf.nn.elu( self.fc2( x ) )
        if self.log: x = self.dp2( x )
        return x

    def __call__(self, states, p_state):
        """Build a network that maps state -> action values."""        
        
        bs, ss, _ = shape_list( states )
        x = self.encode( states )
        xt, *h = self.rnn( x, initial_state = p_state, training = self.log )
        pred = self.fc3( x + xt )

        return tf.reshape( pred, [ bs, ss, self.action_dim, self.atoms ] ), h
    
    def quantile_huber_loss(self, target, pred, actions):
        
        pred = tf.reduce_sum( pred * tf.expand_dims( actions, -1 ), axis = 2 )
        pred_tile = tf.tile( tf.expand_dims( pred, axis = 3 ), [ 1, 1, 1, self.atoms ] )
        
        target_tile = tf.cast( tf.tile( tf.expand_dims( target, axis = 2 ), [ 1, 1, self.atoms, 1 ] ), tf.float32 )
        
        huber_loss = tf.math.square( pred_tile - target_tile )
        # huber_loss = tf.compat.v1.losses.huber_loss( target_tile, pred_tile, reduction = tf.keras.losses.Reduction.NONE )
        
        tau = tf.cast( tf.reshape( np.array( self.tau ), [ 1, self.atoms ] ), tf.float32 )
        inv_tau = tf.cast( 1.0 - tau, tf.float32 )
        tau = tf.cast( tf.tile( tf.expand_dims( tau, axis = 1 ), [ 1, self.atoms, 1 ] ), tf.float32 )
        inv_tau = tf.cast( tf.tile( tf.expand_dims( inv_tau, axis = 1 ), [ 1, self.atoms, 1 ] ), tf.float32 )
        
        error_loss = tf.math.subtract( target_tile, pred_tile )
        loss = tf.where( tf.less( error_loss, 0.0 ), inv_tau[:,tf.newaxis,:,:] * huber_loss, tau[:,tf.newaxis,:,:] * huber_loss )
        loss = tf.reduce_sum( tf.reduce_mean( loss, axis = 3 ), axis = 2 )
        
        return loss

    def train_end_state(self, states, target, actions, dones, we, step, bs):

        bs, _, _ = shape_list( states )

        with tf.GradientTape() as tape:

            theta, _ = self( states, self.zero_states( bs ) )

            huber_loss = self.quantile_huber_loss( target[:,-1,...], theta[:,-1,...], actions[:,-1,...] )            
            loss = tf.reduce_mean( huber_loss )

            l2_loss = ( 
                tf.nn.l2_loss( self.fc1.weights[0] )
                + tf.nn.l2_loss( self.fc1.weights[1] )
                + tf.nn.l2_loss( self.fc2.weights[0] )
                + tf.nn.l2_loss( self.fc2.weights[1] )
                + tf.nn.l2_loss( self.fc3.weights[0] )
                + tf.nn.l2_loss( self.fc3.weights[1] )
            )
            t_loss = loss + ( 2e-4 * l2_loss )
            
        gradients = tape.gradient( t_loss, self.trainable_variables )
        self.optimizer.apply_gradients( zip( gradients, self.trainable_variables ) )

        self.train_loss( loss )
        self.train_l2_loss( 2e-4 * l2_loss )

        return huber_loss, tf.reduce_mean( theta, axis = -1 )

    def train_terminal(self, states, target, actions, dones, we, step, bs):

        bs, ss, _ = shape_list( states )

        with tf.GradientTape() as tape:

            theta, _ = self( states, self.zero_states( bs ) )

            theta = tf.reshape( theta, [ bs * ss, self.action_dim, self.atoms ] )
            target = tf.reshape( target, [ bs * ss, self.atoms ] )
            actions = tf.reshape( actions, [ bs * ss, self.action_dim ] )
            we = tf.reshape( we, [ bs * ss ] )

            huber_loss = tf.reshape( self.quantile_huber_loss( target, theta, actions ) * we, [ bs, ss ] )            
            loss = self.reduce_done( huber_loss, dones )

            l2_loss = ( 
                tf.nn.l2_loss( self.fc1.weights[0] )
                + tf.nn.l2_loss( self.fc1.weights[1] )
                + tf.nn.l2_loss( self.fc2.weights[0] )
                + tf.nn.l2_loss( self.fc2.weights[1] )
                + tf.nn.l2_loss( self.fc3.weights[0] )
                + tf.nn.l2_loss( self.fc3.weights[1] )
            )
            t_loss = loss + ( 2e-4 * l2_loss )
            
        gradients = tape.gradient( t_loss, self.trainable_variables )
        self.optimizer.apply_gradients( zip( gradients, self.trainable_variables ) )

        self.train_loss( loss )
        self.train_l2_loss( 2e-4 * l2_loss )

        return huber_loss, tf.reduce_mean( theta, axis = -1 )

    def train_all_states(self, states, target, actions, dones, we, step, bs):

        bs, *_ = shape_list( states )

        with tf.GradientTape() as tape:

            theta, _ = self( states, None )
            
            huber_loss = self.quantile_huber_loss( target, theta, actions ) * we         
            loss = self.reduce( huber_loss )

            l2_loss = ( 
                tf.nn.l2_loss( self.fc1.weights[0] )
                + tf.nn.l2_loss( self.fc1.weights[1] )
                + tf.nn.l2_loss( self.fc2.weights[0] )
                + tf.nn.l2_loss( self.fc2.weights[1] )
                + tf.nn.l2_loss( self.fc3.weights[0] )
                + tf.nn.l2_loss( self.fc3.weights[1] )
            )
            t_loss = loss + ( 2e-4 * l2_loss )
            
        gradients = tape.gradient( t_loss, self.trainable_variables )
        self.optimizer.apply_gradients( zip( gradients, self.trainable_variables ) )

        self.train_loss( loss )
        self.train_l2_loss( 2e-4 * l2_loss )

        return huber_loss, tf.reduce_mean( theta, axis = -1 )


class RecurrentNALUQRNetwork(tf.Module):

    def __init__(self, state_dim, action_dim, name, f1, f2, r, atoms, max_len, 
                typer='s', train=False, log=False, reduce='sum'):
        
        super(RecurrentNALUQRNetwork, self).__init__()
        
        self.module_type = 'RNALUQRNet'

        self.to_train = train
        self.log = log
        self.atoms = atoms
        self.r = r
        self.f2 = f2
        self.typer = typer
        self.action_dim = action_dim
        self.tau = [ ( 2 * ( i - 1 ) + 1 ) / ( 2 * self.atoms ) for i in range( 1, self.atoms + 1 ) ]
                                
        self.fc1s = nalu( f1, activation = tf.nn.elu, kernel_initializer = tf.keras.initializers.random_normal() )
        self.fc1h = nalu( f1, activation = tf.nn.elu, kernel_initializer = tf.keras.initializers.random_normal() )
        self.fc2 = nalu( f2, activation = tf.nn.elu, kernel_initializer = tf.keras.initializers.random_normal() )
        self.fc3 = nalu( action_dim * self.atoms, kernel_initializer = tf.keras.initializers.random_normal() )

        if typer == 's':
            self.fcr = SimpleRNNCell( r, activation = tf.nn.tanh, dropout = 0 if log else .25 )
        elif typer == 'g':
            self.fcr = nalu_gru_cell( r, tf.keras.initializers.orthogonal() )
        elif typer == 'l':
            self.fcr = LSTMCell( r, activation = tf.nn.tanh, dropout = 0 if log else .25 )
           
        self.rnn = RNN( self.fcr, return_sequences = True, return_state = True, unroll = True )

        self.dp1 = tf.keras.layers.Dropout( .25 )
        self.dp2 = tf.keras.layers.Dropout( .25 )

        self.log_dir = 'logs/rnaluqrnet_{}'.format(name)

        if train:
            
            self.optimizer = Adam( tf.Variable( 2e-4 ) )
            
            if reduce == 'sum':
                self.reduce = lambda loss : tf.reduce_mean( tf.reduce_sum( loss, axis = 1 ) )                
            
            elif reduce == 'mean':
                self.reduce = lambda loss : tf.reduce_mean( tf.reduce_mean( loss, axis = 1 ) )
            
            elif reduce == 'sum_p':
                p = [ ( 2 * ( i - 1 ) + 1 ) / ( 2 * max_len ) for i in range( 1, max_len + 1 ) ]
                self.reduce = lambda loss : tf.reduce_mean( tf.reduce_sum( loss * p, axis = 1 ) )

            elif reduce == 'sum_half':
                p = [ int( i >= max_len // 2 ) for i in range( 0, max_len ) ]
                self.reduce = lambda loss : tf.reduce_mean( tf.reduce_sum( loss * p, axis = 1 ) )

            elif reduce == 'mean_half':
                p = [ int( i >= max_len // 2 ) for i in range( 0, max_len ) ]
                self.reduce = lambda loss : tf.reduce_mean( tf.reduce_mean( loss * p, axis = 1 ) )
                        
            if log:
                run_number = 0
                while os.path.exists(self.log_dir + str(run_number)):
                    run_number += 1
                self.log_dir = self.log_dir + str(run_number)
                self.train_summary_writer = tf.summary.create_file_writer(self.log_dir)
        
        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.train_l2_loss = tf.keras.metrics.Mean('train_l2_loss', dtype=tf.float32)
        self.treward = tf.keras.metrics.Mean('total reward', dtype=tf.float32)
    
    def random_states(self, bs):

        if self.typer == 's': return tf.random.normal( [ bs, self.r ] )
        if self.typer == 'g': return tf.random.normal( [ bs, self.r ] )
        if self.typer == 'l': return ( tf.random.normal( [ bs, self.r ] ), tf.random.normal( [ bs, self.r ] ) )

    def zero_states(self, bs):

        if self.typer == 's': return tf.zeros( [ bs, self.r ] )
        if self.typer == 'g': return tf.zeros( [ bs, self.r ] )
        if self.typer == 'l': return ( tf.zeros( [ bs, self.r ] ), tf.zeros( [ bs, self.r ] ) )

    def __call__(self, states, p_state):
        
        """Build a network that maps state -> action values."""        
        
        bs, ss, _ = shape_list( states )

        xt, *h = self.rnn( states, initial_state = p_state, training = self.log )
        
        xh = self.fc1h( xt )
        xs = self.fc1s( states )
        
        x = xh + xs
        if self.log: x = self.dp1( x )
        
        x = self.fc2( x )
        if self.log: x = self.dp2( x )        
        
        pred = self.fc3( x )

        return tf.reshape( pred, [ bs, ss, self.action_dim, self.atoms ] ), h
    
    def quantile_huber_loss(self, target, pred, actions):
        
        pred = tf.reduce_sum( pred * tf.expand_dims( actions, -1 ), axis = 2 )
        pred_tile = tf.tile( tf.expand_dims( pred, axis = 3 ), [ 1, 1, 1, self.atoms ] )
        
        target_tile = tf.cast( tf.tile( tf.expand_dims( target, axis = 2 ), [ 1, 1, self.atoms, 1 ] ), tf.float32 )
        
        # huber_loss = tf.math.square( pred_tile - target_tile )
        huber_loss = tf.compat.v1.losses.huber_loss( target_tile, pred_tile, reduction = tf.keras.losses.Reduction.NONE )
        
        tau = tf.cast( tf.reshape( np.array( self.tau ), [ 1, self.atoms ] ), tf.float32 )
        inv_tau = tf.cast( 1.0 - tau, tf.float32 )
        tau = tf.cast( tf.tile( tf.expand_dims( tau, axis = 1 ), [ 1, self.atoms, 1 ] ), tf.float32 )
        inv_tau = tf.cast( tf.tile( tf.expand_dims( inv_tau, axis = 1 ), [ 1, self.atoms, 1 ] ), tf.float32 )
        
        error_loss = tf.math.subtract( target_tile, pred_tile )
        loss = tf.where( tf.less( error_loss, 0.0 ), inv_tau[:,tf.newaxis,:,:] * huber_loss, tau[:,tf.newaxis,:,:] * huber_loss )
        loss = tf.reduce_sum( tf.reduce_mean( loss, axis = 3 ), axis = 2 )
        
        return loss

    def train(self, states, target, actions, dones, prev, we, step, bs):

        bs, *_ = shape_list( states )

        with tf.GradientTape() as tape:

            theta, _ = self( states, prev )
            
            huber_loss = self.quantile_huber_loss( target, theta, actions )
            loss = self.reduce( huber_loss )

            t_loss = loss
            
        gradients = tape.gradient( t_loss, self.trainable_variables )
        self.optimizer.apply_gradients( zip( gradients, self.trainable_variables ) )

        self.train_loss( loss )
        self.train_l2_loss( 0 )

        return huber_loss, tf.reduce_mean( theta, axis = -1 )


class TransformerNALUQRNetwork(tf.Module):

    def __init__(self, state_dim, action_dim, name, f1, f2, atoms, min_len, max_len, train=False, log=False, reduce='sum'):
        
        super(TransformerNALUQRNetwork, self).__init__()
        
        self.module_type = 'TNALUQRNet'

        self.to_train = train
        self.log = log
        self.atoms = atoms
        self.action_dim = action_dim
        self.tau = [ ( 2 * ( i - 1 ) + 1 ) / ( 2 * self.atoms ) for i in range( 1, self.atoms + 1 ) ]
                                
        self.fc1s = nalu( f1, activation = tf.nn.elu, kernel_initializer = tf.keras.initializers.random_normal() )
        self.fc1h = nalu( f1, activation = tf.nn.elu, kernel_initializer = tf.keras.initializers.random_normal() )
        self.fc2 = nalu( f2, activation = tf.nn.elu, kernel_initializer = tf.keras.initializers.random_normal() )
        self.fc3 = nalu( action_dim * self.atoms, kernel_initializer = tf.keras.initializers.random_normal() )
           
        self.rnn = transformer_layer( f1, 4, 1, max_len )

        self.dp1 = tf.keras.layers.Dropout( .25 )
        self.dp2 = tf.keras.layers.Dropout( .25 )

        self.log_dir = 'logs/tnaluqrnet_{}'.format(name)

        if train:
            
            self.optimizer = Adam( tf.Variable( 2e-4 ) )
            
            if reduce == 'sum':
                self.reduce = lambda loss : tf.reduce_mean( tf.reduce_sum( loss, axis = 1 ) )
            
            elif reduce == 'mean':
                self.reduce = lambda loss : tf.reduce_mean( tf.reduce_mean( loss, axis = 1 ) )
            
            elif reduce == 'sum_p':
                p = [ ( 2 * ( i - 1 ) + 1 ) / ( 2 * min_len ) for i in range( 1, min_len + 1 ) ]
                self.reduce = lambda loss : tf.reduce_mean( tf.reduce_sum( loss * p, axis = 1 ) )

            elif reduce == 'sum_half':
                p = [ int( i >= min_len // 2 ) for i in range( 0, min_len ) ]
                self.reduce = lambda loss : tf.reduce_mean( tf.reduce_sum( loss * p, axis = 1 ) )

            elif reduce == 'mean_half':
                p = [ int( i >= min_len // 2 ) for i in range( 0, min_len ) ]
                self.reduce = lambda loss : tf.reduce_mean( tf.reduce_mean( loss * p, axis = 1 ) )
                        
            if log:
                run_number = 0
                while os.path.exists(self.log_dir + str(run_number)):
                    run_number += 1
                self.log_dir = self.log_dir + str(run_number)
                self.train_summary_writer = tf.summary.create_file_writer(self.log_dir)
        
        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.train_l2_loss = tf.keras.metrics.Mean('train_l2_loss', dtype=tf.float32)
        self.treward = tf.keras.metrics.Mean('total reward', dtype=tf.float32)

    def __call__(self, states, past=None):
        
        """Build a network that maps state -> action values."""        
        
        bs, ss, _ = shape_list( states )

        # compute input embeding
        xs = self.fc1s( states )

        # compute temporal state
        xt, h, msks = self.rnn( xs,  past )
        xh = self.fc1h( xt )
        
        # residual sum ( current state + temporal state )
        x = xh + xs
        if self.log: x = self.dp1( x )
        
        x = self.fc2( x )
        if self.log: x = self.dp2( x )        
        
        pred = self.fc3( x )

        return tf.reshape( pred, [ bs, ss, self.action_dim, self.atoms ] ), h, msks
    
    def quantile_huber_loss(self, target, pred, actions):
        
        pred = tf.reduce_sum( pred * tf.expand_dims( actions, -1 ), axis = 2 )
        pred_tile = tf.tile( tf.expand_dims( pred, axis = 3 ), [ 1, 1, 1, self.atoms ] )
        
        target_tile = tf.cast( tf.tile( tf.expand_dims( target, axis = 2 ), [ 1, 1, self.atoms, 1 ] ), tf.float32 )
        
        # huber_loss = tf.math.square( pred_tile - target_tile )
        huber_loss = tf.compat.v1.losses.huber_loss( target_tile, pred_tile, reduction = tf.keras.losses.Reduction.NONE )
        
        tau = tf.cast( tf.reshape( np.array( self.tau ), [ 1, self.atoms ] ), tf.float32 )
        inv_tau = tf.cast( 1.0 - tau, tf.float32 )
        tau = tf.cast( tf.tile( tf.expand_dims( tau, axis = 1 ), [ 1, self.atoms, 1 ] ), tf.float32 )
        inv_tau = tf.cast( tf.tile( tf.expand_dims( inv_tau, axis = 1 ), [ 1, self.atoms, 1 ] ), tf.float32 )
        
        error_loss = tf.math.subtract( target_tile, pred_tile )
        loss = tf.where( tf.less( error_loss, 0.0 ), inv_tau[:,tf.newaxis,:,:] * huber_loss, tau[:,tf.newaxis,:,:] * huber_loss )
        loss = tf.reduce_sum( tf.reduce_mean( loss, axis = 3 ), axis = 2 )
        
        return loss

    def train(self, states, target, actions, rewards, dones):

        with tf.GradientTape() as tape:

            theta, _, msks = self( states )
            
            huber_loss = self.quantile_huber_loss( target, theta, actions ) * tf.cast( rewards != 0, tf.float32 )
            loss = self.reduce( huber_loss )

            t_loss = loss
            
        gradients = tape.gradient( t_loss, self.trainable_variables )
        self.optimizer.apply_gradients( zip( gradients, self.trainable_variables ) )

        self.train_loss( loss )
        self.train_l2_loss( 0 )

        return huber_loss, tf.reduce_mean( theta, axis = -1 ), msks


class C51Network(tf.Module):

    def __init__(self, state_dim, action_dim, name, f1, f2, atoms, z, train=False):
        
        super(C51Network, self).__init__()
        
        self.module_type = 'C51Net'

        self.atoms = atoms
        self.action_dim = action_dim
        self.z = z
        self.tau = [ ( 2 * ( i - 1 ) + 1 ) / ( 2 * self.atoms ) for i in range( 1, self.atoms + 1 ) ]
        
        self.fc1 = Dense( f1, kernel_initializer = tf.keras.initializers.random_normal() )
        self.fc2 = Dense( f2, kernel_initializer = tf.keras.initializers.random_normal() )
        self.outputs = []
        for _ in range( action_dim ):
            self.outputs.append( 
                Dense( self.atoms, activation='softmax', kernel_initializer = tf.keras.initializers.random_normal() )
             )

        self.log_dir = 'logs/c51net_{}'.format(name)

        if train:

            self.optimizer = Adam( tf.Variable( 2e-4 ) )
            self.criterion = tf.keras.losses.CategoricalCrossentropy()

            run_number = 0
            while os.path.exists(self.log_dir + str(run_number)):
                run_number += 1
            self.log_dir = self.log_dir + str(run_number)
            self.train_summary_writer = tf.summary.create_file_writer(self.log_dir)
        
        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.reward = tf.keras.metrics.Mean('reward', dtype=tf.float32)
        self.actions = [ tf.keras.metrics.Mean('act_{}'.format(x), dtype=tf.float32) for x in range(action_dim) ]

    def __call__(self, states):
        """Build a network that maps state -> action values."""
        x = tf.nn.elu( self.fc1( states ) )
        x = tf.nn.elu( self.fc2( x ) )
        return [ o( x ) for o in self.outputs ]

    def quantile_huber_loss(self, target, pred, actions):
        pred = tf.reduce_sum( pred * tf.expand_dims( actions, -1 ), axis = 1 )
        pred_tile = tf.tile( tf.expand_dims( pred, axis = 2), [ 1, 1, self.atoms ] )
        
        target_tile = tf.cast( tf.tile( tf.expand_dims( target, axis = 1 ), [ 1, self.atoms, 1 ] ), tf.float32 )
        
        # huber_loss = self.huber_loss( target_tile, pred_tile )
        huber_loss = tf.math.square( pred_tile - target_tile )
        
        tau = tf.cast( tf.reshape( np.array( self.tau ), [ 1, self.atoms ] ), tf.float32 )
        inv_tau = tf.cast( 1.0 - tau, tf.float32 )
        tau = tf.cast( tf.tile( tf.expand_dims( tau, axis = 1 ), [ 1, self.atoms, 1 ] ), tf.float32 )
        inv_tau = tf.cast( tf.tile( tf.expand_dims( inv_tau, axis = 1 ), [ 1, self.atoms, 1 ] ), tf.float32 )
        
        error_loss = tf.math.subtract( target_tile, pred_tile )
        loss = tf.where( tf.less( error_loss, 0.0 ), inv_tau * huber_loss, tau * huber_loss )
        loss = tf.reduce_sum( tf.reduce_mean( loss, axis = 2 ), axis = 1 )
        
        return loss

    def train(self, x, y, w):
        
        y = tf.stop_gradient( y )

        with tf.GradientTape() as tape:
            logits = self( x )
            loss = self.criterion( y, logits, w )        
        gradients = tape.gradient( loss, self.trainable_variables )                
        self.optimizer.apply_gradients( zip( gradients, self.trainable_variables ) )

        self.train_loss(tf.reduce_mean(loss))

        return None


class IQNetwork(tf.Module):

    def __init__(self, state_dim, action_dim, name, f1, f2, atoms, quantile_dim, train=False):
        
        super(IQNetwork, self).__init__()
        
        self.module_type = 'IQNet'

        self.atoms = atoms
        self.quantile_dim = quantile_dim
        self.action_dim = action_dim
        
        self.fc1 = Dense( f1, kernel_initializer = tf.keras.initializers.random_normal() )
        self.fc2 = Dense( f2, kernel_initializer = tf.keras.initializers.random_normal() )

        self.phi = Dense( self.quantile_dim, activation = None, use_bias = False )
        self.phi_bias = tf.cast( tf.Variable( tf.zeros( self.quantile_dim ) ), tf.float32 )
        self.fc = Dense( 128, activation = 'relu' )

        self.generator = tf.random.Generator.from_non_deterministic_state()

        self.fc_q = Dense( action_dim, kernel_initializer = tf.keras.initializers.random_normal() )

        self.log_dir = 'logs/iqnet_{}'.format(name)

        if train:

            self.optimizer = Adam( tf.Variable( 2e-4 ) )
            self.huber_loss = tf.keras.losses.Huber( reduction = tf.keras.losses.Reduction.NONE )

            run_number = 0
            while os.path.exists(self.log_dir + str(run_number)):
                run_number += 1
            self.log_dir = self.log_dir + str(run_number)
            self.train_summary_writer = tf.summary.create_file_writer(self.log_dir)
        
        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.reward = tf.keras.metrics.Mean('reward', dtype=tf.float32)
        self.actions = [ tf.keras.metrics.Mean('act_{}'.format(x), dtype=tf.float32) for x in range(action_dim) ]

    def feature_extraction(self, x):
        x = tf.nn.elu( self.fc1( x ) )
        x = tf.nn.elu( self.fc2( x ) )
        return x

    def __call__(self, states):
        """Build a network that maps state -> action values."""
        x = self.feature_extraction( states )
        
        feature_dim = x.shape[1]
        tau = self.generator.uniform( shape = [ self.atoms, 1 ], minval = 0, maxval = 1 )
        pi_mtx = tf.constant( np.expand_dims( np.pi * tf.range( 0, self.quantile_dim, dtype = tf.float32 ), axis = 0 ) )
        cos_tau = tf.cos( tf.matmul( tau, pi_mtx ) )
        phi = tf.nn.relu( self.phi( cos_tau ) + tf.expand_dims( self.phi_bias, axis = 0 ) )
        phi = tf.expand_dims( phi, axis = 0 )
        x = tf.reshape( x, ( -1, feature_dim ) )
        x = tf.expand_dims( x, 1 )
        x = x * phi
        x = self.fc( x )
        x = self.fc_q( x )
        q = tf.transpose( x, [ 0, 2, 1 ] )

        return q, tau

    def quantile_huber_loss(self, target, pred, actions, tau):
        
        pred = tf.reduce_sum( pred * tf.expand_dims( actions, -1 ), axis = 1 )
        pred_tile = tf.tile( tf.expand_dims( pred, axis = 2), [ 1, 1, self.atoms ] )
        
        target_tile = tf.cast( tf.tile( tf.expand_dims( target, axis = 1 ), [ 1, self.atoms, 1 ] ), tf.float32 )
        
        # huber_loss = self.huber_loss( target_tile, pred_tile )
        huber_loss = tf.compat.v1.losses.huber_loss( target_tile, pred_tile, reduction = tf.keras.losses.Reduction.NONE )
        # tf.math.square( pred_tile - target_tile )
        
        tau = tf.cast( tf.reshape( tau, [ 1, self.atoms ] ), tf.float32 )
        inv_tau = tf.cast( 1.0 - tau, tf.float32 )
        tau = tf.cast( tf.tile( tf.expand_dims( tau, axis = 1 ), [ 1, self.atoms, 1 ] ), tf.float32 )
        inv_tau = tf.cast( tf.tile( tf.expand_dims( inv_tau, axis = 1 ), [ 1, self.atoms, 1 ] ), tf.float32 )
        
        error_loss = tf.math.subtract( target_tile, pred_tile )
        loss = tf.where( tf.less( error_loss, 0.0 ), inv_tau * huber_loss, tau * huber_loss )
        loss = tf.reduce_sum( tf.reduce_mean( loss, axis = 2 ), axis = 1 )
        
        return loss

    def train(self, states, target, actions, w):
        
        with tf.GradientTape() as tape:
            theta, taus = self( states )
            huber_loss = self.quantile_huber_loss( target, theta, actions, taus ) * w
            loss = tf.reduce_mean( huber_loss )        
        gradients = tape.gradient( loss, self.trainable_variables )                
        self.optimizer.apply_gradients( zip( gradients, self.trainable_variables ) )

        self.train_loss(tf.reduce_mean(loss))

        return huber_loss


class FPNetwork(tf.Module):

    def __init__(self, name, f1, f2, tau, entropy_coeff, train=False):
        
        super(FPNetwork, self).__init__()
        
        self.module_type = 'FPNet'

        self.tau = tau
        self.entropy_coeff = entropy_coeff
        
        self.fc1 = Dense( f1, kernel_initializer = tf.keras.initializers.random_normal() )
        self.fc2 = Dense( f2, kernel_initializer = tf.keras.initializers.random_normal() )
        self.fc3 = Dense( tau, kernel_initializer = tf.keras.initializers.random_normal() )

        self.log_dir = 'logs/fpnet_{}'.format(name)

        if train:

            self.optimizer = Adam( tf.Variable( 2e-4 ) )
            self.huber_loss = tf.keras.losses.Huber( reduction = tf.keras.losses.Reduction.NONE )

            run_number = 0
            while os.path.exists(self.log_dir + str(run_number)):
                run_number += 1
            self.log_dir = self.log_dir + str(run_number)
            self.train_summary_writer = tf.summary.create_file_writer(self.log_dir)
        
        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.reward = tf.keras.metrics.Mean('reward', dtype=tf.float32)
        self.actions = [ tf.keras.metrics.Mean('act_{}'.format(x), dtype=tf.float32) for x in range(action_dim) ]

    def __call__(self, states):
                 
        x = tf.nn.elu( self.fc1( states ) )
        x = tf.nn.elu( self.fc2( x ) )
        
        q = tf.nn.log_softmax( self.fc3( x ) )
        q_probs = tf.exp( q )
        taus = tf.math.cumsum( q_probs, axis = 1 )
        taus = tf.concat( [ tf.zeros( tf.shape( q )[0], 1 ) ], taus, axis = 1 )
        taus_ = tf.stop_gradient( taus[:, :-1] + taus[:, 1:] ) / 2.

        entropy = tf.reduce_sum( -(q * q_probs), axis = -1, keepdims=True )
    
        return taus, taus_, entropy

    @staticmethod
    def calc_fraction_loss(FZ_,FZ, taus, weights=None):

        """calculate the loss for the fraction proposal network """
        
        gradients1 = FZ - FZ_[:, :-1]
        gradients2 = FZ - FZ_[:, 1:] 
        
        flag_1 = FZ > tf.concat( [ FZ_[:, :1], FZ[:, :-1] ], axis = 1 )
        flag_2 = FZ < tf.concat( [ FZ[:, 1:], FZ_[:, -1:] ], axis = 1 )

        gradients = ( tf.where( flag_1, gradients1, - gradients1 ) + tf.where( flag_2, gradients2, -gradients2 ) )
        gradients = tf.reshape( gradients, [ tf.shape(taus)[0], gradients.shape[1] ] )
        
        if weights != None:
            loss = tf.reduce_mean( ( tf.reduce_sum( gradients * taus[:, 1:-1], axis = 1 ) * weights ) )
        else:
            loss = tf.reduce_mean( tf.reduce_sum( gradients * taus[:, 1:-1], axis = 1 ) )
        return loss
    
    def get_gradients(self, states, actions, quantiles_f):
        
        with tf.GradientTape() as tape:

            # calc fractional loss 
            taus, taus_, entropy = self( states )
            
            F_Z_expected = quantiles_f( taus_, states )
            Q_expected = gather2( F_Z_expected, actions )

            F_Z_tau = quantiles_f( taus[:, 1:-1], states )
            FZ_tau = tf.stop_gradient( gather2( F_Z_tau, actions ) )
                            
            frac_loss = FPNetwork.calc_fraction_loss( tf.stop_gradient( Q_expected ), FZ_tau, taus )
            entropy_loss = self.entropy_coeff * tf.reduce_mean( entropy ) 
            frac_loss += entropy_loss
            
        gradients = tape.gradient( frac_loss, self.trainable_variables )                

        self.train_loss( tf.reduce_mean( frac_loss ) )

        return gradients
    
    def train(self, gradients):
        self.optimizer.apply_gradients( zip( gradients, self.trainable_variables ) )


class QuantileNetwork(tf.Module):

    def __init__(self, name, action_dim, f1, f2, n_cos, gamma, n_step, train=False):
        
        super(QuantileNetwork, self).__init__()
        
        self.module_type = 'QuantileNet'

        self.n_cos = n_cos
        self.gamma = gamma
        self.n_step = n_step
        self.action_dim = action_dim
        
        self.fc1 = Dense( f1, kernel_initializer = tf.keras.initializers.random_normal() )
        self.fc2 = Dense( f2, kernel_initializer = tf.keras.initializers.random_normal() )
        
        self.fembeding = Dense( f2, kernel_initializer = tf.keras.initializers.random_normal() )
        self.fc3 = Dense( f2, kernel_initializer = tf.keras.initializers.random_normal() )
        
        self.fc4 = Dense( action_dim, kernel_initializer = tf.keras.initializers.random_normal() )

        self.log_dir = 'logs/quantilenet_{}'.format(name)

        if train:

            self.optimizer = Adam( tf.Variable( 2e-4 ) )
            self.huber_loss = tf.keras.losses.Huber( reduction = tf.keras.losses.Reduction.NONE )

            run_number = 0
            while os.path.exists(self.log_dir + str(run_number)):
                run_number += 1
            self.log_dir = self.log_dir + str(run_number)
            self.train_summary_writer = tf.summary.create_file_writer(self.log_dir)
        
        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.reward = tf.keras.metrics.Mean('reward', dtype=tf.float32)
        self.actions = [ tf.keras.metrics.Mean('act_{}'.format(x), dtype=tf.float32) for x in range(action_dim) ]

    @staticmethod
    def calc_cos(taus, pis, n_cos):
        """
        Calculating the cosinus values depending on the number of tau samples
        """
        cos = tf.math.cos( taus[...,tf.newaxis] * pis )
        return cos

    @staticmethod
    def calculate_huber_loss(td_errors, k=1.0):
        """
        Calculate huber loss element-wisely depending on kappa k.
        """
        loss = tf.where( tf.abs( td_errors ) <= k, 0.5 * tf.pow( td_errors, 2 ), k * ( tf.abs( td_errors ) - 0.5 * k ) )
        return loss 

    def __call__(self, taus, states):
                 
        x = tf.nn.elu( self.fc1( states ) )
        x = tf.nn.elu( self.fc2( x ) )

        pis = tf.constant( [ np.pi * i for i in range( 1, self.n_cos + 1 ) ] )[ tf.newaxis, tf.newaxis, : ] # Starting from 0 as in the paper 
        
        batch_size = tf.shape(x)[0]
        emb_size = x.shape[1]
        num_tau = taus.shape[1]
        cos = QuantileNetwork.calc_cos( taus, pis, self.n_cos ) # cos shape ( batch, num_tau, layer_size )
        cos = tf.reshape( cos, [ batch_size * num_tau, self.n_cos ] )

        cos_emb = self.fembeding( cos )
        cos_x = tf.nn.relu( cos_emb )
        cos_x = tf.reshape( cos_x, [ batch_size, num_tau, emb_size ] ) # ( batch, n_tau, layer )
        
        # x has shape (batch, layer_size) for multiplication –> reshape to (batch, 1, layer)
        x = x[:,tf.newaxis,:] * cos_x
        x = tf.reshape( x, [ batch_size * num_tau, emb_size ] )  
        x = tf.nn.relu( self.fc3( x ) )

        out = self.fc4( x )

        return tf.reshape( out, [ batch_size, num_tau, self.action_dim ] )
            
    def get_gradients(self, states, next_state, actions, rewards, dones, fpn, quantiles_f):
        
        with tf.GradientTape() as tape:

            _, taus_, _ = fpn( states )
            F_Z_expected = self( taus_, states )

            n_taus, n_taus_, _ = fpn( next_state )
            F_Z_next = self( n_taus_, next_state )

            Q_expected = gather( F_Z_expected, actions )
            
            Q_targets_next = tf.reduce_sum( ( ( n_taus[:, 1:, tf.newaxis] - n_taus[:, :-1, tf.newaxis] ) ) * F_Z_next, axis = 1 )
            action_indx = tf.argmax( Q_targets_next, axis = 1 )

            F_Z_next = quantiles_f( taus_, next_state )
            Q_targets_next = gather( F_Z_next, action_indx )
            Q_targets = tf.cast( rewards[:,tf.newaxis], tf.float32 ) + ( self.gamma ** self.n_step * Q_targets_next * tf.cast( 1 - dones[:,tf.newaxis], tf.float32 ) )
                
            # Quantile Huber loss
            td_error = Q_expected - Q_targets     
            huber_l = QuantileNetwork.calculate_huber_loss( td_error, 1.0 )
            quantil_l = abs( taus_ - tf.cast( tf.stop_gradient( td_error ) < 0, tf.float32 ) ) * huber_l / 1.0

            loss = tf.reduce_mean( tf.reduce_sum( quantil_l, axis = 1 ) )
                       
        gradients = tape.gradient( loss, self.trainable_variables )

        self.train_loss( tf.reduce_mean( frac_loss ) )

        return gradients
    
    def train(self, gradients):
        self.optimizer.apply_gradients( zip( gradients, self.trainable_variables ) )


class QNetworkVision(tf.Module):

    def __init__(self, state_dim, action_dim, name):
        
        super(QNetworkVision, self).__init__()
        
        self.module_type = 'QNetVision'

        self.c1 = Conv2D( 32, 5, 4, padding = 'same', kernel_initializer = tf.keras.initializers.orthogonal )
        self.c2 = Conv2D( 32, 5, 2, padding = 'same', kernel_initializer = tf.keras.initializers.orthogonal )
        self.c3 = Conv2D( 32, 3, 2, padding = 'same', kernel_initializer = tf.keras.initializers.orthogonal )

        self.fc1 = Dense( 128, kernel_initializer = tf.keras.initializers.orthogonal )
        self.fc2 = Dense( 32, kernel_initializer = tf.keras.initializers.orthogonal )
        self.fc3 = Dense( action_dim, kernel_initializer = tf.keras.initializers.orthogonal )

        self.optimizer = tf.keras.optimizers.Adam( 5e-4 )

        self.log_dir = 'logs/qnet_{}'.format(name)

        run_number = 0
        while os.path.exists(self.log_dir + str(run_number)):
            run_number += 1
        self.log_dir = self.log_dir + str(run_number)
        self.train_summary_writer = tf.summary.create_file_writer(self.log_dir)
        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.reward = tf.keras.metrics.Mean('reward', dtype=tf.float32)
        self.actions = [ tf.keras.metrics.Mean('act_{}'.format(x), dtype=tf.float32) for x in range(action_dim) ]

    def __call__(self, states):        
        """Build a network that maps state -> action values."""

        x = tf.nn.elu( self.c1( states ) )
        x = tf.nn.elu( self.c2( x ) )
        x = tf.nn.elu( self.c3( x ) )

        features = flatten( x )

        x = tf.nn.elu( self.fc1( features ) )
        x = tf.nn.elu( self.fc2( x ) )

        return self.fc3( x )

    def train(self, states, next_states, rewards, actions, dones, gamma, target_net, w):
        
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = tf.reduce_max( target_net( next_states ), axis = -1 )
        
        # Compute Q targets for current states 
        Q_targets = tf.cast( rewards, tf.float32 ) + ( gamma * Q_targets_next * ( 1 - tf.cast( dones, tf.float32 ) ) )

        with tf.GradientTape() as tape:
            # Get expected Q values from local model
            Q_expected = gather( self( states ), actions )
            mse = tf.math.square( Q_expected - Q_targets ) * w
            loss = tf.reduce_mean( mse )
            # loss = tf.keras.losses.mse( Q_targets, Q_expected )
        
        gradients = tape.gradient( loss, self.trainable_variables )
        self.optimizer.apply_gradients( zip( gradients, self.trainable_variables ) )

        self.train_loss(tf.reduce_mean(loss))
        self.reward(tf.reduce_mean(rewards))

        values = self( states )
        for i, a in enumerate( self.actions ):
            a( tf.reduce_mean( gather( values, tf.repeat( i, values.shape[0] ) ) ) )

        return mse


class Vae(tf.Module):

    def __init__(self, name):
        
        super(Vae, self).__init__()
        
        self.module_type = 'Vae'

        self.enc1 = Conv2D( 32, 5, 4, padding = 'SAME', kernel_initializer = tf.keras.initializers.orthogonal )
        self.enc2 = Conv2D( 32, 3, 4, padding = 'SAME', kernel_initializer = tf.keras.initializers.orthogonal )
        self.enc3 = Conv2D( 32, 3, 2, padding = 'SAME', kernel_initializer = tf.keras.initializers.orthogonal )
        self.ph = Dense( 288, kernel_initializer = tf.keras.initializers.orthogonal )

        self.logvar = Dense( 288, kernel_initializer = tf.keras.initializers.orthogonal )
        self.mu = Dense( 288, kernel_initializer = tf.keras.initializers.orthogonal )

        self.dec1 = Conv2DTranspose( 32, 5, 4, padding = 'SAME', kernel_initializer = tf.keras.initializers.orthogonal )
        self.dec2 = Conv2DTranspose( 32, 3, 4, padding = 'SAME', kernel_initializer = tf.keras.initializers.orthogonal )
        self.dec3 = Conv2DTranspose( 32, 3, 2, padding = 'SAME', kernel_initializer = tf.keras.initializers.orthogonal )
        self.rec = Conv2D( 3, 3, 1, padding = 'SAME', kernel_initializer = tf.keras.initializers.orthogonal )

        self.dp = Dropout( .5 )

        self.optimizer = tf.keras.optimizers.Adam( 5e-4 )

        self.log_dir = 'logs/vae_{}'.format(name)

        run_number = 0
        while os.path.exists(self.log_dir + str(run_number)):
            run_number += 1
        self.log_dir = self.log_dir + str(run_number)
        self.train_summary_writer = tf.summary.create_file_writer(self.log_dir)
        self.rec_loss = tf.keras.metrics.Mean('rec_loss', dtype=tf.float32)
        self.lat_loss = tf.keras.metrics.Mean('lat_loss', dtype=tf.float32)

    def encoder(self, x):
        
        x = tf.nn.elu( self.enc1( x ) )
        x = tf.nn.elu( self.enc2( x ) )
        x = tf.nn.elu( self.enc3( x ) )

        features = flatten( x )

        h = tf.nn.elu( self.ph( features ) )

        return h, tf.shape( x )

    def decoder(self, x, shape):

        x =  tf.reshape( x, shape )
        
        x = tf.nn.elu( self.dec1( x ) )
        x = tf.nn.elu( self.dec2( x ) )
        x = tf.nn.elu( self.dec3( x ) )

        rec = self.rec( x )

        return rec

    def lattent_space(self, x):
        
        mu = self.mu( x )
        logvar = self.logvar( x )

        return mu, logvar

    def reparameterize(self, mu, logvar, samples):
        
        samples_z = []
        std = 0.5 * tf.exp( logvar )

        for _ in range(samples):            
            eps = tf.compat.v1.random_normal( shape = tf.shape( std ), mean = 0, stddev = 1, dtype = tf.float32 )
            z = mu + ( eps * std )
            samples_z.append( z )

        return samples_z

    def __call__(self, states, dp):

        states = states / 255.0    
        features, _ = self.encoder( states )
        mu, _ = self.lattent_space( features )
        
        if dp: return self.dp( mu )
        
        return mu

    def train(self, states):

        states = states / 255.0

        with tf.GradientTape() as tape:

            features, shape = self.encoder( states )
            mu, logvar = self.lattent_space( features )
            rp = self.reparameterize( mu, logvar, 4 )
            
            _, var = tf.nn.moments( states, axes = [ 0,1,2,3 ] )
            
            dec_loss = []
            dec_decoded = []
            for r in rp:
                predicts = self.decoder( self.dp( r ), shape )
                dec_loss.append( tf.reduce_mean( tf.math.square( predicts - states ) ) / var )
                dec_decoded.append( predicts )
            rec_loss = tf.reduce_mean( dec_loss )

            lat_loss = -0.5 * tf.reduce_mean( 1.0 + logvar - tf.pow( mu, 2 ) - tf.exp( logvar ), axis = 1 )
            lat_loss /= tf.cast( logvar.shape[-1], tf.float32 )
            lat_loss = tf.reduce_mean( lat_loss )
            
            l2_loss = ( tf.reduce_sum( tf.math.square( self.enc1.weights[0] ) ) +
                        tf.reduce_sum( tf.math.square( self.enc2.weights[0] ) ) +
                        tf.reduce_sum( tf.math.square( self.enc3.weights[0] ) ) +
                        tf.reduce_sum( tf.math.square( self.dec1.weights[0] ) ) +
                        tf.reduce_sum( tf.math.square( self.dec2.weights[0] ) ) +
                        tf.reduce_sum( tf.math.square( self.dec3.weights[0] ) ) +
                        tf.reduce_sum( tf.math.square( self.rec.weights[0] ) ) )

            loss = rec_loss + lat_loss + ( 2e-5 * l2_loss )
        
        gradients = tape.gradient( loss, self.trainable_variables )
        self.optimizer.apply_gradients( zip( gradients, self.trainable_variables ) )

        self.rec_loss( rec_loss )
        self.lat_loss( lat_loss )

        return dec_decoded


class QAttnNetwork(tf.Module):

    def __init__(self, state_dim, action_dim, name, f1, f2, c_size, out_size, n_heads=4, train=False):
        
        super(QAttnNetwork, self).__init__()
        
        self.module_type = 'QNet'
        
        self.qkv = [ conv1d( c_size * 3, 'qkv1' ),
                     conv1d( c_size * 3, 'qkv2' ),
                     conv1d( c_size * 3, 'qkv3' ), ]

        self.norm1 = [ norm(),
                       norm(),
                       norm(), ]

        self.norm2 = [ norm(),
                       norm(),
                       norm(), ]

        self.ln1 = [ conv1d( c_size * 4, 'ln11' ),
                     conv1d( c_size * 4, 'ln12' ),
                     conv1d( c_size * 4, 'ln13' ), ]

        self.ln2 = [ conv1d( c_size, 'ln21' ),
                     conv1d( c_size, 'ln22' ),
                     conv1d( c_size, 'ln23' ), ]

        self.merge = [ conv1d( c_size, 'merge1' ),
                       conv1d( c_size, 'merge2' ),
                       conv1d( out_size, 'merge3' ), ]
        
        self.n_heads = n_heads

        self.fc1 = Dense( f1, kernel_initializer = tf.keras.initializers.random_normal() )
        self.fc2 = Dense( f2, kernel_initializer = tf.keras.initializers.random_normal() )
        self.fc3 = Dense( action_dim, kernel_initializer = tf.keras.initializers.random_normal() )
        self.log_dir = 'logs/attn_qnet_{}'.format(name)

        run_number = 0
        while os.path.exists(self.log_dir + str(run_number)):
            run_number += 1
        
        if train:
            
            self.optimizer = Adam( tf.Variable( 2e-4 ) )
            self.generator = tf.random.Generator.from_non_deterministic_state()

            self.log_dir = self.log_dir + str(run_number)
            self.train_summary_writer = tf.summary.create_file_writer(self.log_dir)

        self.dqn_loss = tf.keras.metrics.Mean('dqn_loss', dtype=tf.float32)
        self.reward = tf.keras.metrics.Mean('reward', dtype=tf.float32)
        self.reward_done = tf.keras.metrics.Mean('reward_done', dtype=tf.float32)

    @staticmethod
    def merge_states(x):
        *start, a, b = shape_list(x)
        return tf.reshape(x, start + [a*b])

    @staticmethod
    def split_states(x, n):
        *start, m = shape_list(x)
        return tf.reshape(x, start + [n, m//n])

    @staticmethod
    def split_heads(x, n_head):
        # From [batch, sequence, features] to [batch, heads, sequence, features]
        return tf.transpose(QAttnNetwork.split_states(x, n_head), [0, 2, 1, 3])

    @staticmethod
    def merge_heads(x):
        # Reverse of split_heads
        return QAttnNetwork.merge_states(tf.transpose(x, [0, 2, 1, 3]))

    @staticmethod
    def multihead_attn(q, k, v):
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul( q, k, transpose_b = True )
        w = w * tf.math.rsqrt( tf.cast( v.shape[-1], w.dtype ) )
        w = tf.nn.softmax( w )
        a = tf.matmul( w, v )
        return a, w
    
    @staticmethod
    def attn(qkv, merge, features, n_head):
        c = qkv( features )
        q, k, v = map( partial( QAttnNetwork.split_heads, n_head = n_head ), tf.split( c, 3, axis = 2 ) )        
        a, msk = QAttnNetwork.multihead_attn( q, k, v )
        a = QAttnNetwork.merge_heads( a )        
        a = merge( a )
        return a, msk

    @staticmethod
    def mlp(x, ln1, ln2):
        x = gelu( ln1( x ) )
        x = ln2( x )
        return x

    @staticmethod
    def block(qkv, merge, norm1, norm2, ln1, ln2, x, n_head):
        a, msk = QAttnNetwork.attn( qkv, merge, x, n_head )
        ax = x + a
        m = QAttnNetwork.mlp( ax, ln1, ln2 )
        mx = ax + m
        return mx, msk

    def get_features(self, states):

        s = states

        b1, msk1 = self.block( self.qkv[0], self.merge[0], self.norm1[0], self.norm2[0], self.ln1[0], self.ln2[0], s, self.n_heads )
        b2, msk2 = self.block( self.qkv[1], self.merge[1], self.norm1[1], self.norm2[1], self.ln1[1], self.ln2[1], b1, self.n_heads )
        b3, msk3 = self.block( self.qkv[2], self.merge[2], self.norm1[2], self.norm2[2], self.ln1[2], self.ln2[2], b2, self.n_heads )

        features = tf.reduce_max( b3, axis = 1 )

        return features, ( msk1, msk2, msk3 )

    def __call__(self, states):
        
        """Build a network that maps state -> action values."""

        features, _ = self.get_features( states )

        x = tf.nn.elu( self.fc1( features ) )
        x = tf.nn.elu( self.fc2( x ) )
        
        return self.fc3( x )
    
    def train(self, states, next_states, rewards, actions, dones, gamma, target_net, w, p_loss):
        
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = tf.reduce_max( target_net( next_states ), axis = -1 )
        
        # Compute Q targets for current states 
        Q_targets = tf.cast( rewards, tf.float32 ) + ( gamma * Q_targets_next * ( 1 - tf.cast( dones, tf.float32 ) ) )

        _, msk = self.get_features( states )

        with tf.GradientTape() as tape:
            # Get expected Q values from local model
            Q_expected = gather( self( states ), actions )
            mse = tf.math.square( Q_expected - Q_targets ) * w
            loss = tf.reduce_mean( mse ) # / tf.convert_to_tensor( p_loss + 1.0, tf.float32 )
        gradients = tape.gradient( loss, self.trainable_variables )                
        self.optimizer.apply_gradients( zip( gradients, self.trainable_variables ) )

        self.dqn_loss(tf.reduce_mean(loss))
        self.reward(tf.reduce_mean(rewards))
        self.reward_done(tf.reduce_mean(tf.cast( rewards, tf.float32 ) * tf.cast( dones, tf.float32 )))
        
        return mse, msk


class VQ(tf.Module):

    def __init__(self, embedding_dim, num_embeddings, commitment_cost, epsilon, name):
        
        super(VQ, self).__init__()
        
        self.module_type = 'Vae'

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.epsilon = epsilon

        self.optimizer = Adam( tf.Variable( 2e-4 ) )

        self.w = tf.compat.v1.get_variable( "vq_w", [ embedding_dim, num_embeddings ], dtype = tf.float32, 
                                            initializer = tf.compat.v1.uniform_unit_scaling_initializer(), trainable = True )

    def __call__(self, x, flat=False):

        if not flat:
            # Flatten input except for last dimension.
            flat_inputs = tf.reshape( x, [ -1, self.embedding_dim ] )
        else:
            flat_inputs = x

        # Calculate distances of input to embedding vectors.
        v0 = tf.reduce_sum( tf.math.square( flat_inputs ), axis = 1, keepdims = True )
        v1 = tf.matmul( flat_inputs, self.w )
        v2 = tf.reduce_sum( tf.square( self.w ), axis = 0, keepdims = True )
        distances = v0 - 2 * v1 + v2

        # Retrieve encoding indices.
        encoding_indices = tf.argmax( -distances, 1 )
        encodings = tf.one_hot( encoding_indices, self.num_embeddings, on_value = 1.0, off_value = 0.0 )

        if not flat:
            encoding_indices = tf.reshape( encoding_indices, tf.shape( x )[:-1] ) # shape=[batch_size, ?, ?]

        # quantized is differentiable w.r.t. tf.transpose(emb_vq),  but is not differentiable w.r.t. encoding_indices.    
        quantized = tf.nn.embedding_lookup( tf.transpose( self.w ), encoding_indices )

        # This step is used to copy the gradient from inputs to quantized.
        quantized_skip_gradient = x + tf.stop_gradient( quantized - x )

        return quantized, quantized_skip_gradient, encodings, encoding_indices

    def train(self, x, compute_gradients=False):

        if compute_gradients:

            with tf.GradientTape() as tape:
            
                quantized, _, encodings, _ = self( x )

                # loss used to optimize w only!
                emb_latent_loss = tf.reduce_mean( tf.math.square( quantized - tf.stop_gradient( x ) ) )
                inp_latent_loss = tf.reduce_mean( tf.math.square( x - tf.stop_gradient( quantized ) ) )
                loss = emb_latent_loss + self.commitment_cost * inp_latent_loss
            
            gradients = tape.gradient( loss, self.trainable_variables )
            self.optimizer.apply_gradients( zip( gradients, self.trainable_variables ) )

        else:
            
            quantized, _, encodings, _ = self( x )

            # loss used to optimize w only!
            emb_latent_loss = tf.reduce_mean( tf.math.square( quantized - tf.stop_gradient( x ) ) )
            inp_latent_loss = tf.reduce_mean( tf.math.square( x - tf.stop_gradient( quantized ) ) )
            loss = emb_latent_loss + self.commitment_cost * inp_latent_loss
            

        # The perplexity is the exponentiation of the entropy, 
        # indicating how many codes are 'active' on average.
        # We hope the perplexity is larger.
        avg_probs = tf.reduce_mean( encodings, 0 )
        perplexity = tf.exp( - tf.reduce_sum( avg_probs * tf.math.log( avg_probs + self.epsilon ) ) )

        return loss, perplexity


class VQVae(tf.Module):

    def __init__(self, embedding_dim, num_embeddings, commitment_cost, epsilon, name):
        
        super(VQVae, self).__init__()
        
        self.module_type = 'VQVae'

        self.enc1 = Conv2D( 32, 5, 2, padding = 'SAME', kernel_initializer = tf.keras.initializers.he_normal() )
        self.enc2 = Conv2D( 32, 3, 2, padding = 'SAME', kernel_initializer = tf.keras.initializers.he_normal() )
        self.enc3 = Conv2D( 32, 3, 2, padding = 'SAME', kernel_initializer = tf.keras.initializers.he_normal() )
        self.pre_embeding = Conv2D( embedding_dim, 1, 1, padding = 'SAME', kernel_initializer = tf.keras.initializers.he_normal() )

        self.vq = VQ( embedding_dim, num_embeddings, commitment_cost, epsilon, '{}_vq'.format( name ) )

        self.pre_rec = Conv2DTranspose( 32, 3, 1, padding = 'SAME', kernel_initializer = tf.keras.initializers.glorot_normal() )
        self.dec1 = Conv2DTranspose( 32, 5, 2, padding = 'SAME', kernel_initializer = tf.keras.initializers.glorot_normal() )
        self.dec2 = Conv2DTranspose( 32, 3, 2, padding = 'SAME', kernel_initializer = tf.keras.initializers.glorot_normal() )
        self.dec3 = Conv2DTranspose( 32, 3, 2, padding = 'SAME', kernel_initializer = tf.keras.initializers.glorot_normal() )
        self.rec = Conv2D( 3, 3, 1, padding = 'SAME', kernel_initializer = tf.keras.initializers.glorot_normal() )

        self.dp = Dropout( .5 )

        lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay( initial_learning_rate = 5e-4,
                                                                       decay_steps = 10000,
                                                                       decay_rate = math.exp(-1),
                                                                       staircase = False )

        adam = tf.keras.optimizers.Adam( learning_rate = lr_scheduler, beta_1 = tf.Variable(0.9), beta_2 = tf.Variable(0.999), epsilon = tf.Variable(1e-7) )
        self.optimizer = LossScaleOptimizer( adam )

        self.log_dir = 'logs/vae_{}'.format(name)

        run_number = 0
        while os.path.exists(self.log_dir + str(run_number)):
            run_number += 1
        self.log_dir = self.log_dir + str(run_number)
        self.train_summary_writer = tf.summary.create_file_writer(self.log_dir)
        self.rec_loss = tf.keras.metrics.Mean('rec_loss', dtype=tf.float32)
        self.lat_loss = tf.keras.metrics.Mean('lat_loss', dtype=tf.float32)
        self.perplexity = tf.keras.metrics.Mean('perplexity', dtype=tf.float32)

    def encoder(self, x):

        x = tf.nn.elu( self.enc1( x ) )
        x = tf.nn.elu( self.enc2( x ) )
        x = tf.nn.elu( self.enc3( x ) )
        x = self.pre_embeding( x )
        return x

    def decoder(self, x):
        x = self.pre_rec( x )
        x = tf.nn.elu( self.dec1( x ) )
        x = tf.nn.elu( self.dec2( x ) )
        x = tf.nn.elu( self.dec3( x ) )
        rec = self.rec( x )
        return rec

    def lattent_space(self, x):
        quantized, quantized_skip_gradient, encodings, encoding_indices = self.vq( x )
        return quantized, quantized_skip_gradient, encodings, encoding_indices
    
    def __call__(self, states, dp):
        states = states / 255.0    
        features = self.encoder( states )
        _, q, _, _ = self.lattent_space( features )
        q = tf.reshape( q, [ tf.shape(states)[0], -1, self.vq.embedding_dim ] )
        if dp: return self.dp( q )        
        return q

    def train(self, states):

        states = states / 255.0
        
        with tf.GradientTape() as tape:

            features = self.encoder( states )

            # comput lattent loss
            lat_loss, perplexity = self.vq.train( features )

            _, q, _, _ = self.lattent_space( features )                        
            _, var = tf.nn.moments( states, axes = [ 0,1,2,3 ] )
            
            predicts = self.decoder( self.dp( q ) )
            rec_loss = tf.reduce_mean( tf.reduce_sum( tf.math.square( predicts - states ) ) / var )
            
            l2_loss = ( tf.reduce_sum( tf.math.square( self.enc1.weights[0] ) ) +
                        tf.reduce_sum( tf.math.square( self.enc2.weights[0] ) ) +
                        tf.reduce_sum( tf.math.square( self.enc3.weights[0] ) ) +
                        tf.reduce_sum( tf.math.square( self.pre_embeding.weights[0] ) ) +
                        tf.reduce_sum( tf.math.square( self.dec1.weights[0] ) ) +
                        tf.reduce_sum( tf.math.square( self.dec2.weights[0] ) ) +
                        tf.reduce_sum( tf.math.square( self.dec3.weights[0] ) ) +
                        tf.reduce_sum( tf.math.square( self.rec.weights[0] ) ) )

            loss = self.optimizer.get_scaled_loss( rec_loss + lat_loss + ( 2e-5 * l2_loss ) )
        
        gradients = self.optimizer.get_unscaled_gradients( tape.gradient( loss, self.trainable_variables ) )
        self.optimizer.apply_gradients( zip( gradients, self.trainable_variables ) )

        self.rec_loss( rec_loss )
        self.lat_loss( lat_loss )
        self.perplexity( perplexity )

        return predicts, features, q

   