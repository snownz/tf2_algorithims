from select import select
import tensorflow as tf 
import numpy as np
import math
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, Dropout, SimpleRNNCell, LSTMCell, GRUCell, RNN, BatchNormalization
from tensorflow.keras.mixed_precision import LossScaleOptimizer
from tensorflow.keras.activations import softmax, elu, relu, tanh, softplus
from tensorflow.keras.initializers import orthogonal, random_normal, truncated_normal
from tensorflow.keras.mixed_precision import experimental as prec
from tensorflow.python.ops.gen_math_ops import sigmoid
import tensorflow_probability as tfp
from math import ceil
from ann_utils import gather, flatten, conv1d, norm, shape_list, Adam, gelu,  nalu_transform, nalu_gru_cell, transformer_layer, dense, simple_rnn, conv2d, gru, convnalu2d, nalu, RMS, vector_quantizer
from cognitive_model import StateUnderstanding, GlobalMemory, FeatureCreator, Actor, Critic, FakeActor, StateUnderstandingMultiEnv_v3, Actorv2

from functools import partial

from random import randint, random

import os

#######################################################################################################################################
# PPO Discrete Section
#######################################################################################################################################

class SimplePPONetwork(tf.Module):

    def __init__(self, action_dim, name, lr, train=False):
        
        super(SimplePPONetwork, self).__init__()
        
        self.module_type = 'PPO'

        self.action_dim = action_dim
        
        # actor
        self.a_fc1 = dense( 512, 'a', activation = relu, kernel_initializer = orthogonal() )
        self.a_fc2 = dense( 256, 'a', activation = relu, kernel_initializer = orthogonal() )
        self.a_fc3 = dense( 64, 'a', activation = relu, kernel_initializer = orthogonal() )
        self.act = dense( action_dim, 'a', activation = softmax )

        # critic
        self.c_fc1 = dense( 512, 'c', activation = relu, kernel_initializer = tf.random_normal_initializer( stddev = 0.01 ) )
        self.c_fc2 = dense( 256, 'c', activation = relu, kernel_initializer = tf.random_normal_initializer( stddev = 0.01 ) )
        self.c_fc3 = dense( 64, 'c', activation = relu, kernel_initializer = tf.random_normal_initializer( stddev = 0.01 ) )
        self.val = dense( 1, 'c', activation = lambda x : x )

        self.log_dir = 'logs/ppo_{}'.format(name)

        self.a_optimizer = Adam( tf.Variable( lr ) )
        self.c_optimizer = Adam( tf.Variable( lr ) )

        run_number = 0
        while os.path.exists(self.log_dir + str(run_number)):
            run_number += 1
        self.log_dir = self.log_dir + str(run_number)
        self.train_summary_writer = tf.summary.create_file_writer(self.log_dir)
        
        self.a_loss = tf.keras.metrics.Mean( 'a_loss', dtype = tf.float32 )
        self.c_loss = tf.keras.metrics.Mean( 'c_loss', dtype = tf.float32 )
        self.reward = tf.keras.metrics.Mean( 'reward', dtype = tf.float32 )

    def probs(self, states):

        x = self.a_fc1( states )
        x = self.a_fc2( x )
        x = self.a_fc3( x )
        probs = self.act( x )

        return probs

    def value(self, states):

        x = self.c_fc1( states )
        x = self.c_fc2( x )
        x = self.c_fc3( x )
        values = self.val( x )

        return values

    def __call__(self, states):

        self.probs( states )
        self.value( states )

    def ppo_loss(self, advantages, prediction_picks, actions, y_pred):
        
        # Defined in https://arxiv.org/abs/1707.06347
        
        LOSS_CLIPPING = 0.2
        ENTROPY_LOSS = 0.001
        
        prob = actions * y_pred
        old_prob = actions * prediction_picks

        prob = tf.clip_by_value( prob, 1e-10, 1.0 )
        old_prob = tf.cast( tf.clip_by_value( old_prob, 1e-10, 1.0 ), tf.float32 )

        ratio = tf.math.exp( tf.math.log( prob ) - tf.math.log( old_prob ) )
        
        p1 = ratio * advantages
        p2 = tf.clip_by_value( ratio, 1 - LOSS_CLIPPING, 1 + LOSS_CLIPPING ) * advantages

        actor_loss = -tf.reduce_mean( tf.minimum( p1, p2 ) )

        entropy = -( y_pred * tf.math.log( y_pred + 1e-10 ) )
        entropy = ENTROPY_LOSS * tf.reduce_mean( entropy )
        
        total_loss = actor_loss - entropy

        return total_loss

    def critic_PPO2_loss(self, values, y_true, y_pred):
        
        LOSS_CLIPPING = 0.2
        clipped_value_loss = values + tf.clip_by_value( y_pred - values, -LOSS_CLIPPING, LOSS_CLIPPING )
        v_loss1 = ( y_true - clipped_value_loss ) ** 2
        v_loss2 = ( y_true - y_pred ) ** 2
        
        value_loss = 0.5 * tf.reduce_mean( tf.maximum( v_loss1, v_loss2 ) )
        #value_loss = K.mean((y_true - y_pred) ** 2) # standard PPO loss
        return value_loss

    def train(self, states, values, target, advantages, predictions, actions, rewards, epochs):

        a_vars = [ v for v in self.trainable_variables if 'a_' in v.name ]
        c_vars = [ v for v in self.trainable_variables if 'c_' in v.name ]

        for e in range( epochs ):

            with tf.GradientTape() as tape_c:
                c_pred = self.value( states )
                c_loss = self.critic_PPO2_loss( values, target, c_pred )
            c_gradients = tape_c.gradient( c_loss, c_vars )
            self.c_optimizer.apply_gradients( zip( c_gradients, c_vars ) )
            self.c_loss( c_loss )

            with tf.GradientTape() as tape_a:
                a_pred = self.probs( states )
                a_loss = self.ppo_loss( advantages, predictions, actions, a_pred )
            a_gradients = tape_a.gradient( a_loss, a_vars )
            self.a_optimizer.apply_gradients( zip( a_gradients, a_vars ) )
            self.a_loss( a_loss )

        self.reward( tf.reduce_mean( rewards ) )


class SimpleRNNPPONetwork(tf.Module):

    def __init__(self, action_dim, name, max_len, lr, typer='s', reduce='mean', sizes=[ 512, 256, 64 ]):
        
        super(SimpleRNNPPONetwork, self).__init__()
        
        self.module_type = 'RnnPPO'

        self.action_dim = action_dim
        self.typer = typer
        self.sizes = sizes
        self.r_sizes = sizes[1]
        
        # actor
        self.a_rnn = SimpleRNNPPONetwork.create_rnn( 'a', relu, self.sizes[1], typer, orthogonal(), True )
        self.a_fc1 = dense( self.sizes[0], 'a', activation = relu, kernel_initializer = orthogonal() )
        self.a_fc2 = dense( self.sizes[2], 'a', activation = relu, kernel_initializer = orthogonal() )
        self.act = dense( action_dim, 'a', activation = softmax )

        # critic
        self.c_rnn = SimpleRNNPPONetwork.create_rnn( 'c', tanh, self.sizes[1], typer, tf.random_normal_initializer( stddev = 0.01 ), True )
        self.c_fc1 = dense( self.sizes[0], 'c', activation = relu, kernel_initializer = tf.random_normal_initializer( stddev = 0.01 ) )
        self.c_fc2 = dense( self.sizes[2], 'c', activation = relu, kernel_initializer = tf.random_normal_initializer( stddev = 0.01 ) )
        self.val = dense( 1, 'c', activation = lambda x : x )

        # optmizer
        self.lr = tf.keras.optimizers.schedules.ExponentialDecay( lr, 1000, 0.96, staircase=False, name=None )
        self.a_optimizer = Adam( self.lr )
        self.c_optimizer = Adam( self.lr )

        self.reduce = SimpleRNNPPONetwork.sequence_reduce( reduce, max_len )

        # logs
        self.log_dir = 'logs/rnn_{}_{}_ppo_{}'.format(typer, reduce, name)

        run_number = 0
        while os.path.exists(self.log_dir + str(run_number)): run_number += 1
        self.log_dir = self.log_dir + str(run_number)
        self.train_summary_writer = tf.summary.create_file_writer(self.log_dir)
        
        self.a_loss = tf.keras.metrics.Mean( 'a_loss', dtype = tf.float32 )
        self.c_loss = tf.keras.metrics.Mean( 'c_loss', dtype = tf.float32 )
        self.reward = tf.keras.metrics.Mean( 'reward', dtype = tf.float32 )
    
    @staticmethod
    def create_rnn(name, activation, size, typer, kernel_initializer, normalize):

        if   typer == 's': fcr = simple_rnn( size, activation = activation, name = name, kernel_initializer = kernel_initializer, norm_h = normalize )
        elif typer == 'g': fcr =        gru( size, activation = activation, name = name, kernel_initializer = kernel_initializer, norm_h = normalize )
        elif typer == 'l': fcr =   LSTMCell( size, activation = activation, name = name, kernel_initializer = kernel_initializer )

        rnn = RNN( fcr, return_sequences = True, return_state = True, unroll = True )

        return rnn

    @staticmethod
    def sequence_reduce(mode, max_len):
        
        if   mode == 'sum':
            reduce = lambda loss : tf.reduce_mean( tf.reduce_sum( loss, axis = 1 ) )                
        elif mode == 'mean':
            reduce = lambda loss : tf.reduce_mean( tf.reduce_mean( loss, axis = 1 ) )
        elif mode == 'sum_p':
            p = np.array( [ ( 2 * ( i - 1 ) + 1 ) / ( 2 * max_len ) for i in range( 1, max_len + 1 ) ] )[np.newaxis,:,np.newaxis]
            reduce = lambda loss : tf.reduce_mean( tf.reduce_sum( loss * p, axis = 1 ) )
        elif mode == 'sum_half':
            p = np.array( [ int( i >= max_len // 2 ) for i in range( 0, max_len ) ] )[np.newaxis,:,np.newaxis]
            reduce = lambda loss, msk : tf.reduce_mean( tf.reduce_sum( loss * msk * tf.cast( p, tf.float32 ), axis = 1 ) )
        elif mode == 'mean_half':
            p = np.array( [ int( i >= max_len // 2 ) for i in range( 0, max_len ) ] )[np.newaxis,:,np.newaxis]
            reduce = lambda loss, msk : tf.reduce_mean( tf.reduce_mean( loss * msk * tf.cast( p, tf.float32 ), axis = 1 ) )
        elif mode == 'sum_end':
            reduce = lambda loss, terminal : tf.reduce_mean( tf.reduce_sum( loss * terminal, axis = 1 ) )

        return reduce

    def random_states(self, bs):

        shape = [ bs, self.sizes[1] ]
        if self.typer == 's': return tf.random.normal( shape )
        if self.typer == 'g': return tf.random.normal( shape )
        if self.typer == 'l': return ( tf.random.normal( shape ), tf.random.normal( shape ) )

    def zero_states(self, bs):

        shape = [ bs, self.sizes[1] ]
        if self.typer == 's': return tf.zeros( shape )
        if self.typer == 'g': return tf.zeros( shape )
        if self.typer == 'l': return ( tf.zeros( shape ), tf.zeros( shape ) )

    def probs(self, states, p_state):

        x = self.a_fc1( states )
        x, h = self.a_rnn( x, initial_state = p_state, training = True )
        x = self.a_fc2( x )
        probs = self.act( x )

        return probs, h

    def value(self, states, p_state):

        x = self.c_fc1( states )
        x, h = self.c_rnn( x, initial_state = p_state, training = True )
        x = self.c_fc2( x )
        values = self.val( x )

        return values, h

    def __call__(self, states, c_p_state, a_p_state):

        self.probs( states, a_p_state )
        self.value( states, c_p_state )

    def ppo_loss(self, advantages, prediction_picks, actions, y_pred, masks):
        
        LOSS_CLIPPING = 0.2
        ENTROPY_LOSS = 0.001
        
        prob = actions * y_pred
        old_prob = actions * prediction_picks

        prob = tf.clip_by_value( prob, 1e-10, 1.0 )
        old_prob = tf.cast( tf.clip_by_value( old_prob, 1e-10, 1.0 ), tf.float32 )

        ratio = tf.math.exp( tf.math.log( prob ) - tf.math.log( old_prob ) )
        
        p1 = ratio * advantages
        p2 = tf.clip_by_value( ratio, 1 - LOSS_CLIPPING, 1 + LOSS_CLIPPING ) * advantages

        actor_loss = -self.reduce( tf.minimum( p1, p2 ), masks )

        entropy = ( y_pred * tf.math.log( y_pred + 1e-10 ) )
        entropy = ENTROPY_LOSS * tf.reduce_mean( entropy )
        
        total_loss = actor_loss - entropy

        return total_loss

    def critic_PPO2_loss(self, values, y_true, y_pred, masks):
        
        LOSS_CLIPPING = 0.2
        clipped_value_loss = values + tf.clip_by_value( y_pred - values, -LOSS_CLIPPING, LOSS_CLIPPING )
        v_loss1 = ( y_true - clipped_value_loss ) ** 2
        v_loss2 = ( y_true - y_pred ) ** 2
        
        value_loss = 0.5 * self.reduce( tf.maximum( v_loss1, v_loss2 ), masks )

        return value_loss

    def train(self, states, values, target, advantages, predictions, actions, rewards, masks, epochs):

        a_vars = [ v for v in self.trainable_variables if 'a_' in v.name or 'a/' in v.name ]
        c_vars = [ v for v in self.trainable_variables if 'c_' in v.name or 'c/' in v.name ]

        bs = tf.shape( states )[0]
        for e in range( epochs ):

            with tf.GradientTape() as tape_c:
                c_pred, _ = self.value( states, self.zero_states(bs) )
                c_loss = self.critic_PPO2_loss( values, target, c_pred, masks )
            c_gradients = tape_c.gradient( c_loss, c_vars )
            self.c_optimizer.apply_gradients( zip( c_gradients, c_vars ) )
            self.c_loss( c_loss )

            with tf.GradientTape() as tape_a:
                a_pred, _ = self.probs( states, self.zero_states(bs) )
                a_loss = self.ppo_loss( advantages, predictions, actions, a_pred, masks )
            a_gradients = tape_a.gradient( a_loss, a_vars )
            self.a_optimizer.apply_gradients( zip( a_gradients, a_vars ) )
            self.a_loss( a_loss )

        self.reward( tf.reduce_mean( rewards ) )


class NaluPPONetwork(tf.Module):

    def __init__(self, action_dim, name, lr, train=False):
        
        super(NaluPPONetwork, self).__init__()
        
        self.module_type = 'NaluPPO'

        self.action_dim = action_dim
        
        # actor
        self.a_fc1 = nalu_transform( 512, name = 'a', activation = tanh, kernel_initializer = tf.random_normal_initializer( stddev = 0.01 ) )
        self.a_fc2 = dense( 256, 'a', activation = relu, kernel_initializer = tf.random_normal_initializer( stddev = 0.01 ) )
        self.a_fc3 = dense( 64, 'a', activation = relu, kernel_initializer = tf.random_normal_initializer( stddev = 0.01 ) )
        self.act = dense( action_dim, 'a', activation = softmax )

        # critic
        self.c_fc1 = nalu_transform( 512, name = 'c', activation = relu, kernel_initializer = tf.random_normal_initializer( stddev = 0.01 ) )
        self.c_fc2 = dense( 256, 'c', activation = relu, kernel_initializer = tf.random_normal_initializer( stddev = 0.01 ) )
        self.c_fc3 = dense( 64, 'c', activation = relu, kernel_initializer = tf.random_normal_initializer( stddev = 0.01 ) )
        self.val = dense( 1, 'c', activation = lambda x : x )

        self.log_dir = 'logs/nalu_ppo_{}'.format(name)

        self.a_optimizer = Adam( tf.Variable( lr ) )
        self.c_optimizer = Adam( tf.Variable( lr ) )

        run_number = 0
        while os.path.exists(self.log_dir + str(run_number)):
            run_number += 1
        self.log_dir = self.log_dir + str(run_number)
        self.train_summary_writer = tf.summary.create_file_writer(self.log_dir)
        
        self.a_loss = tf.keras.metrics.Mean( 'a_loss', dtype = tf.float32 )
        self.c_loss = tf.keras.metrics.Mean( 'c_loss', dtype = tf.float32 )
        self.reward = tf.keras.metrics.Mean( 'reward', dtype = tf.float32 )

    def probs(self, states):

        x = self.a_fc1( states )
        x = self.a_fc2( x )
        x = self.a_fc3( x )
        probs = self.act( x )

        return probs

    def value(self, states):

        x = self.c_fc1( states )
        x = self.c_fc2( x )
        x = self.c_fc3( x )
        values = self.val( x )

        return values

    def __call__(self, states):

        self.probs( states )
        self.value( states )

    def ppo_loss(self, advantages, prediction_picks, actions, y_pred):
        
        # Defined in https://arxiv.org/abs/1707.06347
        
        LOSS_CLIPPING = 0.2
        ENTROPY_LOSS = 0.001
        
        prob = actions * y_pred
        old_prob = actions * prediction_picks

        prob = tf.clip_by_value( prob, 1e-10, 1.0 )
        old_prob = tf.cast( tf.clip_by_value( old_prob, 1e-10, 1.0 ), tf.float32 )

        ratio = tf.math.exp( tf.math.log( prob ) - tf.math.log( old_prob ) )
        
        p1 = ratio * advantages
        p2 = tf.clip_by_value( ratio, 1 - LOSS_CLIPPING, 1 + LOSS_CLIPPING ) * advantages

        actor_loss = -tf.reduce_mean( tf.minimum( p1, p2 ) )

        entropy = -( y_pred * tf.math.log( y_pred + 1e-10 ) )
        entropy = ENTROPY_LOSS * tf.reduce_mean( entropy )
        
        total_loss = actor_loss - entropy

        return total_loss

    def critic_PPO2_loss(self, values, y_true, y_pred):
        
        LOSS_CLIPPING = 0.2
        clipped_value_loss = values + tf.clip_by_value( y_pred - values, -LOSS_CLIPPING, LOSS_CLIPPING )
        v_loss1 = ( y_true - clipped_value_loss ) ** 2
        v_loss2 = ( y_true - y_pred ) ** 2
        
        value_loss = 0.5 * tf.reduce_mean( tf.maximum( v_loss1, v_loss2 ) )
        #value_loss = K.mean((y_true - y_pred) ** 2) # standard PPO loss
        return value_loss

    def train(self, states, values, target, advantages, predictions, actions, rewards, epochs):

        a_vars = [ v for v in self.trainable_variables if 'a/' in v.name or 'a_' in v.name ]
        c_vars = [ v for v in self.trainable_variables if 'c/' in v.name or 'c_' in v.name ]

        for e in range( epochs ):

            with tf.GradientTape() as tape_c:
                c_pred = self.value( states )
                c_loss = self.critic_PPO2_loss( values, target, c_pred )
            c_gradients = tape_c.gradient( c_loss, c_vars )
            self.c_optimizer.apply_gradients( zip( c_gradients, c_vars ) )
            self.c_loss( c_loss )

            with tf.GradientTape() as tape_a:
                a_pred = self.probs( states )
                a_loss = self.ppo_loss( advantages, predictions, actions, a_pred )
            a_gradients = tape_a.gradient( a_loss, a_vars )
            self.a_optimizer.apply_gradients( zip( a_gradients, a_vars ) )
            self.a_loss( a_loss )

        self.reward( tf.reduce_mean( rewards ) )


class NaluAdavancedPPONetwork(tf.Module):

    def __init__(self, action_dim, name, lr, train=False):
        
        super(NaluAdavancedPPONetwork, self).__init__()
        
        self.module_type = 'NaluAPPO'

        self.action_dim = action_dim

        # base
        self.b_fc1 = nalu_transform( 512, name = 'b', activation = tanh, kernel_initializer = tf.random_normal_initializer( stddev = 0.01 ) )
        self.b_fc2 = dense( 256, name = 'b', activation = relu, kernel_initializer = tf.random_normal_initializer( stddev = 0.01 ) )
        
        # actor
        self.a_fc1 = dense( 64, 'a', activation = relu, kernel_initializer = tf.random_normal_initializer( stddev = 0.01 ) )
        self.act = dense( action_dim, 'a', activation = softmax )

        # critic
        self.c_fc1 = dense( 64, 'c', activation = relu, kernel_initializer = tf.random_normal_initializer( stddev = 0.01 ) )
        self.val = dense( 1, 'c', activation = lambda x : x )

        self.log_dir = 'logs/nalu_a_ppo_{}'.format(name)

        self.lr = tf.keras.optimizers.schedules.ExponentialDecay( lr, 1000, 0.96, staircase=False, name=None )
        # self.optimizer = Adam( self.lr )#.get_mixed_precision()

        self.b_optimizer = Adam( self.lr )
        self.a_optimizer = Adam( self.lr )
        self.c_optimizer = Adam( self.lr )

        run_number = 0
        while os.path.exists(self.log_dir + str(run_number)):
            run_number += 1
        self.log_dir = self.log_dir + str(run_number)
        self.train_summary_writer = tf.summary.create_file_writer(self.log_dir)
        
        self.a_loss = tf.keras.metrics.Mean( 'a_loss', dtype = tf.float32 )
        self.c_loss = tf.keras.metrics.Mean( 'c_loss', dtype = tf.float32 )
        self.reward = tf.keras.metrics.Mean( 'reward', dtype = tf.float32 )

    def base(self, states):

        x = self.b_fc1( states )
        x = self.b_fc2( x )

        return x

    def probs(self, states):
        
        x = self.base( states )
        x = self.a_fc1( x )
        probs = self.act( x )

        return probs

    def value(self, states):

        x = self.base( states )
        x = self.c_fc1( x )
        values = self.val( x )

        return values

    def __call__(self, states):

        x = self.b_fc1( states )
        x = self.b_fc2( x )

        xa = self.a_fc1( x )
        probs = self.act( xa )

        xc = self.c_fc1( x )
        values = self.val( xc )

        return probs, values

    def ppo_loss(self, advantages, prediction_picks, actions, y_pred):
        
        # Defined in https://arxiv.org/abs/1707.06347
        
        LOSS_CLIPPING = 0.2
        ENTROPY_LOSS = 0.001
        
        prob = actions * y_pred
        old_prob = actions * prediction_picks

        prob = tf.clip_by_value( prob, 1e-10, 1.0 )
        old_prob = tf.cast( tf.clip_by_value( old_prob, 1e-10, 1.0 ), tf.float32 )

        ratio = tf.math.exp( tf.math.log( prob ) - tf.math.log( old_prob ) )
        
        p1 = ratio * advantages
        p2 = tf.clip_by_value( ratio, 1 - LOSS_CLIPPING, 1 + LOSS_CLIPPING ) * advantages

        actor_loss = -tf.reduce_mean( tf.minimum( p1, p2 ) )

        entropy = -( y_pred * tf.math.log( y_pred + 1e-10 ) )
        entropy = ENTROPY_LOSS * tf.reduce_mean( entropy )
        
        total_loss = actor_loss - entropy

        return total_loss

    def critic_PPO2_loss(self, values, y_true, y_pred):
        
        LOSS_CLIPPING = 0.2
        clipped_value_loss = values + tf.clip_by_value( y_pred - values, -LOSS_CLIPPING, LOSS_CLIPPING )
        v_loss1 = ( y_true - clipped_value_loss ) ** 2
        v_loss2 = ( y_true - y_pred ) ** 2
        
        value_loss = 0.5 * tf.reduce_mean( tf.maximum( v_loss1, v_loss2 ) )
        # value_loss = K.mean((y_true - y_pred) ** 2) # standard PPO loss
        return value_loss

    def train(self, states, values, target, advantages, predictions, actions, rewards, epochs):

        b_vars = [ v for v in self.trainable_variables if 'b/' in v.name or 'b_' in v.name ]
        a_vars = b_vars + [ v for v in self.trainable_variables if 'a/' in v.name or 'a_' in v.name ]
        c_vars = b_vars + [ v for v in self.trainable_variables if 'c/' in v.name or 'c_' in v.name ]

        for e in range( epochs ):
                       
            with tf.GradientTape() as tape_c, tf.GradientTape() as tape_a:
                
                a_pred, c_pred = self( states )

                c_loss = self.critic_PPO2_loss( values, target, c_pred )
                a_loss = self.ppo_loss( advantages, predictions, actions, a_pred )

            a_gradients = tape_a.gradient( a_loss, a_vars )
            c_gradients = tape_c.gradient( c_loss, c_vars )
            b_gradients = [ g1 + g2 for g1, g2 in zip( a_gradients[0:len(b_vars)], c_gradients[0:len(b_vars)] ) ]
            
            self.b_optimizer.apply_gradients( zip( b_gradients, b_vars ) )
            self.a_optimizer.apply_gradients( zip( a_gradients[len(b_vars):], a_vars[len(b_vars):] ) )
            self.c_optimizer.apply_gradients( zip( c_gradients[len(b_vars):], c_vars[len(b_vars):] ) )
            
            self.c_loss( c_loss )
            self.a_loss( a_loss )
        
        self.reward( tf.reduce_mean( rewards ) )


class NaluAdavanced2PPONetwork(tf.Module):

    def __init__(self, action_dim, name, lr, train=False):
        
        super(NaluAdavanced2PPONetwork, self).__init__()
        
        self.module_type = 'NaluA2PPO'

        self.action_dim = action_dim

        # base
        self.b_fc1 = nalu_transform( 512, name = 'b', activation = tanh, kernel_initializer = tf.random_normal_initializer( stddev = 0.01 ) )
        self.b_fc2 = dense( 256, name = 'b', activation = relu, kernel_initializer = tf.random_normal_initializer( stddev = 0.01 ) )
        
        # actor
        self.a_fc1 = dense( 64, 'a', activation = relu, kernel_initializer = tf.random_normal_initializer( stddev = 0.01 ) )
        self.act = dense( action_dim, 'a', activation = softmax )

        # critic
        self.c_fc1 = dense( 64, 'c', activation = relu, kernel_initializer = tf.random_normal_initializer( stddev = 0.01 ) )
        self.val = dense( 1, 'c', activation = lambda x : x )

        self.log_dir = 'logs/nalu_a2_ppo_{}'.format(name)

        self.lr = tf.keras.optimizers.schedules.ExponentialDecay( lr, 1000, 0.96, staircase=False, name=None )
        # self.optimizer = Adam( self.lr )#.get_mixed_precision()

        self.b_optimizer = Adam( self.lr )
        self.a_optimizer = Adam( self.lr )
        self.c_optimizer = Adam( self.lr )

        run_number = 0
        while os.path.exists(self.log_dir + str(run_number)):
            run_number += 1
        self.log_dir = self.log_dir + str(run_number)
        self.train_summary_writer = tf.summary.create_file_writer(self.log_dir)
        
        self.a_loss = tf.keras.metrics.Mean( 'a_loss', dtype = tf.float32 )
        self.c_loss = tf.keras.metrics.Mean( 'c_loss', dtype = tf.float32 )
        self.reward = tf.keras.metrics.Mean( 'reward', dtype = tf.float32 )

    def base(self, states):
        x = self.b_fc1( states )
        x = self.b_fc2( x )
        return x

    def probs(self, states):
        p, lg, _ = self( states )
        return p, lg

    def value(self, states):
        _, _, v = self( states )
        return v

    def __call__(self, states, ac=None):

        xb = self.b_fc1( states )
        xb = self.b_fc2( xb )

        xa = self.a_fc1( xb )
        probs = self.act( xa )

        if ac is None:
            xc = tf.concat( [ xb, tf.stop_gradient( xa ) ], axis = -1 )
        else:
            xc = tf.concat( [ xb, tf.stop_gradient( xa ) ], axis = -1 )

        xc = self.c_fc1( xc )
        values = self.val( xc )        
        
        return probs, xa, values

    def ppo_loss(self, advantages, prediction_picks, actions, y_pred):
        
        # Defined in https://arxiv.org/abs/1707.06347
        
        LOSS_CLIPPING = 0.2
        ENTROPY_LOSS = 0.001
        
        prob = actions * y_pred
        old_prob = actions * prediction_picks

        prob = tf.clip_by_value( prob, 1e-10, 1.0 )
        old_prob = tf.cast( tf.clip_by_value( old_prob, 1e-10, 1.0 ), tf.float32 )

        ratio = tf.math.exp( tf.math.log( prob ) - tf.math.log( old_prob ) )
        
        p1 = ratio * advantages
        p2 = tf.clip_by_value( ratio, 1 - LOSS_CLIPPING, 1 + LOSS_CLIPPING ) * advantages

        actor_loss = -tf.reduce_mean( tf.minimum( p1, p2 ) )

        entropy = -( y_pred * tf.math.log( y_pred + 1e-10 ) )
        entropy = ENTROPY_LOSS * tf.reduce_mean( entropy )
        
        total_loss = actor_loss - entropy

        return total_loss

    def critic_PPO2_loss(self, values, y_true, y_pred):
        
        LOSS_CLIPPING = 0.2
        clipped_value_loss = values + tf.clip_by_value( y_pred - values, -LOSS_CLIPPING, LOSS_CLIPPING )
        v_loss1 = ( y_true - clipped_value_loss ) ** 2
        v_loss2 = ( y_true - y_pred ) ** 2
        
        value_loss = 0.5 * tf.reduce_mean( tf.maximum( v_loss1, v_loss2 ) )
        # value_loss = K.mean((y_true - y_pred) ** 2) # standard PPO loss
        return value_loss

    def train(self, states, values, target, advantages, predictions, actions, rewards, epochs, acs):

        b_vars = [ v for v in self.trainable_variables if 'b/' in v.name or 'b_' in v.name ]
        a_vars = b_vars + [ v for v in self.trainable_variables if 'a/' in v.name or 'a_' in v.name ]
        c_vars = b_vars + [ v for v in self.trainable_variables if 'c/' in v.name or 'c_' in v.name ]

        for e in range( epochs ):
                       
            with tf.GradientTape() as tape_c, tf.GradientTape() as tape_a:
                
                a_pred, _, c_pred = self( states, acs )

                c_loss = self.critic_PPO2_loss( values, target, c_pred )
                a_loss = self.ppo_loss( advantages, predictions, actions, a_pred )

            a_gradients = tape_a.gradient( a_loss, a_vars )
            c_gradients = tape_c.gradient( c_loss, c_vars )
            b_gradients = [ g1 + g2 for g1, g2 in zip( a_gradients[0:len(b_vars)], c_gradients[0:len(b_vars)] ) ]
            
            self.b_optimizer.apply_gradients( zip( b_gradients, b_vars ) )
            self.a_optimizer.apply_gradients( zip( a_gradients[len(b_vars):], a_vars[len(b_vars):] ) )
            self.c_optimizer.apply_gradients( zip( c_gradients[len(b_vars):], c_vars[len(b_vars):] ) )
            
            self.c_loss( c_loss )
            self.a_loss( a_loss )
        
        self.reward( tf.reduce_mean( rewards ) )


class IQNNaluPPONetwork(tf.Module):

    def __init__(self, name, action_dim, quantile_dim, atoms, lr, train=False):
        
        super(IQNNaluPPONetwork, self).__init__()
        
        self.module_type = 'IQNNaluPPO'

        self.action_dim = action_dim
        self.atoms = atoms
        self.quantile_dim = quantile_dim

        # base
        self.b_fc1 = nalu_transform( 512, name = 'b', activation = tanh, kernel_initializer = tf.random_normal_initializer( stddev = 0.01 ) )
        self.b_fc2 = dense( 256, name = 'b', activation = relu, kernel_initializer = tf.random_normal_initializer( stddev = 0.01 ) )
        
        # actor
        self.a_fc1 = dense( 64, 'a', activation = relu, kernel_initializer = orthogonal() )
        self.act = dense( action_dim, 'a', activation = softmax, kernel_initializer = orthogonal() )

        # critic
        self.c_fc1 = dense( 64, 'c', activation = relu, kernel_initializer = tf.random_normal_initializer( stddev = 0.01 ) )
        self.phi = dense( quantile_dim, 'c', kernel_initializer = tf.random_normal_initializer( stddev = 0.01 ), bias = False )
        self.phi_bias = tf.cast( tf.Variable( tf.zeros( quantile_dim ) ), tf.float32 )
        self.fc = dense( 64, 'c', kernel_initializer = tf.random_normal_initializer( stddev = 0.01 ), activation = relu )
        self.val = dense( 1, 'c', activation = lambda x : x )

        self.log_dir = 'logs/iqn_nalu_ppo_{}'.format(name)

        self.lr = tf.keras.optimizers.schedules.ExponentialDecay( lr, 1000, 0.96, staircase=False, name=None )

        self.b_optimizer = Adam( self.lr )
        self.a_optimizer = Adam( self.lr )
        self.c_optimizer = Adam( self.lr )

        run_number = 0
        while os.path.exists(self.log_dir + str(run_number)): run_number += 1
        self.log_dir = self.log_dir + str(run_number)
        self.train_summary_writer = tf.summary.create_file_writer(self.log_dir)
        
        self.a_loss = tf.keras.metrics.Mean( 'a_loss', dtype = tf.float32 )
        self.c_loss = tf.keras.metrics.Mean( 'c_loss', dtype = tf.float32 )
        self.reward = tf.keras.metrics.Mean( 'reward', dtype = tf.float32 )

    def base(self, states):
        x = self.b_fc1( states )
        x = self.b_fc2( x )
        return x

    def probs(self, states):
        p, lg, _, _ = self( states )
        return p, lg

    def value(self, states):
        _, _, v, tau = self( states )
        return v, tau

    def __call__(self, states, ac=None):

        # base model
        xb = self.b_fc1( states )
        xb = self.b_fc2( xb )

        # actor model
        xa = self.a_fc1( xb )
        probs = self.act( xa )

        # critic model
        ## state + action
        if ac is None: xc = tf.concat( [ xb, tf.stop_gradient( xa ) ], axis = -1 )
        else: xc = tf.concat( [ xb, tf.stop_gradient( ac ) ], axis = -1 )

        ## compute pi
        tau = np.random.rand( self.atoms, 1 )
        pi_mtx = tf.constant( np.expand_dims( np.pi * np.arange( 0, self.quantile_dim ), axis = 0 ) )
        cos_tau = tf.cos( tf.matmul( tau, pi_mtx ) )
        phi = relu( self.phi( cos_tau ) + tf.expand_dims( self.phi_bias, axis = 0 ) )
        phi = tf.expand_dims( phi, axis = 0 )
        
        ## quantile layer
        xc = self.c_fc1( xc )
        xc = tf.reshape( xc, ( -1, xc.shape[-1] ) )
        xc = tf.expand_dims( xc, 1 )
        xc = xc * phi
        xc = self.fc( xc )

        ## values
        values = self.val( xc )
        values = tf.transpose( values, [ 0, 2, 1 ] )
        
        return probs, xa, values, tau

    def ppo_loss(self, advantages, prediction_picks, actions, y_pred):
        
        LOSS_CLIPPING = 0.2
        ENTROPY_LOSS = 0.001

        advantages = tf.stop_gradient( tf.reduce_mean( advantages, axis = -1, keepdims = True ) )
        
        prob = actions * y_pred
        old_prob = actions * tf.stop_gradient( prediction_picks )

        prob = tf.clip_by_value( prob, 1e-10, 1.0 )
        old_prob = tf.cast( tf.clip_by_value( old_prob, 1e-10, 1.0 ), tf.float32 )

        ratio = tf.math.exp( tf.math.log( prob ) - tf.math.log( old_prob ) )
        
        p1 = ratio * advantages
        p2 = tf.clip_by_value( ratio, 1 - LOSS_CLIPPING, 1 + LOSS_CLIPPING ) * advantages

        actor_loss = -tf.reduce_mean( tf.minimum( p1, p2 ) )

        entropy = -( y_pred * tf.math.log( y_pred + 1e-10 ) )
        entropy = ENTROPY_LOSS * tf.reduce_mean( entropy )
        
        total_loss = actor_loss - entropy

        return total_loss

    def critic_PPO2_loss(self, values, y_true, y_pred, tau):

        LOSS_CLIPPING = 0.2

        values_tile = tf.tile( tf.expand_dims( values, axis = 3 ), [ 1, 1, 1, self.atoms ] )
        pred_tile = tf.tile( tf.expand_dims( y_pred, axis = 3 ), [ 1, 1, 1, self.atoms ] )
        target_tile = tf.cast( tf.tile( tf.expand_dims( y_true[:,np.newaxis,:], axis = 2 ), [ 1, 1, self.atoms, 1 ] ), tf.float32 )
        
        clipped_value_loss = values_tile + tf.clip_by_value( pred_tile - values_tile, -LOSS_CLIPPING, LOSS_CLIPPING )
        v_loss1 = ( target_tile - clipped_value_loss ) ** 2
        v_loss2 = ( target_tile - pred_tile ) ** 2
        
        tau = tf.cast( tf.reshape( np.array( tau ), [ 1, 1, self.atoms ] ), tf.float32 )
        inv_tau = tf.cast( 1.0 - tau, tf.float32 )
        tau = tf.cast( tf.tile( tf.expand_dims( tau, axis = 1 ), [ 1, 1, self.atoms, 1 ] ), tf.float32 )
        inv_tau = tf.cast( tf.tile( tf.expand_dims( inv_tau, axis = 1 ), [ 1, 1, self.atoms, 1 ] ), tf.float32 )
        
        error_loss1 = tf.math.subtract( target_tile, clipped_value_loss )
        error_loss2 = tf.math.subtract( target_tile, pred_tile )

        loss1 = tf.where( tf.less( error_loss1, 0.0 ), inv_tau * v_loss1, tau * v_loss1 )
        loss2 = tf.where( tf.less( error_loss2, 0.0 ), inv_tau * v_loss2, tau * v_loss2 )
        
        loss1 = tf.reduce_mean( tf.reduce_mean( loss1, axis = -1 ), axis = -1 )
        loss2 = tf.reduce_mean( tf.reduce_mean( loss2, axis = -1 ), axis = -1 )

        value_loss = 0.5 * tf.reduce_mean( tf.maximum( loss1, loss2 ) )

        return value_loss

    def train(self, states, values, target, advantages, predictions, actions, rewards, epochs, acs):

        b_vars = [ v for v in self.trainable_variables if 'b/' in v.name or 'b_' in v.name ]
        a_vars = b_vars + [ v for v in self.trainable_variables if 'a/' in v.name or 'a_' in v.name ]
        c_vars = b_vars + [ v for v in self.trainable_variables if 'c/' in v.name or 'c_' in v.name ]

        for e in range( epochs ):
                       
            with tf.GradientTape() as tape_c, tf.GradientTape() as tape_a:
                
                a_pred, _, c_pred, taus = self( states, acs )

                c_loss = self.critic_PPO2_loss( values, target, c_pred, taus )
                a_loss = self.ppo_loss( advantages, predictions, actions, a_pred )

            a_gradients, _ = tf.clip_by_global_norm( tape_a.gradient( a_loss, a_vars ), 0.02 )
            c_gradients, _ = tf.clip_by_global_norm( tape_c.gradient( c_loss, c_vars ), 0.02 )
            b_gradients = [ g1 + g2 for g1, g2 in zip( a_gradients[0:len(b_vars)], c_gradients[0:len(b_vars)] ) ]
            
            self.b_optimizer.apply_gradients( zip( b_gradients, b_vars ) )
            self.a_optimizer.apply_gradients( zip( a_gradients[len(b_vars):], a_vars[len(b_vars):] ) )
            self.c_optimizer.apply_gradients( zip( c_gradients[len(b_vars):], c_vars[len(b_vars):] ) )
            
            self.c_loss( c_loss )
            self.a_loss( a_loss )
        
        self.reward( tf.reduce_mean( rewards ) )


#######################################################################################################################################
# PPO Discrete Section - Vision
#######################################################################################################################################

class SimpleConvPPONetwork(tf.Module):

    def __init__(self, action_dim, name, lr, train=False):
        
        super(SimpleConvPPONetwork, self).__init__()
        
        self.module_type = 'ConvPPO'

        self.action_dim = action_dim
        
        # base
        self.su = StateUnderstanding()

        # memory
        self.gm = GlobalMemory( action_dim, 512, 32, 128, 32, 4 )

        # dreamer
        self.encode_future = dense( 512, 'b_' )

        ## Next State
        self.encode_state = dense( 1024, 'b_' )
        self.dmem0 = Conv2DTranspose( 128, 5, 4, name = 'dmem0', activation = relu, padding = 'same' )
        self.dmem1 = Conv2DTranspose( 64,  3, 2, name = 'dmem1', activation = relu, padding = 'same' )
        self.dmem2 = Conv2DTranspose( 64,  3, 2, name = 'dmem2', activation = relu, padding = 'same' )
        self.dmem3 = Conv2DTranspose( 32,  3, 2, name = 'dmem3', activation = softmax, padding = 'same' )

        ## Reward 
        self.r_fc1 = dense( 64, 'rw', activation = relu, kernel_initializer = tf.random_normal_initializer( stddev = 0.01 ) )
        self.rw = dense( 1, 'rw', activation = lambda x : x )

        # actor
        self.a_fc1i = dense( 512, 'ac', activation = tanh, kernel_initializer = orthogonal() )
        self.a_fc1v = dense( 512, 'ac', activation = tanh, kernel_initializer = orthogonal() )
        self.a_fc2 = dense( 256, 'ac', activation = relu, kernel_initializer = orthogonal() )
        self.a_fc3 = dense( 64, 'ac', activation = relu, kernel_initializer = orthogonal() )
        self.act = dense( action_dim, 'ac', activation = softmax )

        # critic
        self.c_fc1i = dense( 512, 'cr', activation = tanh, kernel_initializer = tf.random_normal_initializer( stddev = 0.01 ) )
        self.c_fc1v = dense( 512, 'cr', activation = tanh, kernel_initializer = tf.random_normal_initializer( stddev = 0.01 ) )
        self.c_fc2 = dense( 256, 'cr', activation = relu, kernel_initializer = tf.random_normal_initializer( stddev = 0.01 ) )
        self.c_fc3 = dense( 64, 'cr', activation = relu, kernel_initializer = tf.random_normal_initializer( stddev = 0.01 ) )
        self.val = dense( 1, 'cr', activation = lambda x : x )

        self.log_dir = 'logs/conv_ppo_{}'.format(name)

        self.lr = tf.keras.optimizers.schedules.ExponentialDecay( lr, 1000, 0.96, staircase=False, name=None )
        self.b_optimizer = Adam( self.lr )
        self.g_optimizer = RMS( tf.Variable( 2e-5 ) )
        self.a_optimizer = Adam( self.lr )
        self.c_optimizer = Adam( self.lr )

        run_number = 0
        while os.path.exists(self.log_dir + str(run_number)):
            run_number += 1
        self.log_dir = self.log_dir + str(run_number)
        self.train_summary_writer = tf.summary.create_file_writer(self.log_dir)
        
        self.a_loss = tf.keras.metrics.Mean( 'a_loss', dtype = tf.float32 )
        self.c_loss = tf.keras.metrics.Mean( 'c_loss', dtype = tf.float32 )
        self.b_loss_kl = tf.keras.metrics.Mean( 'b_loss_kl', dtype = tf.float32 )
        self.b_loss_rec = tf.keras.metrics.Mean( 'b_loss_rec', dtype = tf.float32 )
        self.g_loss_rec_next = tf.keras.metrics.Mean( 'g_loss_rec_next', dtype = tf.float32 )
        self.g_loss_rec_next_val = tf.keras.metrics.Mean( 'g_loss_rec_next_val', dtype = tf.float32 )
        self.b_perplexity = tf.keras.metrics.Mean( 'b_perplexity', dtype = tf.float32 )
        self.reward = tf.keras.metrics.Mean( 'reward', dtype = tf.float32 )
        self.m_acc = tf.keras.metrics.Accuracy( 'm_acc', dtype = tf.float32 )

        self.b_loss_rec( 1 )

    def encode_next_state(self, state, memory, a):

        bs, ss, *_ = state.shape.as_list()
        
        #
        x = self.encode_future( tf.concat( [ state, memory, a ], axis = -1 ) )

        #
        xs = self.encode_state( x )
        xs = tf.reshape( xs, [ bs * ss, -1 ] )
        xs = tf.expand_dims( tf.expand_dims( xs, axis = -2 ), axis = -2 )
        xs = self.dmem0( xs )
        xs = self.dmem1( xs )
        xs = self.dmem2( xs )
        xs = self.dmem3( xs )
        xs = tf.reshape( xs, [ bs, ss ] + self.su.index_shape )

        #
        xr = self.r_fc1( x )
        values = self.rw( xr )
        
        return xs, values

    def probs(self, states_i, states_v, eval=True):

        vq = self.su.encode( states_i, eval )
        e = tf.cast( flatten( vq['encoding_indices'] ), tf.float32 )
        xi = self.a_fc1i( tf.stop_gradient( e ) )
        xv = self.a_fc1v( states_v )
        x = self.a_fc2( xi + xv )
        x = self.a_fc3( x )
        probs = self.act( x )

        return probs

    def value(self, states_i, states_v, eval=True):

        vq = self.su.encode( states_i, eval )
        e = tf.cast( flatten( vq['encoding_indices'] ), tf.float32 )
        xi = self.c_fc1i( tf.stop_gradient( e ) )
        xv = self.c_fc1v( states_v )
        x = self.c_fc2( xi + xv )
        x = self.c_fc3( x )
        values = self.val( x )

        return values

    def __call__(self, states_i, p_states_i, states_v, ac, rw, dx):
        
        vq, _ = self.su( states_i )
        p_vq, _ = self.su( p_states_i )
        su = tf.cast( tf.expand_dims( flatten( vq['encoding_indices'] ), axis = 1 ), tf.float32 )
        p_su = tf.cast( tf.expand_dims( flatten( p_vq['encoding_indices'] ), axis = 1 ), tf.float32 )
        gm, _, ae = self.gm( su, p_su, ac, rw, dx, 0, self.gm.reset(1) )
        self.encode_next_state( su, gm, ae )
        self.probs( states_i, states_v )
        self.value( states_i, states_v )

    def ppo_loss(self, advantages, prediction_picks, actions, y_pred):
        
        LOSS_CLIPPING = 0.2
        ENTROPY_LOSS = 0.001
        
        prob = actions * y_pred
        old_prob = actions * prediction_picks

        prob = tf.clip_by_value( prob, 1e-10, 1.0 )
        old_prob = tf.cast( tf.clip_by_value( old_prob, 1e-10, 1.0 ), tf.float32 )

        ratio = tf.math.exp( tf.math.log( prob ) - tf.math.log( old_prob ) )
        
        p1 = ratio * advantages
        p2 = tf.clip_by_value( ratio, 1 - LOSS_CLIPPING, 1 + LOSS_CLIPPING ) * advantages

        actor_loss = -tf.reduce_mean( tf.minimum( p1, p2 ) )

        entropy = -( y_pred * tf.math.log( y_pred + 1e-10 ) )
        entropy = ENTROPY_LOSS * tf.reduce_mean( entropy )
        
        total_loss = actor_loss + entropy

        return total_loss

    def critic_PPO2_loss(self, values, y_true, y_pred):
        
        LOSS_CLIPPING = 0.2
        clipped_value_loss = values + tf.clip_by_value( y_pred - values, -LOSS_CLIPPING, LOSS_CLIPPING )
        v_loss1 = ( y_true - clipped_value_loss ) ** 2
        v_loss2 = ( y_true - y_pred ) ** 2
        
        value_loss = 0.5 * tf.reduce_mean( tf.maximum( v_loss1, v_loss2 ) )
        return value_loss

    def train_rl(self, states_i, states_v, values, target, advantages, predictions, actions, rewards, epochs):

        a_vars = [ v for v in self.trainable_variables if 'ac_' in v.name ]
        c_vars = [ v for v in self.trainable_variables if 'cr_' in v.name ]

        for e in range( epochs ):
           
            with tf.GradientTape() as tape_c:
                c_pred = self.value( states_i, states_v )
                c_loss = self.critic_PPO2_loss( values, target, c_pred )
            c_gradients = tape_c.gradient( c_loss, c_vars )
            self.c_optimizer.apply_gradients( zip( c_gradients, c_vars ) )
            self.c_loss( c_loss )

            with tf.GradientTape() as tape_a:
                a_pred = self.probs( states_i, states_v )
                a_loss = self.ppo_loss( advantages, predictions, actions, a_pred )
            a_gradients = tape_a.gradient( a_loss, a_vars )
            self.a_optimizer.apply_gradients( zip( a_gradients, a_vars ) )
            self.a_loss( a_loss )

        self.reward( tf.reduce_mean( rewards ) )
        
    def train_vae(self, states_i):

        b_vars = self.su.variables

        with tf.GradientTape() as tape_b:
            vq, dec = self.su( states_i, eval = False )
            b_loss = vq['rec_loss'] + vq['loss']
        
        b_gradients = tape_b.gradient( b_loss, b_vars )
        self.b_optimizer.apply_gradients( zip( b_gradients, b_vars ) )
        
        self.b_loss_kl( vq['loss'] )
        self.b_loss_rec( vq['rec_loss'] )
        self.b_perplexity( vq['perplexity'] )
        
        return dec, states_i, tf.gather( self.su.colors, vq['encoding_indices'] )

    def train_memory(self, states_i, p_states_i, n_states_i, ac, p_ac, rw, prw, dx, bs):

        g_vars = self.gm.variables + self.encode_future.trainable_variables + self.encode_state.trainable_variables + self.dmem0.trainable_variables + self.dmem1.trainable_variables + self.dmem2.trainable_variables + self.dmem3.trainable_variables + self.r_fc1.trainable_variables + self.rw.trainable_variables
        ce = tf.keras.losses.SparseCategoricalCrossentropy( from_logits = False )
        mse = tf.keras.losses.MeanSquaredError()

        bs, ss, *_ = states_i.shape.as_list()
        
        t_bound = 2
        p = np.array( [ int( i >= t_bound ) for i in range( 0, ss ) ] )[np.newaxis,:,np.newaxis,np.newaxis,np.newaxis]
        with tf.GradientTape() as tape_g:
            
            f_states = tf.reshape( states_i, [ bs * ss, 128, 128, 3 ] )
            f_p_states = tf.reshape( p_states_i, [ bs * ss, 128, 128, 3 ] )
            f_n_states = tf.reshape( n_states_i, [ bs * ss, 128, 128, 3 ] )

            pa = tf.argmax( p_ac, axis = -1 )
            a = tf.argmax( ac, axis = -1 )

            vq = self.su.encode( f_states, eval = False )
            p_vq = self.su.encode( f_p_states, eval = False )
            n_vq = self.su.encode( f_n_states, eval = False )

            su = tf.stop_gradient( tf.cast( tf.reshape( flatten( vq['encoding_indices'] ), [ bs, ss, -1 ] ), tf.float32 ) )
            p_su = tf.stop_gradient( tf.cast( tf.reshape( flatten( p_vq['encoding_indices'] ), [ bs, ss, -1 ] ), tf.float32 ) )
            n_su = tf.stop_gradient( tf.reshape( n_vq['encoding_indices'], [ bs, ss, 32, 32 ] ) )
            
            gm , _, _ = self.gm( su, p_su, pa, prw, dx, 0, self.gm.reset(bs) )
            dec_next_index, pred_values = self.encode_next_state( su, gm, self.gm.get_ac_emb( a ) )
            loss1 = ce( n_su, dec_next_index, p )
            loss2 = mse( tf.expand_dims( rw, -1 ), pred_values, p )

            loss = loss1 + loss2

            di = tf.argmax( dec_next_index, -1 )

        g_gradients = tape_g.gradient( loss, g_vars )
        self.g_optimizer.apply_gradients( zip( g_gradients, g_vars ) )
        
        self.g_loss_rec_next( loss1 )
        self.g_loss_rec_next_val( loss2 )
        self.m_acc.update_state( n_su, di )

        qp = self.su.e_vq.get_vq( di )        
        qp = tf.reshape( qp, [ bs * ss, 32, 32, 32 ] )

        next_state_p = self.su.decode( qp, eval = False )
        next_state_t = self.su.decode( n_vq['quantize'], eval = False )

        images = tf.squeeze( tf.concat( tf.split( tf.reshape( states_i, [ bs, ss, 128, 128, 3 ] ), num_or_size_splits = ss, axis = 1 ), axis = 3 ), axis = 1 )
        n_images = tf.squeeze( tf.concat( tf.split( tf.reshape( n_states_i, [ bs, ss, 128, 128, 3 ] ), num_or_size_splits = ss, axis = 1 ), axis = 3 ), axis = 1 )
        
        p_index = tf.squeeze( tf.concat( tf.split( tf.reshape( di, [ bs, ss, 32, 32 ] ), num_or_size_splits = ss, axis = 1 )[10:], axis = 3 ), axis = 1 )
        t_index = tf.squeeze( tf.concat( tf.split( tf.reshape( n_su, [ bs, ss, 32, 32 ] ), num_or_size_splits = ss, axis = 1 )[10:], axis = 3 ), axis = 1 )
        

        next_state_p = tf.squeeze( tf.concat( tf.split( tf.reshape( next_state_p, [ bs, ss, 128, 128, 3 ] ), num_or_size_splits = ss, axis = 1 ), axis = 3 ), axis = 1 )
        next_state_t = tf.squeeze( tf.concat( tf.split( tf.reshape( next_state_t, [ bs, ss, 128, 128, 3 ] ), num_or_size_splits = ss, axis = 1 ), axis = 3 ), axis = 1 )
        
        return ( images, n_images, tf.gather( self.su.colors, p_index ), tf.gather( self.su.colors, t_index ), next_state_p, next_state_t )


#######################################################################################################################################
# PPO Continuous Section
#######################################################################################################################################

class SimpleCPPONetwork(tf.Module):

    def __init__(self, action_dim, name, lr, action_bound, std_bound, train=False):
        
        super(SimpleCPPONetwork, self).__init__()
        
        self.module_type = 'CPPO'

        self.action_dim = action_dim
        self.action_bound = action_bound
        self.std_bound = std_bound
        
        # actor
        self.a_fc1 = dense( 512, 'a', activation = relu, kernel_initializer = orthogonal() )
        self.a_fc2 = dense( 256, 'a', activation = relu, kernel_initializer = orthogonal() )
        self.a_fc3 = dense( 64, 'a', activation = relu, kernel_initializer = orthogonal() )
        self.mu = dense( action_dim, 'a', activation = softmax )
        self.std = dense( action_dim, 'a', activation = softplus )

        # critic
        self.c_fc1 = dense( 512, 'c', activation = relu, kernel_initializer = tf.random_normal_initializer( stddev = 0.01 ) )
        self.c_fc2 = dense( 256, 'c', activation = relu, kernel_initializer = tf.random_normal_initializer( stddev = 0.01 ) )
        self.c_fc3 = dense( 64, 'c', activation = relu, kernel_initializer = tf.random_normal_initializer( stddev = 0.01 ) )
        self.val = dense( 1, 'c', activation = lambda x : x )

        self.log_dir = 'logs/cppo_{}'.format(name)

        self.lr = tf.keras.optimizers.schedules.ExponentialDecay( lr, 1000, 0.96, staircase=False, name=None )
        self.a_optimizer = Adam( self.lr )
        self.c_optimizer = Adam( self.lr )

        run_number = 0
        while os.path.exists(self.log_dir + str(run_number)):
            run_number += 1
        self.log_dir = self.log_dir + str(run_number)
        self.train_summary_writer = tf.summary.create_file_writer(self.log_dir)
        
        self.a_loss = tf.keras.metrics.Mean( 'a_loss', dtype = tf.float32 )
        self.c_loss = tf.keras.metrics.Mean( 'c_loss', dtype = tf.float32 )
        self.reward = tf.keras.metrics.Mean( 'reward', dtype = tf.float32 )

    def probs(self, states):

        x = self.a_fc1( states )
        x = self.a_fc2( x )
        x = self.a_fc3( x )

        mu = self.mu( x ) * self.action_bound
        std = self.std( x )

        return mu, std

    def log_pdf(self, mu, std, action):

        std = tf.clip_by_value( std, self.std_bound[0], self.std_bound[1] )
        var = std ** 2
        log_policy_pdf = -0.5 * (action - mu) ** 2 / var - 0.5 * tf.math.log(var * 2 * np.pi)

        return tf.reduce_sum( log_policy_pdf, 1, keepdims = True )

    def get_action(self, states):

        mu, std = self.probs( states )
        action = tf.random.normal( mean = mu, stddev = std, shape = [ tf.shape( states )[0], self.action_dim ] )
        action = tf.clip_by_value( action, -self.action_bound, self.action_bound )
        log_policy = self.log_pdf( mu, std, action )

        return log_policy, action

    def value(self, states):

        x = self.c_fc1( states )
        x = self.c_fc2( x )
        x = self.c_fc3( x )
        values = self.val( x )

        return values

    def __call__(self, states):

        self.probs( states )
        self.value( states )

    def ppo_loss(self, advantages, prediction_picks, actions, y_pred):
        
        # Defined in https://arxiv.org/abs/1707.06347
        
        LOSS_CLIPPING = 0.2
        ENTROPY_LOSS = 0.001
        
        prob = actions * y_pred
        old_prob = actions * prediction_picks

        prob = tf.clip_by_value( prob, 1e-10, 1.0 )
        old_prob = tf.cast( tf.clip_by_value( old_prob, 1e-10, 1.0 ), tf.float32 )

        ratio = tf.math.exp( tf.math.log( prob ) - tf.math.log( old_prob ) )
        
        p1 = ratio * advantages
        p2 = tf.clip_by_value( ratio, 1 - LOSS_CLIPPING, 1 + LOSS_CLIPPING ) * advantages

        actor_loss = -tf.reduce_mean( tf.minimum( p1, p2 ) )

        entropy = -( y_pred * tf.math.log( y_pred + 1e-10 ) )
        entropy = ENTROPY_LOSS * tf.reduce_mean( entropy )
        
        total_loss = actor_loss - entropy

        return total_loss

    def critic_PPO2_loss(self, values, y_true, y_pred):
        
        LOSS_CLIPPING = 0.2
        clipped_value_loss = values + tf.clip_by_value( y_pred - values, -LOSS_CLIPPING, LOSS_CLIPPING )
        v_loss1 = ( y_true - clipped_value_loss ) ** 2
        v_loss2 = ( y_true - y_pred ) ** 2
        
        value_loss = 0.5 * tf.reduce_mean( tf.maximum( v_loss1, v_loss2 ) )
        #value_loss = K.mean((y_true - y_pred) ** 2) # standard PPO loss
        return value_loss

    def train(self, states, values, target, advantages, predictions, actions, rewards, epochs):

        a_vars = [ v for v in self.trainable_variables if 'a_' in v.name ]
        c_vars = [ v for v in self.trainable_variables if 'c_' in v.name ]

        for e in range( epochs ):

            with tf.GradientTape() as tape_c:
                c_pred = self.value( states )
                c_loss = self.critic_PPO2_loss( values, target, c_pred )
            c_gradients = tape_c.gradient( c_loss, c_vars )
            self.c_optimizer.apply_gradients( zip( c_gradients, c_vars ) )
            self.c_loss( c_loss )

            with tf.GradientTape() as tape_a:
                a_pred = self.probs( states )
                a_loss = self.ppo_loss( advantages, predictions, actions, a_pred )
            a_gradients = tape_a.gradient( a_loss, a_vars )
            self.a_optimizer.apply_gradients( zip( a_gradients, a_vars ) )
            self.a_loss( a_loss )

        self.reward( tf.reduce_mean( rewards ) )


#######################################################################################################################################
# Old Section
#######################################################################################################################################

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
        
        self.su = StateUnderstanding( state_dim, f1, f1, train )
        self.ac = Actorv2( f2, action_dim, f2, atoms, train )

        self.log_dir = 'logs/nqrnet_{}'.format(name)

        if train:

            self.optimizer = Adam( tf.Variable( 2e-4 ), amsgrad = False )# .get_mixed_precision()
            
            if self.log:
                run_number = 0
                while os.path.exists(self.log_dir + str(run_number)):
                    run_number += 1
                self.log_dir = self.log_dir + str(run_number)
                self.train_summary_writer = tf.summary.create_file_writer(self.log_dir)
        
        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.reward = tf.keras.metrics.Mean('reward', dtype=tf.float32)
    
    def __call__(self, states, eval=True):
        
        """Build a network that maps state -> action values."""
        
        # State Understanding
        s = self.su( states, eval = eval )
        ac, tau = self.ac( s, eval = eval )

        if eval:
            return ac, tau
        else:
            return ac, tau, s

    def train(self, states, target, actions, w):

        with tf.GradientTape() as tape:
            
            theta, tau, s = self( states, eval = False )
            
            huber_loss = self.ac.quantile_huber_loss( target, theta, actions, tau )
            # loss = tf.reduce_mean( huber_loss * w )
            loss = tf.reduce_mean( huber_loss )
                    
        gradients = tape.gradient( loss, self.trainable_variables )
        
        self.optimizer.apply_gradients( zip( gradients, self.trainable_variables ) )

        self.train_loss( loss )

        return huber_loss, tf.reduce_mean( theta, axis = -1 ), loss


class NALUQRMultiNetwork(tf.Module):

    def __init__(self, envs, name, f1, f2, atoms, maxlen, train=False):
        
        super(NALUQRMultiNetwork, self).__init__()
        
        self.module_type = 'NQRNet'

        self.f1 = f1
        self.f2 = f2
        self.atoms = atoms
        
        self.su = StateUnderstandingMultiEnv_v3( envs, f1, f1, maxlen )
        # self.ac = ActorMultiEnv_v2( envs, f2, atoms )
        self.ac = { x['id']: FakeActor( f2, x['action_dim'], atoms ) for x in envs }

        self.wa = tf.Variable( tf.random.normal( [ 4, f1 ] ), name = "self_w_a", trainable = True )
        self.self_learning = dense( 8 )
        self.self_learning( tf.zeros( [ 1, 1, f1 ] ) )

        self.log_dir = 'logs/nqrnet_{}'.format(name)

        if train:

            self.optimizer = Adam( tf.Variable( 2e-4 ) ).get_mixed_precision()
            
            self.reduce = lambda loss : tf.reduce_mean( tf.reduce_mean( loss, axis = 1 ) )
                
            run_number = 0
            
            # while os.path.exists(self.log_dir + str(run_number)): run_number += 1
            self.log_dir = self.log_dir + str(run_number)

            self.train_summary_writer_total = tf.summary.create_file_writer( '{}'.format( self.log_dir ) )

            self.train_summary_writer = {}
            self.train_loss = {}
            self.train_l2_loss = {}
            self.reward = {}
            for e in envs:
                self.train_summary_writer[e['id']] = tf.summary.create_file_writer( '{}_{}'.format( self.log_dir, e['name'] + '_' + e['id'] ) )
                self.train_loss[e['id']] = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
                self.reward[e['id']] = tf.keras.metrics.Mean('reward', dtype=tf.float32)
    
    def __call__(self, states, env, past=None, eval=True):
        
        """Build a network that maps state -> action values."""
        
        # State Understanding
        s, past, mask, vls = self.su( states, None, env, past, eval = eval )
        ac, _ = self.ac[env]( s, env )

        if eval:
            return ac, past, mask
        else:
            return ac, past, mask, vls

    def train(self, states, target, actions, next_states, t_masks, envs):
        
        ls = {}
        ls_self = {}
        masks = {}
        thetas = {}
        loses = 0
        su_values = [
            tf.zeros( [ 8 ] ),
            tf.zeros( [ self.f1 ] ),
            tf.zeros( [ self.f1 ] ),
            tf.zeros( [ self.f1 ] ),
            tf.zeros( [ self.f1 ] ),
            tf.zeros( [ self.f1 ] ),
            tf.zeros( [ self.f1 ] ),
            tf.zeros( [ self.f1 ] ),
        ]

        with tf.GradientTape() as tape:
            
            for e in envs:
                            
                # compute loss for each env
                theta, _, mask, vls = self( states[e], e, eval = False )
                huber_loss = self.ac[e].quantile_huber_loss( target[e], theta, actions[e] ) * t_masks[e]
                loss1 = self.reduce( huber_loss )

                # compute next state
                acs = tf.gather( self.wa, tf.cast( actions[e], tf.int32 ) )
                next_state_predict = self.self_learning( vls[3] + acs )
                loss2 = self.reduce( tf.keras.losses.huber( next_states[e], next_state_predict ) )

                loss = loss1 + loss2
                
                self.train_loss[e]( loss )
                ls[e] = loss1
                ls_self[e] = loss2
                masks[e] = mask
                thetas[e] = theta
                
                loses += loss
                
                su_values = [ x1 + tf.reduce_mean( tf.reduce_mean( x2, axis = 0 ), axis = 0 ) for x1, x2 in zip( su_values, vls ) ]

            # compute l2 loss
            l2_loss = \
                tf.reduce_sum( [ tf.nn.l2_loss( w ) for w in self.su.weights ] )
            
            loses += ( 2e-6 * l2_loss )

            scaled_loss = self.optimizer.get_scaled_loss( loses )
                
        grads = self.optimizer.get_unscaled_gradients( tape.gradient( scaled_loss, self.trainable_variables ) )
        self.optimizer.apply_gradients( zip( grads, self.trainable_variables ) )

        return ls, ls_self, masks, thetas, [ x1 / len( envs ) for x1 in su_values ]

    def train2(self, states, target, actions, t_masks, envs):
        
        ls = {}
        masks = {}
        thetas = {}
        grads = [ tf.zeros_like( x ) for x in self.trainable_variables ]
        su_values = [
            tf.zeros( [ 8 ] ),
            tf.zeros( [ self.f1 ] ),
            tf.zeros( [ self.f1 ] ),
            tf.zeros( [ self.f1 ] ),
            tf.zeros( [ self.f1 ] ),
            tf.zeros( [ self.f1 ] ),
            tf.zeros( [ self.f1 ] ),
            tf.zeros( [ self.f1 ] ),
        ]
        for e in envs:
            
            with tf.GradientTape() as tape:
                
                # compute loss for each env
                theta, _, mask, vls = self( states[e], e, eval = False )
                huber_loss = self.ac[e].quantile_huber_loss( target[e], theta, actions[e] ) * t_masks[e]
                loss = self.reduce( huber_loss )
                self.train_loss[e]( loss )
                ls[e] = loss
                masks[e] = mask
                thetas[e] = theta

                # compute l2 loss
                l2_loss = \
                    tf.reduce_sum( [ tf.nn.l2_loss( w ) for w in self.su.weights ] ) +\
                    tf.reduce_sum( [ tf.nn.l2_loss( w ) for w in self.ac[e].weights ] )
                
                tloss = loss + ( 1e-5 * l2_loss )
                scaled_loss = self.optimizer.get_scaled_loss( tloss )

                su_values = [ x1 + tf.reduce_mean( tf.reduce_mean( x2, axis = 0 ), axis = 0 ) for x1, x2 in zip( su_values, vls ) ]
                
            grad = self.optimizer.get_unscaled_gradients( tape.gradient( scaled_loss, self.trainable_variables ) )
            for i, ( g, v ) in enumerate( zip( grad, self.trainable_variables ) ):                
                if g is None: continue                
                if 'su_recurrent' in v.name:        g = g * self.su.envs[e]['recurrent']
                elif 'su_non_recurrent' in v.name:  g = g * ( 1 - self.su.envs[e]['recurrent'] )
                grads[i] += g

        self.optimizer.apply_gradients( zip( grads, self.trainable_variables ) )

        return ls, masks, thetas, [ x1 / len( envs ) for x1 in su_values ]

  
class MNALUQRNetwork(tf.Module):

    def __init__(self, name,
        state_dim, sf1, sf2, # state understanding
        m_hsize, m, n, max_size, num_blocks, n_read, n_att_heads, lr, decay, # global memory
        fc1, fc2, # feature creator
        action_dim, df1, atoms, # actor and critic
        train=False):
        
        super(MNALUQRNetwork, self).__init__()
        
        self.module_type = 'MNQRNet'

        self.to_train = train
        self.atoms = atoms
        self.action_dim = action_dim

        self.su = StateUnderstanding( state_dim, sf1, sf2, train )
        self.gm = GlobalMemory( action_dim, m_hsize, m, n, 64, max_size, num_blocks, n_read, n_att_heads, lr, decay, train )
        self.fc = FeatureCreator( fc1, fc2, train )
        self.ac = Actor( df1, action_dim, atoms, train )
        self.cr = Critic( atoms, train )

        self.log_dir = 'logs/mnqrnet_{}'.format(name)

        self.optimizer = Adam( tf.Variable( 2e-4 ) )

        self.a_train_loss = tf.keras.metrics.Mean('a_train_loss', dtype=tf.float32)
        self.c_train_loss = tf.keras.metrics.Mean('c_train_loss', dtype=tf.float32)

        self.train_l2_loss = tf.keras.metrics.Mean('train_l2_loss', dtype=tf.float32)

        self.reward = tf.keras.metrics.Mean('reward', dtype=tf.float32)
        self.actions = [ tf.keras.metrics.Mean('act_{}'.format(x), dtype=tf.float32) for x in range(action_dim) ]

        if train:

            # p = [ int( i >= max_size // 2 ) for i in range( 0, max_size ) ]
            # self.reduce = lambda loss : tf.reduce_mean( tf.reduce_sum( loss * p, axis = 1 ) )
            self.reduce = lambda loss : tf.reduce_mean( tf.reduce_sum( loss, axis = 1 ) )
            
            run_number = 0
            while os.path.exists(self.log_dir + str(run_number)):
                run_number += 1
            self.log_dir = self.log_dir + str(run_number)
            self.train_summary_writer = tf.summary.create_file_writer(self.log_dir)
    
    def reset(self, bs):
        return self.gm.reset( bs )

    def __call__(self, state, _state, action, reward, done, x_w, gparams, step, gpast=None):
        
        """Build a network that maps state -> action values."""

        # State Understanding
        s, ps = self.su( state, _state )
        
        # Global Memory        
        y, p_state, mask = self.gm( tf.stop_gradient( s ), tf.stop_gradient( ps ), action, reward, done, x_w, step, gparams, gpast )
        gpresent, *gparams = p_state

        # Feature Creator
        h = self.fc( s, tf.stop_gradient( y ) )
        
        # Actor
        ac, h_ac = self.ac( h )

        # Critic
        c = self.cr( tf.stop_gradient( s ), tf.stop_gradient( h_ac ), y )

        return ac, c, gpresent, gparams, mask

    def train(self, states, _states, actions, x_ws, steps, gparams, ac_target, cr_target, at, rt, dt):
        
        # bs, s, *_ = states.shape.as_list()
        # sts = tf.stack( [ tf.range( start = steps[b,0], limit = steps[b,0] + s, dtype = tf.float32 ) for b in range( bs ) ], axis = 0 )
        with tf.GradientTape() as tape:
            
            a_theta, c_theta, _, _, msk = self( states, _states, at, rt, dt, x_ws, gparams, steps )
            
            # a ideia é aprender conforme o lr da memoria diminua
            cr_huber_loss = self.cr.quantile_huber_loss( cr_target, c_theta ) # * tf.stop_gradient( ( 1 - self.gm.memory.memory.alpha( sts ) ) )
            
            # aprende apos o critico saber extrair a recompensa da memoria
            #c_rate = tf.reduce_mean( tf.square( cr_target - c_theta ), axis = -1 ) 
            ac_huber_loss = self.ac.quantile_huber_loss( ac_target, a_theta, actions ) # * tf.stop_gradient( ( 1 - tf.clip_by_value( c_rate, 0, 1 ) ) )

            al = self.reduce( ac_huber_loss )
            cl = self.reduce( cr_huber_loss )
            loss =  al + cl 
            
            l2_loss =\
               tf.reduce_sum( [ tf.nn.l2_loss( w ) for w in self.su.trainable_variables ] ) +\
               tf.reduce_sum( [ tf.nn.l2_loss( w ) for w in self.gm.trainable_variables ] ) +\
               tf.reduce_sum( [ tf.nn.l2_loss( w ) for w in self.fc.trainable_variables ] ) +\
               tf.reduce_sum( [ tf.nn.l2_loss( w ) for w in self.ac.trainable_variables ] ) +\
               tf.reduce_sum( [ tf.nn.l2_loss( w ) for w in self.cr.trainable_variables ] ) 
            
            t_loss = loss + ( 2e-5 * l2_loss )
            
        gradients = tape.gradient( t_loss, self.trainable_variables )

        self.optimizer.apply_gradients( zip( gradients, self.trainable_variables ) )

        self.a_train_loss( al )
        self.c_train_loss( cl )

        self.train_l2_loss( 2e-5 * l2_loss )

        return loss, a_theta, c_theta, msk


class NALUQRPNetwork(tf.Module):

    def __init__(self, state_dim, action_dim, name, f1, f2, quantile_dim, atoms, entropy_beta, train=False, log=False):
        
        super(NALUQRPNetwork, self).__init__()
        
        self.module_type = 'NQRPNet'

        self.to_train = train
        self.atoms = atoms
        self.quantile_dim = quantile_dim
        self.log = log
        self.action_dim = action_dim
        self.entropy_beta = entropy_beta
        
        # base
        self.b_fc1 = nalu_transform( f1, activation = elu, kernel_initializer = truncated_normal(), name = 'b_su1' )
        self.b_fc2 = nalu_transform( f2, activation = elu, kernel_initializer = truncated_normal(), name = 'b_su2' )

        self.b_dp1 = tf.keras.layers.Dropout( .25 )
        self.b_dp2 = tf.keras.layers.Dropout( .25 )

        # critic
        self.c_embeding = tf.Variable( tf.random.normal( [ action_dim, f2 ] ), trainable = True, name = 'c_embeding', dtype = tf.float32 )
        self.c_fc1 = dense( quantile_dim, activation = elu, kernel_initializer = random_normal(), name = 'c' )
        self.c_fc2 = dense( quantile_dim, activation = elu, kernel_initializer = random_normal(), name = 'c' )

        self.c_phi = dense( quantile_dim, kernel_initializer = random_normal(), bias = False, name = 'c' )
        self.c_phi_bias = tf.cast( tf.Variable( tf.zeros( quantile_dim ) ), tf.float32 )
        self.c_fc = dense( f2, kernel_initializer = random_normal(), activation = elu, name = 'c' )
        self.c_fc_q = dense( 1, kernel_initializer = random_normal(), name = 'c' )
        
        self.c_dp1 = tf.keras.layers.Dropout( .25 )
        self.c_dp2 = tf.keras.layers.Dropout( .25 )

        # actor
        self.a_fc1 = dense( f2, activation = elu, kernel_initializer = orthogonal(), name = 'a' )
        self.a_fc2 = dense( f2, activation = elu, kernel_initializer = orthogonal(), name = 'a' )
        self.a_act = dense( action_dim, activation = lambda x:x, kernel_initializer = orthogonal(), name = 'a' )

        self.a_dp1 = tf.keras.layers.Dropout( .25 )
        self.a_dp2 = tf.keras.layers.Dropout( .25 )

        self.log_dir = 'logs/nqrpnet_{}'.format(name)

        if train:

            self.optimizer_b = Adam( tf.Variable( 5e-4 ) )
            self.optimizer_c = Adam( tf.Variable( 5e-4 ) )
            self.optimizer_a = Adam( tf.Variable( 1e-3 ) )
            
            if self.log:
                run_number = 0
                while os.path.exists(self.log_dir + str(run_number)):
                    run_number += 1
                self.log_dir = self.log_dir + str(run_number)
                self.train_summary_writer = tf.summary.create_file_writer(self.log_dir)
        
        self.train_actor_loss = tf.keras.metrics.Mean('train_actor_loss', dtype=tf.float32)
        self.train_critic_loss = tf.keras.metrics.Mean('train_critic_loss', dtype=tf.float32)
        self.reward = tf.keras.metrics.Mean('reward', dtype=tf.float32)
    
    def base(self, states, eval=True):

        x1 = self.b_fc1( states )
        if not eval: x1 = self.b_dp1( x1 )
        x2 = self.b_fc2( x1 )
        if not eval: x2 = self.b_dp2( x2 )

        return x2

    def actor(self, base, eval=True):
                
        x1 = self.a_fc1( base )
        if not eval: x1 = self.b_dp1( x1 )
        x2 = self.a_fc2( x1 )
        if not eval: x2 = self.b_dp2( x2 )

        logits = self.a_act( x2 )
        probs = tf.nn.softmax( logits )
        distCat = tfp.distributions.Categorical( probs = probs )

        return logits, probs, distCat

    def critic(self, base, action, eval=True):
        
        ac_emb = tf.gather( self.c_embeding, action )
        h = base + ac_emb

        h1 = self.c_fc1( h )
        if not eval: h1 = self.c_dp1( h1 )
        v = self.c_fc2( h1 )
        if not eval: v = self.c_dp2( v )

        tau = np.random.rand( self.atoms, 1 )
        pi_mtx = tf.constant( np.expand_dims( np.pi * np.arange( 0, self.quantile_dim ), axis = 0 ) )
        cos_tau = tf.cos( tf.matmul( tau, pi_mtx ) )
        phi = elu( self.c_phi( cos_tau ) + tf.expand_dims( self.c_phi_bias, axis = 0 ) )
        phi = tf.expand_dims( phi, axis = 0 )
        x = tf.reshape( v, ( -1, v.shape[-1] ) )
        x = tf.expand_dims( x, 1 )
        x = x * phi
        x = self.c_fc( x )
        x = self.c_fc_q( x )
        q = tf.transpose( x, [ 0, 2, 1 ] )

        return q, tau

    def get_action(self, states, eval=True):
        
        base = self.base( states, eval )
        
        logits, _, distCat = self.actor( base, eval )
        selected_action = distCat.sample()
        
        return logits, selected_action

    def get_real_action(self, states, eval=True):
        base = self.base( states, eval )        
        _, probs, _ = self.actor( base, eval )
        selected_action = tf.argmax( probs, axis = -1 )
        return selected_action

    def get_critic(self, states, action, eval=True):
        
        base = self.base( states, eval )                
        critic, tau = self.critic( base, action, eval )

        return critic, tau

    def __call__(self, states, action=None, eval=True):
        
        base = self.base( states, eval )
        
        logits, probs, distCat = self.actor( base, eval )
        
        if action is None:
            action = distCat.sample()
        
        critic, tau = self.critic( base, action, eval )

        return logits, probs, distCat, critic, tau

    def actor_loss_a2c(self, logits, actions, adv, clip=0.1):

        adv = tf.squeeze( tf.reduce_mean( adv, axis = -1 ), axis = -1 )

        ce_loss = tf.keras.losses.SparseCategoricalCrossentropy( from_logits = False )
        policy_loss = ce_loss( actions, logits, sample_weight = tf.stop_gradient( adv ) )
        
        return policy_loss

        # adv = tf.squeeze( tf.reduce_mean( gaes, axis = -1 ), axis = -1 )

        # # entropy = tf.reduce_mean( tf.math.negative( distCat.entropy() ) )
        # # entropy *= 0.001

        # log_prob = distCat.log_prob( actions )
        # ac = tf.one_hot( actions, self.action_dim )
        # s_old_logits = tf.reduce_sum( ac * old_logits, axis = -1 )

        # ratio = tf.exp( log_prob - s_old_logits )

        # clipped_ratio = tf.clip_by_value( ratio, 1 - clip, 1 + clip )
        # surrogate = tf.math.negative( tf.minimum( ratio * adv, clipped_ratio * adv ) )
        
        # # surr1 = ratio * adv
        # # surr2 = tf.clip_by_value( ratio, 1.0 - clip, 1.0 + clip ) * adv
        # # surr = tf.minimum( surr1, surr2 )
        # # policy_loss = tf.math.negative( tf.reduce_mean( surr ) )

        # # loss = policy_loss + entropy
       
        # return tf.reduce_mean( surrogate )

    def actor_loss_ppo2(self, old_probs, distCat, actions, adv, clip=0.1):

        adv = tf.squeeze( tf.reduce_mean( adv, axis = -1 ), axis = -1 )

        probs = distCat.probs
        
        entropy = tf.reduce_mean( tf.math.negative( distCat.entropy() ) )
        entropy *= 0.001

        log_prob = distCat.log_prob( actions )
        ac = tf.one_hot( actions, self.action_dim )
        s_old_log_probs = tf.reduce_sum( ac * tf.nn.log_softmax( old_probs ), axis = -1 )

        ratio = tf.exp( log_prob - s_old_log_probs )
        
        surr1 = ratio * adv
        surr2 = tf.clip_by_value( ratio, 1.0 - clip, 1.0 + clip ) * adv
        surr = tf.minimum( surr1, surr2 )
        policy_loss = tf.math.negative( tf.reduce_mean( surr ) )

        loss = policy_loss + entropy
       
        return tf.reduce_mean( loss )

    def actor_loss_ppo(self, old_probs, distCat, actions, adv, clip=0.1):

        adv = tf.squeeze( tf.reduce_mean( adv, axis = -1 ), axis = -1 )

        probs = distCat.probs
        entropy = tf.reduce_mean( tf.math.negative( tf.math.multiply( probs, tf.math.log( probs ) ) ) )

        ac = tf.one_hot( actions, self.action_dim )

        s_old_logits = tf.reduce_sum( ac * softmax( old_probs ), axis = -1 )
        s_probs = tf.reduce_sum( ac * distCat.probs, axis = -1 )


        sur1 = []
        sur2 = []

        for pb, t, op in zip( s_probs, adv, s_old_logits ):
            
            t     = tf.constant( t )
            op    = tf.constant( op )
            ratio = tf.math.divide( pb, op )
            s1    = tf.math.multiply( ratio, t )
            s2    = tf.math.multiply( tf.clip_by_value( ratio, 1.0 - clip, 1.0 + clip ), t )
            sur1.append(s1)
            sur2.append(s2)

        sr1 = tf.stack( sur1 )
        sr2 = tf.stack( sur2 )

        loss = tf.math.negative( tf.reduce_mean( tf.math.minimum( sr1, sr2 ) ) + 0.001 * entropy )
               
        return loss

    def quantile_huber_loss(self, target, pred, tau):

        pred_tile = tf.tile( tf.expand_dims( pred, axis = 3 ), [ 1, 1, 1, self.atoms ] )
        target_tile = tf.cast( tf.tile( tf.expand_dims( target, axis = 2 ), [ 1, 1, self.atoms, 1 ] ), tf.float32 )
        
        huber_loss = tf.compat.v1.losses.huber_loss( target_tile, pred_tile, reduction = tf.keras.losses.Reduction.NONE )
        
        tau = tf.cast( tf.reshape( np.array( tau ), [ 1, 1, self.atoms ] ), tf.float32 )
        inv_tau = tf.cast( 1.0 - tau, tf.float32 )
        tau = tf.cast( tf.tile( tf.expand_dims( tau, axis = 1 ), [ 1, 1, self.atoms, 1 ] ), tf.float32 )
        inv_tau = tf.cast( tf.tile( tf.expand_dims( inv_tau, axis = 1 ), [ 1, 1, self.atoms, 1 ] ), tf.float32 )
        
        error_loss = tf.math.subtract( target_tile, pred_tile )
        loss = tf.where( tf.less( error_loss, 0.0 ), inv_tau * huber_loss, tau * huber_loss )
        loss = tf.reduce_sum( tf.reduce_mean( loss, axis = -1 ), axis = -1 )
        
        return tf.reduce_mean( tf.squeeze( loss, axis = -1 ) )

    def train(self, states, target, actions, adv, old_policys):
        
        b_vars = [ v for v in self.trainable_variables if 'b_' in v.name ]
        a_vars = [ v for v in self.trainable_variables if 'a_' in v.name ]
        c_vars = [ v for v in self.trainable_variables if 'c_' in v.name ]
        
        with tf.GradientTape() as b_tape, tf.GradientTape() as a_tape, tf.GradientTape() as c_tape:
        #with tf.GradientTape() as tape: 

            logits, _, distCat, critic, tau = self( states )
            critic_loss = 0.5 * self.quantile_huber_loss( target, critic, tau )
            actor_loss = self.actor_loss_ppo2( old_policys, distCat, actions, adv )
            base_loss = critic_loss + actor_loss

        # gradients = tape.gradient( base_loss, self.trainable_variables )
        # gradients, _ = tf.clip_by_global_norm( gradients, 0.5 )
        # self.optimizer_b.apply_gradients( zip( gradients, self.trainable_variables ) )
        
        gradients_b = b_tape.gradient( base_loss,   b_vars )
        gradients_c = c_tape.gradient( critic_loss, c_vars )
        gradients_a = a_tape.gradient( actor_loss,  a_vars )

        gradients_b, _ = tf.clip_by_global_norm( gradients_b, 0.5 )
        gradients_c, _ = tf.clip_by_global_norm( gradients_c, 0.5 )
        gradients_a, _ = tf.clip_by_global_norm( gradients_a, 0.5 )

        self.optimizer_b.apply_gradients( zip( gradients_b, b_vars ) )
        self.optimizer_c.apply_gradients( zip( gradients_c, c_vars ) )
        self.optimizer_a.apply_gradients( zip( gradients_a, a_vars ) )

        self.train_actor_loss( actor_loss )
        self.train_critic_loss( critic_loss )

        return tf.reduce_mean( critic, axis = -1 ), actor_loss, critic_loss, logits


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
                                
        self.fc1s = nalu_transform( f1, activation = tf.nn.elu, kernel_initializer = tf.keras.initializers.random_normal() )
        self.fc1h = nalu_transform( f1, activation = tf.nn.elu, kernel_initializer = tf.keras.initializers.random_normal() )
        self.fc2 = nalu_transform( f2, activation = tf.nn.elu, kernel_initializer = tf.keras.initializers.random_normal() )
        self.fc3 = nalu_transform( action_dim * self.atoms, kernel_initializer = tf.keras.initializers.random_normal() )

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
                                
        self.fc1s = nalu_transform( f1, activation = tf.keras.activations.gelu, kernel_initializer = tf.keras.initializers.random_normal() )
        self.fc1h = Dense( f1, activation = tf.keras.activations.gelu, kernel_initializer = tf.keras.initializers.random_normal() )
        self.fc2 = nalu_transform( f2, activation = tf.keras.activations.gelu, kernel_initializer = tf.keras.initializers.random_normal() )
        self.fc3 = Dense( action_dim * self.atoms, kernel_initializer = tf.keras.initializers.orthogonal() )
           
        self.rnn = transformer_layer( f1, 4, 1, max_len )

        self.dp1 = tf.keras.layers.Dropout( .25 )
        self.dp2 = tf.keras.layers.Dropout( .25 )

        self.stepsct = 100
        self.change = False
        self.threshold = 0.7

        self.log_dir = 'logs/tnaluqrnet_{}'.format(name)

        if train:
            
            # self.optimizer_initial = Adam( tf.Variable( 2e-4 ) )
            # self.optimizer_end = SGD( tf.Variable( 2e-4 ) )

            self.optimizer_initial = SGD( tf.Variable( 2e-4 ) )
            self.optimizer_end = Adam( tf.Variable( 2e-4 ) )
            
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
        
        # State Understanding 
        bs, ss, _ = shape_list( states )

        states = tf.stack( [ states, states ], axis = 1 )

        ## compute input embeding
        xs = self.fc1s( states )
        xs1, xs2 = tf.unstack( xs, axis = 1 )

        ## compute temporal state
        xt, hr, msks = self.rnn( xs2,  past )
        xh = self.fc1h( xt )
        
        ## current state and temporal state
        xr = tf.stack( [ xs1, xh ], axis = 1 )
        if self.log: xr = self.dp1( xr )
        
        # Feature Creator
        x2 = self.fc2( xr )
        if self.log: x2 = self.dp2( x2 )

        # Actor        
        h1, h2 = tf.unstack( x2, axis = 1 )
        h = tf.concat( [ h1, h2 ], axis = -1 )
        
        # pred = tf.nn.tanh( tf.reshape( self.fc3( h ), [ bs, ss, self.action_dim, self.atoms ] ) )
        pred = tf.reshape( self.fc3( h ), [ bs, ss, self.action_dim, self.atoms ] )

        return pred, hr, msks, ( xs1, xs2, xt, xh, h1, h2 )
    
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

    def train(self, states, target, actions, rewards, dones, step):

        with tf.GradientTape() as tape:

            theta, _, msks, _ = self( states )
            
            huber_loss = self.quantile_huber_loss( target, theta, actions ) * tf.cast( rewards != 0, tf.float32 )
            loss = self.reduce( huber_loss )

            l2_loss =\
               tf.reduce_sum( [ tf.nn.l2_loss( w ) for w in self.fc1s.weights    ] ) +\
               tf.reduce_sum( [ tf.nn.l2_loss( w ) for w in self.fc1h.weights    ] ) +\
               tf.reduce_sum( [ tf.nn.l2_loss( w ) for w in self.fc2.weights     ] ) +\
               tf.reduce_sum( [ tf.nn.l2_loss( w ) for w in self.rnn.weights[1:] ] ) +\
               tf.reduce_sum( [ tf.nn.l2_loss( w ) for w in self.fc3.weights     ] )

            t_loss = loss + ( 9e-5 * l2_loss )
        
        gradients = tape.gradient( t_loss, self.trainable_variables )

        self.train_loss( loss )
        self.train_l2_loss( 9e-5 * l2_loss )

        self.optimizer_end.apply_gradients( zip( gradients, self.trainable_variables ) )

        # if self.train_loss.result().numpy() <= self.threshold and not self.change:
        #     self.change = True
        #     self.stepsct = 100
        #     self.threshold = 0.7 * self.threshold

        # if ( self.stepsct > 0 and self.change ) or self.train_loss.result().numpy() <= self.threshold:
        #     self.optimizer_end.apply_gradients( zip( gradients, self.trainable_variables ) )
        #     self.stepsct -= 1
        # else:
        #     self.change = False
        #     self.optimizer_initial.apply_gradients( zip( gradients, self.trainable_variables ) )

        return huber_loss, tf.reduce_mean( theta, axis = -1 ), loss, msks, self.change


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

   
