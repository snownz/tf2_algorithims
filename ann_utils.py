import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np
from tensorflow.keras.mixed_precision import LossScaleOptimizer


def shape_list(x):
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def gather(values, indexs):
    one_hot = tf.one_hot( indexs, values.shape[-1], dtype = tf.float32 )
    val = tf.reduce_sum( one_hot * values, axis = -1 )
    return val

def softmax(x, axis=-1):
    ex = tf.exp(x)
    return ex / tf.reduce_sum( ex, axis = axis, keepdims = True )

def flatten(x):
    shapes = x.get_shape().as_list()
    shapes = np.multiply.reduce( shapes[1:len(shapes)] )
    x = tf.reshape( x, [ -1, shapes ] )    
    return x

def categorical_sample(policy):
    action = tf.squeeze( tf.random.categorical( policy, 1 ), axis =-1 )
    return action

def l2_loss(variables, factor):
    ls = []
    for v in variables:
        ls.append( tf.nn.l2_loss( v ) )
    return factor * tf.reduce_mean( ls )

def gelu(x):
    return 0.5 * x * ( 1 + tf.tanh( np.sqrt( 2 / np.pi ) * ( x + 0.044715 * tf.pow( x, 3 ) ) ) )

class dense(Layer):

    def __init__(self, num_outputs, initializer=tf.keras.initializers.truncated_normal(), bias=True):
        super(dense, self).__init__()
        self.num_outputs = num_outputs
        self.use_bias = bias
        self.initializer = initializer
    
    def build(self, input_shape):
        self.kernel = self.add_weight( 'kernel', shape = [ int( input_shape[-1] ), self.num_outputs], initializer = self.initializer )
        if self.use_bias:
            self.bias = self.add_weight( 'bias', shape = [ self.num_outputs ], initializer = tf.keras.initializers.Zeros() )

    def call(self, input):
        x = input @ self.kernel
        if self.use_bias:
            x += self.bias
        return x

class nalu(Layer):

    def __init__(self, num_outputs, kernel_initializer, initializer=tf.keras.initializers.Orthogonal(), bias=True):
        
        super(nalu, self).__init__()
        self.num_outputs = num_outputs
        self.use_bias = bias
        self.initializer = initializer
        self.kernel_initializer = kernel_initializer
    
    def build(self, input_shape):

        self.gt = self.add_weight( "w_gt", [ input_shape[ -1 ], self.num_outputs ], initializer = self.initializer )
        self.wt = self.add_weight( "w_wt", [ input_shape[ -1 ], self.num_outputs ], initializer = self.initializer )
        self.mt = self.add_weight( "w_mt", [ input_shape[ -1 ], self.num_outputs ], initializer = self.initializer )

        self.kernel = self.add_weight( 'kernel', shape = [ int( input_shape[-1] ), self.num_outputs], initializer = self.kernel_initializer )
        self.ga = self.add_weight( 'ga', shape = [ 1, self.num_outputs ], initializer = self.initializer )

        if self.use_bias:
            self.bias = self.add_weight( 'bias', shape = [ self.num_outputs ], initializer = tf.keras.initializers.Zeros() )

    def call(self, input):

        w = tf.multiply( tf.tanh( self.wt ), tf.sigmoid( self.mt ) )
        g = tf.sigmoid( tf.matmul( input, self.gt ) )
        a = tf.matmul( input, w )
        # m = tf.sinh( tf.matmul( tf.asinh( input ), w ) )
        m = tf.exp( tf.matmul( tf.math.log( tf.abs( input ) + 1e-10 ), w ) )
        arithimetic_x = ( g * a ) + ( ( 1 - g ) * m )

        trnaformation_x = input @ self.kernel
        if self.use_bias:
            trnaformation_x += self.bias        
        trnaformation_x = tf.nn.relu( trnaformation_x )

        return ( tf.nn.sigmoid( self.ga ) * trnaformation_x ) + ( ( 1 - tf.nn.sigmoid( self.ga ) ) * arithimetic_x )

class simple_nac(Layer):

    def __init__(self, num_outputs, kernel_initializer, initializer=tf.keras.initializers.Orthogonal()):
        
        super(simple_nac, self).__init__()
        self.num_outputs = num_outputs
        self.use_bias = bias
        self.initializer = initializer
        self.kernel_initializer = kernel_initializer
    
    def build(self, input_shape):

        self.wt = self.add_weight( "w_wt", [ input_shape[ -1 ], self.num_outputs ], initializer = self.initializer )
        self.mt = self.add_weight( "w_mt", [ input_shape[ -1 ], self.num_outputs ], initializer = self.initializer )

    def call(self, input):

        w = tf.multiply( tf.tanh( self.wt ), tf.sigmoid( self.mt ) )
        a = tf.matmul( input, w )
        arithimetic_x = a

        return arithimetic_x

class nalu_gru_cell(Layer):

    def __init__(self, num_outputs, kernel_initializer, initializer=tf.keras.initializers.Orthogonal(), bias=True):
        super(nalu_gru_cell, self).__init__()
        self.state_size = num_outputs
        self.initializer = initializer
        self.kernel_initializer = kernel_initializer
        self.use_bias = bias

        self.z_op = simple_nac( num_outputs, kernel_initializer, initializer )
        self.r_op = simple_nac( num_outputs, kernel_initializer, initializer )
        self.h_op = simple_nac( num_outputs, kernel_initializer, initializer )
    
    def build(self, input_shape):

        self.z_op.build( input_shape )
        self.r_op.build( input_shape )
        self.h_op.build( input_shape )

        if self.use_bias:
            self.biasz = self.add_weight( 'biasz', shape = [ self.state_size ], initializer = tf.keras.initializers.Zeros() )
            self.biasr = self.add_weight( 'biasr', shape = [ self.state_size ], initializer = tf.keras.initializers.Zeros() )
            self.biash = self.add_weight( 'biash', shape = [ self.state_size ], initializer = tf.keras.initializers.Zeros() )

    def call(self, input, states):
        
        states = states[0]

        z = tf.nn.sigmoid( self.z_op( tf.concat( [ states, input ], axis = -1 ) ) )
        if self.use_bias: z = z + self.biasz

        r = tf.nn.sigmoid( self.r_op( tf.concat( [ states, input ], axis = -1 ) ) )
        if self.use_bias: z = z + self.biasr

        h_ = tf.nn.tanh( self.h_op( tf.concat( [ r * states, input ], axis = -1 ) ) )
        if self.use_bias: h_ = h_ + self.biash

        h = ( 1 - z ) * states + z * h_

        return h, h

class conv2d(Layer):

    def __init__(self, kernel_size, channels, stride, padding="SAME", initializer=tf.keras.initializers.Orthogonal(), bias=True):
        super(conv2d, self).__init__()
        self.kn = kernel_size
        self.ch = channels
        self.s = stride
        self.p = padding
        self.use_bias = bias
        self.initializer = initializer
    
    def build(self, input_shape):
        # kernel shape
        k = [ self.kn, self.kn, input_shape[-1], self.ch ] if type( self.kn ) is int else [ self.kn[0], self.kn[1], input_shape[-1], self.ch ]
        self.kernel = self.add_weight( "kernel_2d", shape = k, initializer = self.initializer )
        self.strides = ( 1, self.s, self.s, 1 ) if type( self.s ) is int else ( 1, self.s[0], self.s[1], 1 )
        if self.use_bias:
            self.bias = self.add_weight( 'bias', shape = [ self.ch ], initializer = tf.keras.initializers.Zeros() )

    def call(self, input):
        x = tf.nn.conv2d( input, self.kernel, self.strides, padding = self.p )
        if self.use_bias:
            x = tf.nn.bias_add( x, self.bias, name = "_bias_add" )
        return x

class conv1d(Layer):

    def __init__(self, nf, name, initializer=tf.keras.initializers.Orthogonal()):
        super(conv1d, self).__init__()  
        self.initializer = initializer
        self.nf = nf
        self.nme = name

    def build(self, input_shape):

        self.kernel = self.add_weight( '{}_kernel_1d'.format( self.nme ), shape = [ 1, input_shape[2], self.nf ], initializer = self.initializer )
        self.bias = self.add_weight( '{}_bias'.format( self.nme ), shape = [ self.nf ], initializer = tf.keras.initializers.Zeros() )

    def call(self, input):
        *start, nx = shape_list( input )
        i = tf.reshape( input, [ -1, nx ] )
        w = tf.reshape( self.kernel, [ -1, self.nf ] )
        c = tf.matmul( i, w )
        cb = c + self.bias
        r = tf.reshape( cb, start + [ self.nf ] )
        return r

class norm(Layer):

    def __init__(self, axis=-1, epsilon=1e-5):
        super(norm, self).__init__()
        self.axis = axis
        self.epsilon = epsilon
    
    def build(self, input_shape):
        self.g = self.add_weight( 'g', shape = [ int( input_shape[-1] ) ], initializer = tf.constant_initializer(1) )
        self.b = self.add_weight( 'b', shape = [ int( input_shape[-1] ) ], initializer = tf.constant_initializer(0) )

    def call(self, input):
        x = input
        u = tf.reduce_mean( x, axis = self.axis, keepdims = True )
        s = tf.reduce_mean( tf.square( x - u ), axis = self.axis, keepdims = True )
        x = ( x - u ) * tf.math.rsqrt( s + self.epsilon )
        x = x * self.g + self.b
        return x

class Adam(tf.keras.optimizers.Adam):
    
    def __init__(self, learning_rate, beta_1 = tf.Variable(0.9), beta_2 = tf.Variable(0.999), epsilon = tf.Variable(1e-7), decay = tf.Variable(0.0)):
        super(Adam, self).__init__( learning_rate, beta_1 = beta_1, beta_2 = beta_2, epsilon = epsilon )
        self.iterations
        self.decay = decay

    def get_mixed_precision(self):
        return LossScaleOptimizer( self )

class RMS(tf.keras.optimizers.RMSprop):
    
    def __init__(self, learning_rate, rho=tf.Variable(0.9), momentum=tf.Variable(0.0), epsilon=tf.Variable(1e-07), centered=False):
        super(RMS, self).__init__( learning_rate, rho=rho, momentum=momentum, epsilon=epsilon, centered=centered )
        self.iterations
        self.decay = tf.Variable(0.0)

    def get_mixed_precision(self):
        return LossScaleOptimizer( self )