import tensorflow as tf
from tensorflow.keras.layers import Layer, LayerNormalization
import numpy as np
from tensorflow.keras.mixed_precision import LossScaleOptimizer
from functools import partial

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

def attn(features, past, qkv, merge, n_head):

    def attention_mask(nd, ns, dtype):

        """1's in the lower triangle, counting from the lower right corner.
        Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
        """
        i = tf.range(nd)[:,None]
        j = tf.range(ns)
        m = i >= j - ns + nd
        return tf.cast(m, dtype)

    def merge_states(x):
        *start, a, b = shape_list(x)
        return tf.reshape(x, start + [a*b])

    def split_states(x, n):
        *start, m = shape_list(x)
        return tf.reshape(x, start + [n, m//n])

    def split_heads(x, n_head):
        # From [batch, sequence, features] to [batch, heads, sequence, features]
        return tf.transpose(split_states(x, n_head), [0, 2, 1, 3])

    def merge_heads(x):
        # Reverse of split_heads
        return merge_states(tf.transpose(x, [0, 2, 1, 3]))

    def mask_attn_weights(w):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        _, _, nd, ns = shape_list(w)
        b = attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w*b - tf.cast(1e10, w.dtype)*(1-b)
        return w

    def multihead_attn(q, k, v):
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul( q, k, transpose_b = True )
        w = w * tf.math.rsqrt( tf.cast( v.shape[-1], w.dtype ) )
        
        w = mask_attn_weights( w )
        w = tf.nn.softmax( w )
        a = tf.matmul( w, v )

        return a, w
    
    c = qkv( features )
    q, k, v = map( partial( split_heads, n_head = n_head ), tf.split( c, 3, axis = 2 ) )

    present = tf.stack( [ k, v ], axis = 1 )
    if past is not None:
        pk, pv = tf.unstack( past, axis = 1 )
        k = tf.concat( [ pk, k ], axis = -2 )
        v = tf.concat( [ pv, v ], axis = -2 )

    a, msk = multihead_attn( q, k, v )
    ah = merge_heads( a )        
    am = merge( ah )

    return am, present, msk

def expand_tile(value, size):

    """Add a new axis of given size."""
    value = tf.convert_to_tensor( value )
    ndims = value.shape.ndims
    return tf.tile( tf.expand_dims( value, axis = 0 ), [ size ] + [ 1 ] * ndims )

def positions_for(sequences, past_length):

    batch_size = tf.shape( sequences )[0]
    nsteps = tf.shape( sequences )[1]
    return expand_tile( past_length + tf.range( nsteps ), batch_size )

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

    def __init__(self, num_outputs, kernel_initializer, activation=lambda x:x, initializer=tf.keras.initializers.truncated_normal(), bias=True):
        
        super(nalu, self).__init__()
        self.num_outputs = num_outputs
        self.use_bias = bias
        self.initializer = initializer
        self.kernel_initializer = kernel_initializer
        self.activation = activation
    
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
        trnaformation_x = self.activation( trnaformation_x )

        return ( tf.nn.sigmoid( self.ga ) * trnaformation_x ) + ( ( 1 - tf.nn.sigmoid( self.ga ) ) * arithimetic_x )

class simple_nac(Layer):

    def __init__(self, num_outputs, initializer=tf.keras.initializers.truncated_normal()):
        
        super(simple_nac, self).__init__()
        self.num_outputs = num_outputs
        self.initializer = initializer
    
    def build(self, input_shape):

        self.wt = self.add_weight( "w_wt", [ input_shape[ -1 ], self.num_outputs ], initializer = self.initializer )
        self.mt = self.add_weight( "w_mt", [ input_shape[ -1 ], self.num_outputs ], initializer = self.initializer )

    def call(self, input):

        w = tf.multiply( tf.tanh( self.wt ), tf.sigmoid( self.mt ) )
        a = tf.matmul( input, w )

        return a

class nalu_gru_cell(Layer):

    def __init__(self, num_outputs, kernel_initializer, bias=True):
        
        super(nalu_gru_cell, self).__init__()
        
        self.state_size = num_outputs
        self.kernel_initializer = kernel_initializer
        self.use_bias = bias

        self.z_oph = dense( num_outputs, kernel_initializer )
        self.z_opi = dense( num_outputs, kernel_initializer )

        self.r_oph = dense( num_outputs, kernel_initializer )
        self.r_opi = dense( num_outputs, kernel_initializer )

        self.h_oph = simple_nac( num_outputs )
        self.h_opi = simple_nac( num_outputs )
    
    def build(self, input_shape):

        if self.use_bias:
            self.biasz = self.add_weight( 'biasz', shape = [ self.state_size ], initializer = tf.keras.initializers.Zeros() )
            self.biasr = self.add_weight( 'biasr', shape = [ self.state_size ], initializer = tf.keras.initializers.Zeros() )
            self.biash = self.add_weight( 'biash', shape = [ self.state_size ], initializer = tf.keras.initializers.Zeros() )

    def call(self, input, states):
        
        states = states[0]

        z = self.z_oph( states ) + self.z_opi( input )
        if self.use_bias: z = z + self.biasz
        z = tf.nn.sigmoid( z )

        r = self.r_oph( states ) + self.r_opi( input )
        if self.use_bias: z = z + self.biasr
        r = tf.nn.sigmoid( r )

        h_ = self.h_oph( r * states ) +  self.h_opi( input )
        if self.use_bias: h_ = h_ + self.biash
        h_ = tf.nn.elu( h_ )

        h = ( ( 1 - z ) * states ) + ( z * h_ )

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

class transformer_block(Layer):

    def __init__(self, num_outputs, n_heads, initializer=tf.keras.initializers.he_normal()):
        
        super(transformer_block, self).__init__()

        self.qkv = conv1d( num_outputs * 3, 'qkv', initializer = initializer )
        self.ln1 = conv1d( num_outputs * 4, 'ln1', initializer = initializer )
        self.ln2 = conv1d( num_outputs, 'ln2', initializer = initializer )
        self.merge = conv1d( num_outputs, 'merge', initializer = initializer )

        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()
        
        self.n_heads = n_heads

        self.num_outputs = num_outputs
    
    def build(self, input_shape):
        pass

    def call(self, input, past):
        
        x = input
        a, present, msk = attn( self.norm1( x ), past, self.qkv, self.merge, self.n_heads )
        xa = x + a
        m = gelu( self.ln1( self.norm2( xa ) ) )
        m = self.ln2( m )
        xm = x + m

        return xm, present, msk

class transformer_layer(Layer):

    def __init__(self, num_outputs, n_heads, num_blocks, max_len, initializer=tf.keras.initializers.RandomNormal( stddev = 0.01 )):        
        
        super(transformer_layer, self).__init__()

        self.max_len = max_len
        self.num_outputs = num_outputs
        self.num_blocks = num_blocks
        self.initializer = initializer
        self.blocks = [ transformer_block( num_outputs, n_heads ) for _ in range( num_blocks ) ]        
    
    def build(self, input_shape):
        
        self.wpe = self.add_weight( 'positional_embeding', shape = [ self.max_len, self.num_outputs ], initializer = self.initializer )
        
    def call(self, input, past):
        
        x = input

        past_length = 0 if past is None else tf.shape(past)[-2]
        h = x + tf.gather( self.wpe, positions_for( x, past_length ) )

        presents = []
        msks = []
        pasts = tf.unstack( past, axis = 1 ) if not past is None else [ None ] * self.num_blocks
        for idx, past in enumerate( pasts ):
            h, present, msk = self.blocks[idx]( h, past )
            presents.append( present )
            msks.append( msk )
        present = tf.stack( presents, axis = 1 )
        return h, present, msks

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