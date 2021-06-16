import tensorflow as tf
from tensorflow.keras.layers import Layer, LayerNormalization, Dense, RNN, Conv1D, SimpleRNNCell, GRUCell, GlobalAveragePooling2D
from tensorflow.keras.activations import gelu, sigmoid, tanh
import numpy as np
from tensorflow.keras.mixed_precision import LossScaleOptimizer
from functools import partial
from tensorflow.keras.initializers import GlorotUniform

def shape_list(x):
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def gather(values, indexs):
    one_hot = tf.one_hot( indexs, values.shape[-1], dtype = values.dtype )
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

def gelu2(x):
    return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))


class dense(Layer):

    def __init__(self, num_outputs, name, activation = lambda x: x, kernel_initializer=tf.keras.initializers.truncated_normal(), bias=True):
        super(dense, self).__init__()
        self.lname = name
        self.num_outputs = num_outputs
        self.use_bias = bias
        self.activation = activation
        self.initializer = kernel_initializer
   
    def build(self, input_shape):
        self.kernel = self.add_weight( '{}_kernel'.format(self.lname), shape = [ self.num_outputs,  int( input_shape[-1] ) ], initializer = self.initializer )
        if self.use_bias:
            self.bias = self.add_weight( '{}_bias'.format(self.lname), shape = [ self.num_outputs ], initializer = tf.keras.initializers.Zeros() )

    def call(self, input, w_mask=None):

        x = tf.matmul( input, self.kernel, transpose_b = True )

        if self.use_bias:
            x = tf.nn.bias_add( x, self.bias )
        return self.activation( x )


class simple_rnn(SimpleRNNCell):

    def __init__(self, num_outputs, name, activation = lambda x: x, kernel_initializer=tf.keras.initializers.truncated_normal(), bias=True, norm_h=True):
        super(simple_rnn, self).__init__( num_outputs, activation, bias, kernel_initializer, name = name )
        self.norm_h = norm_h
        self.norm = norm( name )

    def build(self, input_shape):
        super().build( input_shape )
        if self.norm_h:
            self.norm.build( tf.TensorShape( [ self.state_size ] ) )
   
    def call(self, inputs, states):
        x, h = super().call( inputs, states )        
        if self.norm_h: return x, self.norm( h[0] )
        else: return x, h


class gru(GRUCell):

    def __init__(self, num_outputs, name, activation = lambda x: x, kernel_initializer=tf.keras.initializers.truncated_normal(), bias=True, norm_h=True):
        super(gru, self).__init__( num_outputs, activation = activation, use_bias = bias, kernel_initializer = kernel_initializer, name = name )
        self.norm_h = norm_h
        self.norm = norm( name )

    def build(self, input_shape):
        super().build( input_shape )
        if self.norm_h:
            self.norm.build( tf.TensorShape( [ self.state_size ] ) )
   
    def call(self, inputs, states):
        x, h = super().call( inputs, states )        
        if self.norm_h: return x, self.norm( h[0] )
        else: return x, h


class norm(Layer):

    def __init__(self, name, axis=-1, epsilon=1e-5):
        
        super(norm, self).__init__(name=name)
        self.axis = axis
        self.epsilon = epsilon

    def build(self, input_shape):

        self.g = self.add_weight( name = 'nomr_g', shape = [ int( input_shape[-1] ) ], initializer = tf.constant_initializer( 1 ), trainable = True )
        self.b = self.add_weight( name = 'nomr_b', shape = [ int( input_shape[-1] ) ], initializer = tf.constant_initializer( 0 ), trainable = True )
        
    def call(self, input):

        x = input
        u = tf.reduce_mean( x, axis = self.axis, keepdims = True )
        s = tf.reduce_mean( tf.square( x - u ), axis = self.axis, keepdims = True )
        nx = ( x - u ) * tf.math.rsqrt( s + self.epsilon )
        nx = ( nx * self.g ) + self.b
        
        return nx


class dense_w(Layer):

    def __init__(self, num_outputs, name, activation=lambda x:x, bias=True):
        
        super(dense_w, self).__init__(name=name)
        self.num_outputs = num_outputs
        self.use_bias = bias
        self.activation = activation
    
    def build(self, input_shape):

        if self.use_bias:
            self.bias = self.add_weight( 'bias', shape = [ self.num_outputs ], initializer = tf.keras.initializers.Zeros() )

    def call(self, input, i, kn, we):

        exceded = tf.abs( ( tf.shape(kn)[0] * i ) - tf.shape(we)[0] )
        t_kn = tf.tile( kn, [ i, i ] )[exceded:,exceded:] * we

        trnaformation_x = input @ t_kn
        if self.use_bias:
            trnaformation_x += self.bias        
        trnaformation_x = self.activation( trnaformation_x )

        return trnaformation_x


class nalu(Layer):

    def __init__(self, num_outputs, name, initializer=tf.keras.initializers.truncated_normal()):
        
        super(nalu, self).__init__(name=name)
        self.num_outputs = num_outputs
        self.initializer = initializer
           
    def build(self, input_shape):

        self.gt = self.add_weight( "w_gt", [ input_shape[ -1 ], self.num_outputs ], initializer = self.initializer )
        self.wt = self.add_weight( "w_wt", [ input_shape[ -1 ], self.num_outputs ], initializer = self.initializer )
        self.mt = self.add_weight( "w_mt", [ input_shape[ -1 ], self.num_outputs ], initializer = self.initializer )

    def call(self, input):

        x1 = input

        w = tf.multiply( tf.tanh( self.wt ), tf.sigmoid( self.mt ) )
        g = tf.sigmoid( tf.matmul( x1, self.gt ) )
        a = tf.matmul( x1, w )
        m = tf.sinh( tf.matmul( tf.asinh( x1 ), w ) )
        arithimetic_x = ( g * a ) + ( ( 1 - g ) * m )

        return arithimetic_x


class nalu_transform(Layer):

    def __init__(self, num_outputs, kernel_initializer, name, activation=lambda x:x, initializer=tf.keras.initializers.truncated_normal(), bias=True):
        
        super(nalu_transform, self).__init__(name=name)
        self.num_outputs = num_outputs
        self.use_bias = bias
        self.initializer = initializer
        self.kernel_initializer = kernel_initializer
        self.activation = activation
        self.embeding = conv1d( num_outputs * 3, 'embeding', self.initializer )
     
    def build(self, input_shape):

        self.gt = self.add_weight( "w_gt", [ input_shape[ -1 ], self.num_outputs ], initializer = self.initializer )
        self.wt = self.add_weight( "w_wt", [ input_shape[ -1 ], self.num_outputs ], initializer = self.initializer )
        self.mt = self.add_weight( "w_mt", [ input_shape[ -1 ], self.num_outputs ], initializer = self.initializer )

        self.kernel_1 = self.add_weight( 'kernel_1', shape = [ int( input_shape[-1] ), self.num_outputs], initializer = self.kernel_initializer )
        self.kernel_3 = self.add_weight( 'kernel_3', shape = [ self.num_outputs * 3, self.num_outputs], initializer = self.kernel_initializer )
        self.embeding_k = self.add_weight( 'embeding_k', shape = [ 1, input_shape[ -1 ], 1 ], initializer = self.initializer )
        self.norm = norm('norm')
                
        if self.use_bias:
            self.bias_1 = self.add_weight( 'bias_1', shape = [ self.num_outputs ], initializer = tf.keras.initializers.Zeros() )
            self.bias_3 = self.add_weight( 'bias_3', shape = [ self.num_outputs ], initializer = tf.keras.initializers.Zeros() )

    def nalu_cell(self, x):
        wa = tf.multiply( tf.tanh( self.wt ), tf.sigmoid( self.mt ) )
        g = tf.sigmoid( tf.matmul( x, self.gt ) )
        a = tf.matmul( x, wa )
        m = tf.sinh( tf.matmul( tf.asinh( x ), wa ) )
        value = ( g * a ) + ( ( 1 - g ) * m )
        return value

    def perceptron_cell(self, x, w, b, act):
        wt = w
        value = tf.matmul( x, wt )
        if not b is None:
            value = tf.nn.bias_add( value, b )
        value = act( value )
        return value

    def test_cell(self, x):

        x = tf.expand_dims( x, axis = -1 )
        x = self.embeding( x )
        q, k, v = tf.split( x, 3, axis = -1 )

        w = tf.matmul( q, k, transpose_b = True )
        w = w * tf.math.rsqrt( tf.cast( v.shape[-1], w.dtype ) )        
        w = tf.nn.softmax( w )
        a = tf.matmul( w, v )
        
        value = self.norm( tf.reduce_sum( a * self.embeding_k, axis = 1 ) )

        return value

    def call(self, input):

        x = input

        c1 = self.perceptron_cell( x, self.kernel_1, self.bias_1, lambda v:v )                # linear
        c3 = self.nalu_cell( x )                                                              # arithimetic 
        c4 = self.test_cell( x )                                                              # attention
        features = tf.concat( [ c1, c3, c4 ], axis = -1 )                                     # features
        value = self.perceptron_cell( features, self.kernel_3, self.bias_3, self.activation ) # non-linear

        return value


class nalu_transform_w(Layer):

    def __init__(self, num_outputs, kernel_initializer, name, activation=lambda x:x, initializer=tf.keras.initializers.truncated_normal(), bias=True):
        
        super(nalu_transform_w, self).__init__(name=name)
        self.num_outputs = num_outputs
        self.use_bias = bias
        self.initializer = initializer
        self.kernel_initializer = kernel_initializer
        self.activation = activation      
    
    def build(self, input_shape):

        self.gt = self.add_weight( "w_gt", [ input_shape[ -1 ], self.num_outputs ], initializer = self.initializer )
        if self.use_bias:
            self.bias = self.add_weight( 'bias', shape = [ self.num_outputs ], initializer = tf.keras.initializers.Zeros() )

    def call(self, input, i, wt, mt, kn, we):

        exceded = tf.abs( ( tf.shape(wt)[0] * i ) - tf.shape(we)[0] )
        t_wt = tf.tile( wt, [ i, i ] )[exceded:,exceded:] * we
        t_mt = tf.tile( mt, [ i, i ] )[exceded:,exceded:] * we
        t_kn = tf.tile( kn, [ i, i ] )[exceded:,exceded:] * we

        wk = t_kn
        wa = tf.multiply( tf.tanh( t_wt ), tf.sigmoid( t_mt ) )

        x1, x2 = tf.unstack( input, axis = 1 )

        g = tf.sigmoid( tf.matmul( x1, self.gt ) )
        a = tf.matmul( x1, wa )
        m = tf.sinh( tf.matmul( tf.asinh( x1 ), wa ) )
        arithimetic_x = ( g * a ) + ( ( 1 - g ) * m )

        trnaformation_x = x2 @ wk
        if self.use_bias:
            trnaformation_x += self.bias        
        trnaformation_x = self.activation( trnaformation_x )

        return tf.stack( [ arithimetic_x, trnaformation_x ], axis = 1 )


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

    def __init__(self, name,  filters, kernel, stide, padding = 'same', activation = lambda x: x, initializer=tf.keras.initializers.he_normal(), bias=True):
        super(conv2d, self).__init__(name=name)
        self.f = filters
        self.k = kernel
        self.s = stide
        self.p = padding.upper()
        self.use_bias = bias
        self.activation = activation
        self.initializer = initializer
    
    def build(self, input_shape):

        k = [ self.k, self.k, input_shape[-1], self.f ] if type( self.k ) is int else [ self.k[0], self.k[1], input_shape[-1], self.f ]
        self.strides = ( 1, self.s, self.s, 1 ) if type( self.s ) is int else ( 1, self.s[0], self.s[1], 1 )
        
        self.kernel = self.add_weight( 'kernel', shape = k, initializer = self.initializer, trainable = True )
        if self.use_bias:
            self.bias = self.add_weight( 'bias', shape = [ self.f ], initializer = tf.keras.initializers.Zeros(), trainable = True )

    def call(self, input):

        c = tf.nn.conv2d( input, self.kernel, self.strides, padding = self.p )
        c = tf.nn.bias_add( c, self.bias, name = "_bias" )

        if self.use_bias:
            c += self.bias

        return self.activation( c )


class convnalu2d(Layer):

    def __init__(self, name,  filters, kernel, stide, padding = 'same', initializer=tf.keras.initializers.he_normal()):
        super(convnalu2d, self).__init__(name=name)
        self.f = filters
        self.k = kernel
        self.s = stide
        self.p = padding.upper()
        self.initializer = initializer
        self.pooling = GlobalAveragePooling2D()
    
    def build(self, input_shape):

        k = [ self.k, self.k, input_shape[-1], self.f ] if type( self.k ) is int else [ self.k[0], self.k[1], input_shape[-1], self.f ]
        self.strides = ( 1, self.s, self.s, 1 ) if type( self.s ) is int else ( 1, self.s[0], self.s[1], 1 )
                
        self.gt = self.add_weight( "w_gt", [ input_shape[ -1 ], self.f ], initializer = self.initializer, trainable = True  )
        self.wt = self.add_weight( "w_wt", k, initializer = self.initializer, trainable = True  )
        self.mt = self.add_weight( "w_mt", k, initializer = self.initializer, trainable = True )

    def call(self, input):
        
        avg_poll = tf.reshape( self.pooling( input ), [-1, 1, 1, input.shape[-1] ] )
        w = tf.multiply( tf.tanh( self.wt ), tf.sigmoid( self.mt ) )
        g = tf.sigmoid( tf.matmul( avg_poll, self.gt ) )
        a = tf.nn.conv2d( input, w, self.strides, padding = self.p )        
        m = tf.sinh( tf.nn.conv2d( tf.asinh( input ), w, self.strides, padding = self.p ) )
        arithimetic_x = ( g * a ) + ( ( 1 - g ) * m )

        return arithimetic_x


class conv2dt(Layer):

    def __init__(self, name, filters, kernel, stide, activation=lambda x:x, padding = 'same', initializer=tf.keras.initializers.he_normal(), bias=True):
        super(conv2dt, self).__init__(name=name)
        self.f = filters
        self.k = kernel
        self.s = stide
        self.p = padding.upper()
        self.use_bias = bias
        self.activation = activation
        self.initializer = initializer
    
    def build(self, input_shape):

        k = [ self.k, self.k, self.f, input_shape[-1] ] if type( self.k ) is int else [ self.k[0], self.k[1], self.f, input_shape[-1] ]
        self.strides = ( 1, self.s, self.s, 1 ) if type( self.s ) is int else ( 1, self.s[0], self.s[1], 1 )
        self.out_shape = [ input_shape[1] * self.s, input_shape[2] * self.s, self.f ] if type( self.s ) is int \
                    else [ input_shape[1] * self.s[0], input_shape[2] * self.s[1], self.f ]
        
        self.kernel = self.add_weight( 'kernel', shape = k, initializer = self.initializer, trainable = True )
        if self.use_bias:
            self.bias = self.add_weight( 'bias', shape = [ self.f ], initializer = tf.keras.initializers.Zeros(), trainable = True )

    def call(self, input):
        
        bs, *_ = shape_list( input )
        c = c = tf.nn.conv2d_transpose( input, self.kernel, [ bs ] + self.out_shape, self.strides, padding = self.p )
        c = tf.nn.bias_add( c, self.bias, name = "_bias" )

        if self.use_bias:
            c += self.bias
        return self.activation( c )


class conv1d(Layer):

    def __init__(self, nf, name, initializer=tf.keras.initializers.Orthogonal()):
        super(conv1d, self).__init__()  
        self.initializer = initializer
        self.nf = nf
        self.nme = name
        
    def build(self, input_shape):

        self.kernel = self.add_weight( '{}_kernel_1d'.format( self.nme ), shape = [ 1, input_shape[ -1 ], self.nf ], initializer = self.initializer )
        self.bias = self.add_weight( '{}_bias'.format( self.nme ), shape = [ self.nf ], initializer = tf.keras.initializers.Zeros() )

    def call(self, input):
        *start, nx = shape_list( input )
        i = tf.reshape( input, [ -1, nx ] )
        w = tf.reshape( self.kernel, [ -1, self.nf ] )
        c = tf.matmul( i, w )
        cb = c + self.bias
        r = tf.reshape( cb, start + [ self.nf ] )
        return r


class conv1dsnac(Layer):

    def __init__(self, nf, name, initializer=tf.keras.initializers.Orthogonal()):
        super(conv1dsnac, self).__init__()  
        self.initializer = initializer
        self.nf = nf
        self.nme = name
        
    def build(self, input_shape):

        self.wt = self.add_weight( "w_wt", [ 1, input_shape[ -1 ], self.nf ], initializer = self.initializer )
        self.mt = self.add_weight( "w_mt", [ 1, input_shape[ -1 ], self.nf ], initializer = self.initializer )

    def call(self, input):
        *start, nx = shape_list( input )
        i = tf.reshape( input, [ -1, nx ] )
        w = tf.reshape( tf.multiply( tf.tanh( self.wt ), tf.sigmoid( self.mt ) ), [ -1, self.nf ] )
        c = tf.matmul( i, w )
        r = tf.reshape( c, start + [ self.nf ] )
        return r


class transformer_block(Layer):

    def __init__(self, num_outputs, n_heads, initializer=tf.keras.initializers.he_normal()):
        
        super(transformer_block, self).__init__()

        self.qkv = conv1d( num_outputs * 3, 'qkv', initializer = initializer )
        self.ln1 = conv1d( num_outputs * 4, 'ln1', initializer = initializer )
        self.ln2 = conv1d( num_outputs, 'ln2', initializer = initializer )
        self.merge = conv1d( num_outputs, 'merge', initializer = initializer )

        self.norm1 = norm('n1')
        self.norm2 = norm('n2')
        
        self.n_heads = n_heads

        self.num_outputs = num_outputs
        
    def build(self, input_shape):
        pass

    def call(self, input, past, eval):
        
        x = input
        a, present, msk = attn( self.norm1( x ), past, self.qkv, self.merge, self.n_heads )
        xa = x + a
        m1 = gelu( self.ln1( self.norm2( xa ) ) )
        m = self.ln2( m1 )
        xm = xa + m

        return xm, present, msk


class transformer_snac_block(Layer):

    def __init__(self, num_outputs, n_heads, initializer=tf.keras.initializers.he_normal()):
        
        super(transformer_snac_block, self).__init__()

        self.qkv = conv1dsnac( num_outputs * 3, 'qkv', initializer = initializer )
        self.ln1 = conv1dsnac( num_outputs, 'ln1', initializer = initializer )
        self.merge = conv1dsnac( num_outputs, 'merge', initializer = initializer )

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
        m = self.ln1( self.norm2( xa ) )
        xm = xa + m

        return xm, present, msk


class transformer_layer(Layer):

    def __init__(self, num_outputs, n_heads, num_blocks, max_len, name, initializer=tf.keras.initializers.RandomNormal( stddev = 0.01 ), embeding_inputs=False):
        
        super(transformer_layer, self).__init__(name=name)

        self.max_len = max_len
        self.embeding_inputs = embeding_inputs
        self.num_outputs = num_outputs
        self.num_blocks = num_blocks
        self.initializer = initializer
        self.blocks = [ transformer_block( num_outputs, n_heads ) for _ in range( num_blocks ) ]
        
    def build(self, input_shape):
        
        if self.embeding_inputs:
            self.wie = self.add_weight( 'input_embeding', shape = [ input_shape[-1], self.num_outputs ], initializer = self.initializer )
        self.wpe = self.add_weight( 'positional_embeding', shape = [ self.max_len, self.num_outputs ], initializer = self.initializer )
        
    def call(self, input, past, eval):

        if self.embeding_inputs:
            x = tf.matmul( input, self.wie )
        else:
            x = input
        
        past_length = 0 if past is None else tf.shape(past)[-2]
        h = x + tf.gather( self.wpe, positions_for( x, past_length ) )

        presents = []
        msks = []
        pasts = tf.unstack( past, axis = 1 ) if not past is None else [ None ] * self.num_blocks

        for idx, past in enumerate( pasts ):            
            h, present, msk = self.blocks[idx]( h, past, eval )
            presents.append( present )
            msks.append( msk )
        
        presents = tf.stack( presents, axis = 1 )        
        msks = tf.stack( msks, axis = 0 )

        return h, presents, msks


class transformer_nac_layer(Layer):

    def __init__(self, num_outputs, n_heads, num_blocks, max_len, initializer=tf.keras.initializers.RandomNormal( stddev = 0.01 )):
        
        super(transformer_nac_layer, self).__init__()

        self.max_len = max_len
        self.num_outputs = num_outputs
        self.num_blocks = num_blocks
        self.initializer = initializer
        self.blocks1 = [ transformer_snac_block( num_outputs, n_heads ) for _ in range( num_blocks ) ]        
        self.blocks2 = [ transformer_block( num_outputs, n_heads ) for _ in range( num_blocks ) ]
    
    def build(self, input_shape):
        
        self.wpe = self.add_weight( 'positional_embeding', shape = [ self.max_len, self.num_outputs ], initializer = self.initializer )
        
    def call(self, input, past):
        
        x1, x2 = tf.unstack( input, axis = 1 )

        past_length = 0 if past is None else tf.shape(past)[-2]
        h1 = x1 + tf.gather( self.wpe, positions_for( x1, past_length ) )
        h2 = x2 + tf.gather( self.wpe, positions_for( x2, past_length ) )

        presents = [[],[]]
        msks = [[],[]]
        pasts1, pasts2 = tf.unstack( past, axis = 1 ) if not past is None else ( [ None ] * self.num_blocks, [ None ] * self.num_blocks )
        if not past is None:
            pasts1 = tf.unstack( pasts1, axis = 1 )
            pasts2 = tf.unstack( pasts2, axis = 1 )

        for idx, past in enumerate( zip( pasts1, pasts2 ) ):
            
            h1, present1, msk1 = self.blocks1[idx]( h1, past[0] )
            h2, present2, msk2 = self.blocks2[idx]( h2, past[1] )

            presents[0].append( present1 )
            presents[1].append( present2 )

            msks[0].append( msk1 )
            msks[1].append( msk2 )

        presents[0] = tf.stack( presents[0], axis = 1 )
        presents[1] = tf.stack( presents[1], axis = 1 )
        
        presents = tf.stack( presents, axis = 1 )
        
        msks[0] = tf.stack( msks[0], axis = 0 )
        msks[1] = tf.stack( msks[1], axis = 0 )

        msks = tf.stack( msks, axis = 0 )

        return tf.stack( ( h1, h2 ), axis = 1 ), presents, msks


class vector_quantizer(Layer):

    """Neural Discrete Representation Learning (https://arxiv.org/abs/1711.00937)"""
    
    def __init__(self, embedding_dim, num_embeddings, commitment_cost, name):
        
        super(vector_quantizer, self).__init__( name = name )
        
        # embedding_dim: D, num_embeddings: K
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
    def build(self, input_shape):
        # (D, K)
        self.w = self.add_weight( 'embedding', shape = [ self.embedding_dim, self.num_embeddings ], initializer = GlorotUniform(), trainable = True )
        
    def call(self, inputs):

        # (BxHxW, D)
        flat_inputs = tf.reshape( inputs, [ -1, self.embedding_dim ] )
        
        # (BxHxW, K) = (BxHxW, 1)                                              - (BxHxW, D) x (D, K)                  + (1, K)
        distances    = ( tf.reduce_sum( flat_inputs**2, 1, keepdims = True ) ) - 2 * tf.matmul( flat_inputs, self.w ) + tf.reduce_sum( self.w**2, 0, keepdims = True )
        
        encoding_indices = tf.argmax( -distances, 1 ) # (BxHxW)
        encodings = tf.one_hot( encoding_indices, self.num_embeddings ) # (BxHxW, K)
        encoding_indices = tf.reshape( encoding_indices, tf.shape( inputs )[:-1] ) # (B, H, W)
        quantized = self.quantize( encoding_indices ) # NOTICE (B, H, W, D)
        
        e_latent_loss = tf.reduce_mean( ( tf.stop_gradient( quantized ) - inputs ) ** 2 )
        q_latent_loss = tf.reduce_mean( ( quantized - tf.stop_gradient( inputs ) ) ** 2 )
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # skip gradient to compute derivates for prior layers
        quantized_skip = inputs + tf.stop_gradient( quantized - inputs )
        
        avg_probs = tf.reduce_mean( encodings, 0 )
        # It indicates how many codes are 'active' on average.
        perplexity = tf.exp( - tf.reduce_sum( avg_probs * tf.math.log( avg_probs + 1e-10 ) ) )
        
        return {
            'quantize': quantized_skip,
            'loss': loss,
            'perplexity': perplexity,
            'encodings': encodings,
            'encoding_indices': encoding_indices
        }

    def get_vq(self, encoding_indices):
        quantized = self.quantize( encoding_indices )
        return quantized

    @property
    def embeddings(self):
        return self.w

    def quantize(self, encoding_indices): # (B, H, W)
        w = tf.transpose( self.embeddings.read_value(), [1,0] ) # (K, D)
        return tf.nn.embedding_lookup( w, encoding_indices )  # (B, H, W, D)


"""
Source from: https://github.com/ajithcodesit/Neural_Turing_Machine
"""
class ntm_read_head(Layer):

    def __init__(self, memory_locations=128, memory_vector_size=20, maximum_shifts=3):
        super(ntm_read_head, self).__init__()        
        self.memory_locations = memory_locations
        self.memory_vector_size = memory_vector_size
        self.maximum_shifts = maximum_shifts
        self.addr_mech = ntm_addressing( memory_locations, memory_vector_size, maximum_shifts, reading = True )
        
    def call(self, input, w_t_1, M_t, aux_M_t=None):
        
        w_t = self.addr_mech( input, w_t_1, M_t )
        r_t = tf.squeeze( tf.matmul( tf.expand_dims( w_t, axis = 1 ), M_t ), axis = 1 )

        if aux_M_t is None: return r_t, w_t, None
        
        a_r_t = tf.squeeze( tf.matmul( tf.expand_dims( w_t, axis = 1 ), aux_M_t ), axis = 1 )
        return r_t, w_t, a_r_t


class ntm_write_head(Layer):

    def __init__(self, memory_locations=128, memory_vector_size=20, maximum_shifts=3):
        super(ntm_write_head, self).__init__()        
        self.memory_locations = memory_locations
        self.memory_vector_size = memory_vector_size
        self.maximum_shifts = maximum_shifts
        self.addr_mech = ntm_addressing( memory_locations, memory_vector_size, maximum_shifts, reading = False )
        
    def call(self, input, w_t_1, M_t_1, aux_M_t_1=None, input_plus=None):

        w_t, e_t, a_t = self.addr_mech( input, w_t_1, M_t_1 )
        
        w_t = tf.expand_dims( w_t, axis = 1 )

        # Erase
        e_t = tf.expand_dims( e_t, axis = 1 )
        M_tidle_t = tf.multiply( M_t_1, ( 1.0 - tf.matmul( w_t, e_t, transpose_a = True ) ) )

        # Add
        a_t = tf.expand_dims( a_t, axis = 1 )
        M_t = M_tidle_t + tf.matmul( w_t, a_t, transpose_a = True )

        if aux_M_t_1 is None: return M_t, tf.squeeze( e_t, axis = 1 ), tf.squeeze( a_t, axis = 1 ), tf.squeeze( w_t, axis = 1 )
        
        # Erase
        a_M_tidle_t = tf.multiply( aux_M_t_1, ( 1.0 - tf.matmul( w_t, e_t, transpose_a = True ) ) )
        
        # Add
        input_plus = tf.expand_dims( input_plus, axis = 1 )
        a_M_t = a_M_tidle_t + tf.matmul( w_t, input_plus, transpose_a = True )

        return M_t, a_M_t, tf.squeeze( e_t, axis = 1 ), tf.squeeze( a_t, axis = 1 ), tf.squeeze( w_t, axis = 1 )


class ntm_addressing(Layer):

    def __init__(self, memory_locations=128, memory_vector_size=20, maximum_shifts=3, reading=True):
        
        super(ntm_addressing, self).__init__()        
        
        self.memory_locations = memory_locations
        self.memory_vector_size = memory_vector_size
        self.maximum_shifts = maximum_shifts
        self.reading = reading

        self.read_split = [ self.memory_vector_size, 1, 1, self.maximum_shifts, 1 ]
        self.write_split = [ self.memory_vector_size, 1, 1, self.maximum_shifts, 1, self.memory_vector_size, self.memory_vector_size ]

        if self.reading: self.emit_len = np.sum( self.read_split )
        else: self.emit_len = np.sum( self.write_split )

        self.fc_addr = Dense( units = self.emit_len, activation = tf.keras.activations.tanh, name = "emit_params",
                              kernel_initializer = 'glorot_uniform', bias_initializer = 'glorot_normal' )
        
    """
        1 - Key vector
        2 - Key strength
        3 - Interpolation gate
        4 - Shift weighting
        5 - Sharpen
    """
    def emit_addressing_params(self, k_t, beta_t, g_t, s_t, gamma_t):
        return k_t, tf.nn.softplus( beta_t ), tf.nn.sigmoid( g_t ), tf.nn.softmax( s_t, axis = -1 ), 1.0 + tf.nn.softplus( gamma_t )               

    def emit_head_params(self, fc_output):

        if self.reading:
            k_t, beta_t, g_t, s_t, gamma_t = tf.split( fc_output, self.read_split, axis = -1 )
            k_t, beta_t, g_t, s_t, gamma_t = self.emit_addressing_params( k_t, beta_t, g_t, s_t, gamma_t )
            return k_t, beta_t, g_t, s_t, gamma_t, None, None        
        else:
            k_t, beta_t, g_t, s_t, gamma_t, e_t, a_t = tf.split( fc_output, self.write_split, axis = -1 )
            k_t, beta_t, g_t, s_t, gamma_t = self.emit_addressing_params( k_t, beta_t, g_t, s_t, gamma_t )
            return k_t, beta_t, g_t, s_t, gamma_t, tf.nn.sigmoid( e_t ), a_t

    @staticmethod
    def cosine_similarity(k, m):
        k_mag = tf.sqrt( tf.reduce_sum( tf.square( k ), axis = -1 ) )
        m_mag = tf.sqrt( tf.reduce_sum( tf.square( m ), axis =- 1 ) )
        mag_prod = tf.multiply( k_mag, m_mag )
        dot = tf.squeeze( tf.keras.layers.dot( [ k, m ], axes = ( -1, -1 ) ), axis = 1 )
        return tf.divide( dot, mag_prod )
    
    @staticmethod
    def circular_convolution(w, s):
        kernels = tf.TensorArray( dtype = s.dtype, size = s.shape[0] )
        for i in tf.range( 0, s.shape[0] ):
            kernels = kernels.write( i, tf.roll( w, shift = ( i - tf.math.floordiv( s.shape[0], 2 ) ), axis = 0 ) )
        w_circ_conv = tf.transpose( kernels.stack() )
        return tf.reduce_sum( w_circ_conv * s, axis = 1 )

    def content_addressing(self, M_t, k_t, beta_t):
        k_t = tf.expand_dims( k_t, axis = 1 )
        w_c_t = tf.nn.softmax( beta_t * self.cosine_similarity( k_t, M_t ), axis =- 1 )
        return w_c_t
    
    def interpolation(self, w_t_prev, g_t, w_c_t):
        w_g_t = ( g_t * w_c_t ) + ( ( 1 - g_t ) * w_t_prev )
        return w_g_t
    
    def convolutional_shift(self, w_g_t, s_t):
        convolved_weights = tf.TensorArray( dtype = w_g_t.dtype, size = w_g_t.shape[0] )        
        for i in tf.range( s_t.shape[0] ):
            cc = self.circular_convolution( w_g_t[i], s_t[i] )
            convolved_weights = convolved_weights.write( i, cc )        
        w_tidle_t = convolved_weights.stack()
        return w_tidle_t

    def sharpening(self, w_tidle_t, gamma_t):
        w_raised = tf.pow( w_tidle_t, gamma_t )
        w_t = tf.divide( w_raised, tf.reduce_sum( w_raised, axis = -1, keepdims = True ) )
        return w_t
    
    def call(self, controller_output, w_t_prev, M_t):
        
        # Controller outputs used for addressing
        k_t, beta_t, g_t, s_t, gamma_t, e_t, a_t = self.emit_head_params( self.fc_addr( controller_output ) )

        # Addressing main mechanism
        w_c_t = self.content_addressing( M_t, k_t, beta_t )
        w_g_t = self.interpolation( w_t_prev, g_t, w_c_t )
        w_tidle_t = self.convolutional_shift( w_g_t, s_t )
        w_t = self.sharpening( w_tidle_t, gamma_t )

        if self.reading:
            return w_t  # The new weight over the N locations of the memory matrix, and
        else:
            return w_t, e_t, a_t
    

class ntm_plus_memory(Layer):

    def __init__(self,controller_size=100, memory_locations=128, memory_vector_size=20, maximum_shifts=3,
                 output_size=8, learn_r_bias=False, learn_w_bias=False, learn_m_bias=False):

        super(ntm_plus_memory, self).__init__()
    
        self.memory_locations = memory_locations  # N locations
        self.memory_vector_size = memory_vector_size  # M size memory vectors
        self.maximum_shifts = maximum_shifts
        
        self.rc = dense( memory_vector_size // 2, tf.keras.initializers.glorot_uniform() )
        self.ac = dense( memory_vector_size // 2, tf.keras.initializers.glorot_uniform() )
        self.sc = nalu_transform( controller_size // 2, tf.keras.initializers.glorot_uniform(), tf.keras.activations.gelu )
        self.controller = dense( controller_size, tf.keras.initializers.glorot_uniform() )

        self.read_head = ntm_read_head( self.memory_locations, self.memory_vector_size, self.maximum_shifts )
        self.write_head = ntm_write_head( self.memory_locations, self.memory_vector_size, self.maximum_shifts )

        self.final_fc = nalu_transform( output_size, tf.keras.initializers.glorot_uniform() )

        # The bias vector (These bias vectors can be learned or initialized to the same value)
        self.r_bias = tf.Variable( tf.random.truncated_normal( [ 1, self.memory_vector_size ], mean = 0.0, stddev = 0.5), trainable = learn_r_bias )
        self.w_bias = tf.Variable( tf.nn.softmax( tf.random.normal( [ 1, self.memory_locations ] ) ), trainable = learn_w_bias )
        self.M_bias = tf.Variable( tf.ones( [ 1, self.memory_locations, self.memory_vector_size ] ) * 1e-6, trainable = learn_m_bias )

        # Extra outputs that are tracked
        self.e_t = None
        self.a_t = None

        # For visualizing the NTM working (Must be used only during predictions)
        self.reads = []
        self.adds = []
        self.read_weights = []
        self.write_weights = []

    def reset_ntm_state(self, batch_size):  # Creates a new NTM state
        # This has to be manually called if stateful is set to true
        w_t_1 = tf.tile( self.w_bias, [ batch_size, 1 ] )
        M_t = tf.tile( self.M_bias, [ batch_size, 1, 1 ] )
        return w_t_1, M_t

    def call(self, state, _state, reward, actions, M_t, p_M_t, w_t_1, stateful=False, training=True):
        
        # if not stateful:  # A new state will not be created at the start of each new batch
        #     self.reset_debug_vars()
            
        r = tf.keras.activations.gelu( self.rc( reward ) )
        a = tf.keras.activations.gelu( self.ac( actions ) )

        s = self.sc( state )
        s_ = self.sc( _state )

        hs1, hs2 = tf.unstack( s, axis = 1 )
        hs1_, hs2_ = tf.unstack( s_, axis = 1 )

        hs = tf.concat( [ hs1, hs2 ], axis = -1 )
        hs_ = tf.concat( [ hs1_, hs2_ ], axis = -1 )
        
        hs_plus = tf.concat( [ r, a ], axis = -1 )

        controller_read_outputs = self.controller( hs )  # [Batch size, Controller size]
        controller_write_outputs = self.controller( hs_ )  # [Batch size, Controller size]

        # [Batch size, M, N], [Batch size, M], [Batch size, M], [Batch size, N]
        M_t, p_M_t, e_t, a_t, w_t_1 = self.write_head( controller_write_outputs, w_t_1, M_t, p_M_t, hs_plus )
        if not training:
            self.adds.append( a_t.numpy()[0] )
            self.write_weights.append( e_t.numpy()[0] )

        # [Batch size, M], [Batch size, N]
        r_t_1, w_t_1, a_r_t_1 = self.read_head( controller_read_outputs, w_t_1, M_t, p_M_t )
        if not training:
            self.reads.append( r_t_1.numpy()[0] )
            self.read_weights.append( w_t_1.numpy()[0] )

        fc_input = tf.concat( [ controller_read_outputs, r_t_1, a_r_t_1 ], axis = -1 )  # [Batch size, Controller size + M + PM],
        fc_input = tf.stack( [ fc_input, fc_input ], axis = 1 )  # [Batch size, 2, Controller size + M + PM],
        output_t = tf.keras.activations.gelu( self.final_fc( fc_input ) ) # [Batch size, Output size]

        return output_t, M_t, p_M_t, w_t_1


"""
Source from: https://github.com/arnaudvl/differentiable-neural-computer
"""
class naludnc(Layer):

    def __init__(self, output_dim, memory_shape=(100, 20), memory_aux_size=10, n_read=3):

        super(dnc, self).__init__()

        # define output data size
        self.output_dim = output_dim  # Y
        
        # define size of memory matrix
        self.N, self.W = memory_shape  # N, W
        self.mem_aux = memory_aux_size

        self.state_size = [ 
            tf.TensorShape( [ self.N, self.W ] ),
            tf.TensorShape( [ self.N, 1 ] ),
            tf.TensorShape( [ self.N, self.N ] ),
            tf.TensorShape( [ self.N, 1 ] ),
            tf.TensorShape( [ self.N, 3 ] ),
            tf.TensorShape( [ self.N, 1 ] ),
         ]

        # define number of read heads
        self.R = n_read  # R

        # size of output vector from controller that defines interactions with memory matrix:
        # R read keys + R read strengths + write key + write strength + erase vector +
        # write vector + R free gates + allocation gate + write gate + R read modes
        self.interface_dim = self.R * self.W + 3 * self.W + 5 * self.R + 3  # I

        # neural net output = output of controller + interface vector with memory
        self.controller_dim = self.output_dim + self.interface_dim  # Y+I

        # controller variables
        # initialize controller hidden state
        self.fc1 = nalu_transform( self.controller_dim, kernel_initializer = tf.keras.initializers.RandomNormal() )
        self.fc2 = dense( self.controller_dim, initializer = tf.keras.initializers.RandomNormal() )

    def build(self, input_shapes):

        # define and initialize weights for controller output and interface vectors
        self.W_output = self.add_weight( name = 'dnc_net_output_weights', shape = [ self.controller_dim, self.output_dim ], 
                                         initializer = tf.keras.initializers.truncated_normal( stddev = 0.1 ) ) # [Y+I,Y]
        
        self.W_interface = self.add_weight( name = 'dnc_interface_weights', shape = [ self.controller_dim, self.interface_dim ], 
                                            initializer = tf.keras.initializers.truncated_normal( stddev = 0.1 ) ) # [Y+I,I]

        self.W_read_out = self.add_weight( name = 'dnc_read_vector_weights', shape = [ self.R * self.W, self.output_dim ], 
                                           initializer = tf.keras.initializers.truncated_normal( stddev = 0.1 ) ) # [R*W,Y]
        
    def reset(self, bs):

        # initialize memory matrix with zeros
        M = tf.zeros( [ bs, self.N, self.W ] )  # [N,W]

        # usage vector records which locations in the memory are used and which are free
        usage = tf.fill( [ bs, self.N, 1 ], 1e-6 )  # [N,1]

        # temporal link matrix L[i,j] records to which degree location i was written to after j
        L = tf.zeros( [ bs, self.N, self.N ] )  # [N,N]

        # precedence vector determines degree to which a memory row was written to at t-1
        W_precedence = tf.zeros( [ bs, self.N, 1 ] )  # [N,1]

        # initialize R read weights and vectors and write weights
        W_read = tf.fill( [ bs, self.N, self.R ], 1e-6 )  # [N,R]
        W_write = tf.fill( [ bs, self.N, 1 ], 1e-6 )  # [N,1]

        return M, usage, L, W_precedence, W_read, W_write
    
    def content_lookup(self, key, strength, M):
        """
        Attention mechanism: content based addressing to read from and write to the memory.
        Params
        ------
        key
            Key vector emitted by the controller and used to calculate row-by-row
            cosine similarity with the memory matrix.
        strength
            Strength scalar attached to each key vector (1x1 or 1xR).
        Returns
        -------
        Similarity measure for each row in the memory used by the read heads for associative
        recall or by the write head to modify a vector in memory.
        """
        # The l2 norm applied to each key and each row in the memory matrix
        norm_mem = tf.nn.l2_normalize( M, -1 )  # [N,W]
        norm_key = tf.nn.l2_normalize( key, -1 )     # [1,W] for write or [R,W] for read

        # get similarity measure between both vectors, transpose before multiplication
        # write: [N*W]*[W*1] -> [N*1]
        # read: [N*W]*[W*R] -> [N,R]
        sim = tf.matmul( norm_mem, norm_key, transpose_b = True )
        return tf.nn.softmax( sim * strength, 1 )  # [N,1] or [N,R]
    
    def allocation_weighting(self, usage):

        bs = tf.shape( usage )[0]

        """
        Memory needs to be freed up and allocated in a differentiable way.
        The usage vector shows how much each memory row is used.
        Unused rows can be written to. Usage of a row increases if
        we write to it and can decrease if we read from it, depending on the free gates.
        Allocation weights are then derived from the usage vector.
        Returns
        -------
        Allocation weights for each row in the memory.
        """
        # sort usage vector in ascending order and keep original indices of sorted usage vector
        sorted_usage, free_list = tf.nn.top_k( -1 * tf.transpose( usage, perm = [ 0, 2, 1 ] ), k = self.N )
        sorted_usage *= -1
        cumprod = tf.math.cumprod( sorted_usage, axis = -1, exclusive = True )
        unorder = ( 1 - sorted_usage ) * cumprod

        W_alloc = tf.zeros( [ bs, self.N ] )
        I = tf.tile( tf.constant( np.identity( self.N, dtype = np.float32 ) )[tf.newaxis,...], [ bs, 1, 1 ] )

        # for each usage vec
        for pos, idx in enumerate( tf.unstack( free_list[:,0,:], axis = -1 ) ):
            # flatten
            
            # como tem dimensao de batch, é nescessario usar onehot para filtrar a linha
            # m = tf.squeeze( tf.slice( I, [ idx , 0 ], [ 1, -1 ] ), axis = 1 )
            one_hot = tf.one_hot( idx, tf.shape( free_list )[-1] )[:,:,tf.newaxis]
            m = tf.reduce_sum( I * one_hot, axis = 1 )
            
            # add to allocation weight matrix
            W_alloc += m * unorder[ :, 0, pos ][:,tf.newaxis]

        # return the allocation weighting for each row in memory
        # return tf.reshape( W_alloc, [ bs, self.N, 1 ] )
        return W_alloc[:,:,tf.newaxis]
    
    def controller(self, x):
        x1 = self.fc1( x )
        h1, h2 = tf.unstack( x1, axis = 1 )
        h = tf.concat( [ h1, h2 ], axis = -1 )        
        x2 = tf.nn.tanh( self.fc2( h ) )
        return x2
    
    def partition_interface(self, interface):

        bs = tf.shape( interface )[0]

        """
        Partition the interface vector in the read and write keys and strengths,
        the free, allocation and write gates, read modes and erase and write vectors.
        """
        # convert interface vector into a set of read write vectors
        # partition = tf.constant( [ [0] * ( self.R * self.W ) + [1] * self.R +
        #                            [2] * self.W + [3] + [4] * self.W + [5] * self.W +
        #                            [6] * self.R + [7] + [8] + [9] * (self.R * 3 ) ],
        #                         dtype = tf.int32 )

        # ( k_read, b_read, k_write, b_write, erase, write_v, free_gates, alloc_gate,
        #  write_gate, read_modes ) = tf.dynamic_partition( interface[0], partition, 10 )

        partition = [ ( self.R * self.W ), self.R, self.W, 1, self.W, self.W, self.R, 1, 1, ( self.R * 3 ) ]
        ( k_read, b_read, k_write, b_write, erase, write_v, free_gates, alloc_gate,
          write_gate, read_modes ) = tf.split( interface, partition, axis = -1 )

        # R read keys and strengths
        k_read = tf.reshape( k_read, [ bs, self.R, self.W ] )       # [R,W]
        b_read = 1 + tf.nn.softplus( tf.expand_dims( b_read, 1 ) )  # [1,R]

        # write key, strength, erase and write vectors
        k_write = tf.expand_dims( k_write, 1 )                        # [1,W]
        b_write = 1 + tf.nn.softplus( tf.expand_dims( b_write, 1 ) )  # [1,1]
        erase = tf.nn.sigmoid( tf.expand_dims( erase, 1 ) )           # [1,W]
        write_v = tf.expand_dims( write_v, 1 )                        # [1,W]

        # the degree to which locations at read heads will be freed
        free_gates = tf.nn.sigmoid( tf.expand_dims( free_gates, 1 ) )  # [1,R]

        # the fraction of writing that is being allocated in a new location
        alloc_gate = tf.reshape( tf.nn.sigmoid( alloc_gate ), [bs, 1] )  # 1

        # the amount of information to be written to memory
        write_gate = tf.reshape( tf.nn.sigmoid( write_gate ), [bs, 1] )  # 1

        # softmax distribution over the 3 read modes (forward, content lookup, backward)
        read_modes = tf.reshape( read_modes, [ bs, 3, self.R ] )  # [3,R]
        read_modes = tf.nn.softmax( read_modes, axis = 1 )

        return ( k_read, b_read, k_write, b_write, erase, write_v,
                free_gates, alloc_gate, write_gate, read_modes )

    def write(self, free_gates, alloc_gate, write_gate, k_write, b_write, erase, write_v, usage, W_read, W_write, M):

        bs = tf.shape( W_read )[0]
        
        """ Write to the memory matrix. """
        # memory retention vector represents by how much each location will not be freed by the free gates
        retention = tf.reduce_prod( 1 - free_gates * W_read, axis = -1 )
        retention = tf.reshape( retention, [ bs, self.N, 1 ] )  # [N,1]

        # update usage vector which is used to dynamically allocate memory
        usage = ( usage + W_write - usage * W_write ) * retention

        # compute allocation weights using dynamic memory allocation
        W_alloc = self.allocation_weighting( usage )  # [N,1]
        W_alloc = tf.reshape( W_alloc, [ bs, self.N ] )

        # apply content lookup for the write vector to figure out where to write to
        W_lookup = self.content_lookup( k_write, b_write, M )
        W_lookup = tf.reshape( W_lookup, [ bs, self.N ] )  # [N,1]

        # define our write weights now that we know how much space to allocate for them and where to write to
        W_write = write_gate * ( alloc_gate * W_alloc + ( 1 - alloc_gate ) * W_lookup )

        # update memory matrix: erase memory and write using the write weights and vector
        M = ( ( M * ( 1 - tf.matmul( W_write[:,:,tf.newaxis], erase ) ) ) + tf.matmul( W_write[:,:,tf.newaxis], write_v ) )

        return usage, W_write, M

    def read(self, k_read, b_read, read_modes, W_write, L, W_precedence, W_read, M):
        
        bs = tf.shape( W_read )[0]

        """ Read from the memory matrix. """
        # update memory link matrix used later for the forward and backward read modes
        W_write_cast = tf.matmul( W_write[:,:,tf.newaxis], tf.ones( [ bs, 1, self.N ] ) )  # [bs,N,N]        
        L = ( ( 1 - W_write_cast - tf.transpose( W_write_cast, perm=[0,2,1] ) ) * L + tf.matmul( W_write[:,:,tf.newaxis], W_precedence, transpose_b = True ) )  # [N,N]
        L *= ( tf.ones( [ bs, self.N, self.N ] ) - tf.tile( tf.constant( np.identity( self.N, dtype = np.float32 ) )[tf.newaxis,...], [ bs, 1, 1 ] ) )

        # update precedence vector which determines degree to which a memory row was written to at t-1
        W_precedence = ( ( 1 - tf.reduce_sum( W_write, axis = 1 ) )[:,tf.newaxis] * W_precedence[:,:,0] + W_write )[:,:,tf.newaxis]

        # apply content lookup for the read vector(s) to figure out where to read from
        W_lookup = self.content_lookup( k_read, b_read, M )
        #W_lookup = tf.reshape( W_lookup, [ bs, self.N, self.R ] )  # [N,R]

        # compute forward and backward read weights using the link matrix
        # forward weights recall information written in sequence and backward weights in reverse
        W_fwd = tf.matmul( L, W_read )  # [N,N]*[N,R] -> [N,R]
        W_bwd = tf.matmul( L, W_read, transpose_a = True)  # [N,R]

        # 3 modes: forward, backward and content lookup
        fwd_mode    = read_modes[:,2][:,tf.newaxis,:] * W_fwd
        lookup_mode = read_modes[:,1][:,tf.newaxis,:] * W_lookup
        bwd_mode    = read_modes[:,0][:,tf.newaxis,:] * W_bwd

        # read weights = backward + content lookup + forward mode weights
        W_read = bwd_mode + lookup_mode + fwd_mode  # [N,R]

        # create read vectors by applying read weights to memory matrix
        read_v = tf.transpose( tf.matmul( M, W_read, transpose_a = True ), perm = [ 0, 2, 1 ] )  # ([W,N]*[N,R])^T -> [R,W]

        return L, W_precedence, W_read, read_v

    def step_write(self, x, usage, W_read, W_write, M):

        interface = tf.matmul( x, self.W_interface )  # [1,Y+I] * [Y+I,I] -> [1,I]

        # partition the interface vector
        ( _, _, k_write, b_write, erase, write_v, free_gates, alloc_gate, write_gate, _ ) = self.partition_interface( interface )

        # write to memory
        n_usage, n_W_write, n_M = self.write( free_gates, alloc_gate, write_gate, k_write, b_write, erase, write_v, usage, W_read, W_write, M )

        return n_M, n_W_write, n_usage

    def step_read(self, x, W_write, L, W_precedence, W_read, M):

        interface = tf.matmul( x, self.W_interface )  # [1,Y+I] * [Y+I,I] -> [1,I]

        # partition the interface vector
        ( k_read, b_read, _, _, _, _, _, _, _, read_modes ) = self.partition_interface( interface )

        n_L, n_W_precedence, n_W_read, read_v = self.read( k_read, b_read, read_modes, W_write, L, W_precedence, W_read, M )

        return n_L, n_W_precedence, n_W_read, read_v

    def step(self, x_read, x_write, M, usage, L, W_precedence, W_read, W_write):
        
        bs = tf.shape(x_read)[0]

        """
        Update the controller, compute the output and interface vectors,
        write to and read from memory and compute the output.
        """
        # update controller
        controller_read_out = self.controller( x_read )
        controller_write_out = self.controller( x_write )

        # write to memory
        n_M, n_W_write, n_usage = self.step_write( controller_write_out, usage, W_read, W_write, M )

        # read from memory
        n_L, n_W_precedence, n_W_read, read_v = self.step_read( controller_read_out, n_W_write, L, W_precedence, W_read, n_M )
        
        # flatten read vectors and multiply them with W matrix before adding to controller output
        read_v_out = tf.matmul( tf.reshape( read_v, [ bs, self.R * self.W ] ), self.W_read_out )  # [1,RW]*[RW,Y] -> [1,Y]

        # compute output vectors
        output_v = tf.matmul( controller_read_out, self.W_output )  # [1,Y+I] * [Y+I,Y] -> [1,Y]

        # compute output
        y = output_v + read_v_out
        
        return y, n_M, n_usage, n_L, n_W_precedence, n_W_read, n_W_write[:,:,tf.newaxis]

    def call(self, inputs, states):

        x_read, x_write = tf.unstack( inputs, axis = 1 )

        M, usage, L, W_precedence, W_read, W_write = states

        ( y_seq, n_M, n_usage, n_L, n_W_precedence, n_W_read, 
          n_W_write ) = self.step( x_read, x_write, M, usage, L, W_precedence, W_read, W_write )

        return y_seq, ( n_M, n_usage, n_L, n_W_precedence, n_W_read, n_W_write )


class dnc(Layer):

    def __init__(self, output_dim, memory_shape=(100, 20), n_read=3, external_controller=False):

        super(dnc, self).__init__()

        self.external_controller = external_controller

        # define output data size
        self.output_dim = output_dim  # Y
        
        # define size of memory matrix
        self.N, self.W = memory_shape  # N, W

        self.state_size = [ 
            tf.TensorShape( [ self.N, self.W ] ),
            tf.TensorShape( [ self.N, 1 ] ),
            tf.TensorShape( [ self.N, self.N ] ),
            tf.TensorShape( [ self.N, 1 ] ),
            tf.TensorShape( [ self.N, n_read ] ),
            tf.TensorShape( [ self.N, 1 ] ),
         ]

        # define number of read heads
        self.R = n_read  # R

        # size of output vector from controller that defines interactions with memory matrix:
        # R read keys + R read strengths + write key + write strength + erase vector +
        # write vector + R free gates + allocation gate + write gate + R read modes
        self.interface_dim = self.R * self.W + 3 * self.W + 5 * self.R + 3  # I

        # neural net output = output of controller + interface vector with memory
        self.controller_dim = self.output_dim + self.interface_dim  # Y+I

        

        if not external_controller:
            # controller variables
            # initialize controller hidden state
            self.fc1 = dense( self.controller_dim, initializer = tf.keras.initializers.RandomNormal() )
            self.fc2 = dense( self.controller_dim, initializer = tf.keras.initializers.RandomNormal() )
        
    def build(self, input_shapes):

        # define and initialize weights for controller output and interface vectors
        
        if not self.external_controller:

            self.W_output = self.add_weight( name = 'dnc_net_output_weights', shape = [ self.controller_dim, self.output_dim ], 
                                            initializer = tf.keras.initializers.truncated_normal( stddev = 0.1 ) ) # [Y+I,Y]
        
        self.W_interface = self.add_weight( name = 'dnc_interface_weights', shape = [ self.controller_dim, self.interface_dim ], 
                                            initializer = tf.keras.initializers.truncated_normal( stddev = 0.1 ) ) # [Y+I,I]

        self.W_read_out = self.add_weight( name = 'dnc_read_vector_weights', shape = [ self.R * self.W, self.output_dim ], 
                                           initializer = tf.keras.initializers.truncated_normal( stddev = 0.1 ) ) # [R*W,Y]
        
    def reset(self, bs):

        # initialize memory matrix with zeros
        M = tf.zeros( [ bs, self.N, self.W ] )  # [N,W]

        # usage vector records which locations in the memory are used and which are free
        usage = tf.fill( [ bs, self.N, 1 ], 1e-6 )  # [N,1]

        # temporal link matrix L[i,j] records to which degree location i was written to after j
        L = tf.zeros( [ bs, self.N, self.N ] )  # [N,N]

        # precedence vector determines degree to which a memory row was written to at t-1
        W_precedence = tf.zeros( [ bs, self.N, 1 ] )  # [N,1]

        # initialize R read weights and vectors and write weights
        W_read = tf.fill( [ bs, self.N, self.R ], 1e-6 )  # [N,R]
        W_write = tf.fill( [ bs, self.N, 1 ], 1e-6 )  # [N,1]

        return M, usage, L, W_precedence, W_read, W_write
    
    def content_lookup(self, key, strength, M):
        """
        Attention mechanism: content based addressing to read from and write to the memory.
        Params
        ------
        key
            Key vector emitted by the controller and used to calculate row-by-row
            cosine similarity with the memory matrix.
        strength
            Strength scalar attached to each key vector (1x1 or 1xR).
        Returns
        -------
        Similarity measure for each row in the memory used by the read heads for associative
        recall or by the write head to modify a vector in memory.
        """
        # The l2 norm applied to each key and each row in the memory matrix
        norm_mem = tf.nn.l2_normalize( M, -1 )  # [N,W]
        norm_key = tf.nn.l2_normalize( key, -1 )     # [1,W] for write or [R,W] for read

        # get similarity measure between both vectors, transpose before multiplication
        # write: [N*W]*[W*1] -> [N*1]
        # read: [N*W]*[W*R] -> [N,R]
        sim = tf.matmul( norm_mem, norm_key, transpose_b = True )
        return tf.nn.softmax( sim * strength, 1 )  # [N,1] or [N,R]
    
    def allocation_weighting(self, usage):

        bs = tf.shape( usage )[0]

        """
        Memory needs to be freed up and allocated in a differentiable way.
        The usage vector shows how much each memory row is used.
        Unused rows can be written to. Usage of a row increases if
        we write to it and can decrease if we read from it, depending on the free gates.
        Allocation weights are then derived from the usage vector.
        Returns
        -------
        Allocation weights for each row in the memory.
        """
        # sort usage vector in ascending order and keep original indices of sorted usage vector
        sorted_usage, free_list = tf.nn.top_k( -1 * tf.transpose( usage, perm = [ 0, 2, 1 ] ), k = self.N )
        sorted_usage *= -1
        cumprod = tf.math.cumprod( sorted_usage, axis = -1, exclusive = True )
        unorder = ( 1 - sorted_usage ) * cumprod

        W_alloc = tf.zeros( [ bs, self.N ] )
        I = tf.tile( tf.constant( np.identity( self.N, dtype = np.float32 ) )[tf.newaxis,...], [ bs, 1, 1 ] )

        # for each usage vec
        for pos, idx in enumerate( tf.unstack( free_list[:,0,:], axis = -1 ) ):
            # flatten
            
            # como tem dimensao de batch, é nescessario usar onehot para filtrar a linha
            # m = tf.squeeze( tf.slice( I, [ idx , 0 ], [ 1, -1 ] ), axis = 1 )
            one_hot = tf.one_hot( idx, tf.shape( free_list )[-1] )[:,:,tf.newaxis]
            m = tf.reduce_sum( I * one_hot, axis = 1 )
            
            # add to allocation weight matrix
            W_alloc += m * unorder[ :, 0, pos ][:,tf.newaxis]

        # return the allocation weighting for each row in memory
        # return tf.reshape( W_alloc, [ bs, self.N, 1 ] )
        return W_alloc[:,:,tf.newaxis]
    
    def controller(self, x):
        x1 = tf.keras.activations.gelu( self.fc1( x ) )
        x2 = self.fc2( x1 )
        return x2
    
    def partition_interface(self, interface):

        bs = tf.shape( interface )[0]

        """
        Partition the interface vector in the read and write keys and strengths,
        the free, allocation and write gates, read modes and erase and write vectors.
        """
        
        # convert interface vector into a set of read write vectors
        # partition = tf.constant( [ [0] * ( self.R * self.W ) + [1] * self.R +
        #                            [2] * self.W + [3] + [4] * self.W + [5] * self.W +
        #                            [6] * self.R + [7] + [8] + [9] * (self.R * 3 ) ],
        #                         dtype = tf.int32 )

        # ( k_read, b_read, k_write, b_write, erase, write_v, free_gates, alloc_gate,
        #  write_gate, read_modes ) = tf.dynamic_partition( interface[0], partition, 10 )

        partition = [ ( self.R * self.W ), self.R, self.W, 1, self.W, self.W, self.R, 1, 1, ( self.R * 3 ) ]
        ( k_read, b_read, k_write, b_write, erase, write_v, free_gates, alloc_gate,
          write_gate, read_modes ) = tf.split( interface, partition, axis = -1 )

        # R read keys and strengths
        k_read = tf.reshape( k_read, [ bs, self.R, self.W ] )       # [R,W]
        b_read = 1 + tf.nn.softplus( tf.expand_dims( b_read, 1 ) )  # [1,R]

        # write key, strength, erase and write vectors
        k_write = tf.expand_dims( k_write, 1 )                        # [1,W]
        b_write = 1 + tf.nn.softplus( tf.expand_dims( b_write, 1 ) )  # [1,1]
        erase = tf.nn.sigmoid( tf.expand_dims( erase, 1 ) )           # [1,W]
        write_v = tf.expand_dims( write_v, 1 )                        # [1,W]

        # the degree to which locations at read heads will be freed
        free_gates = tf.nn.sigmoid( tf.expand_dims( free_gates, 1 ) )  # [1,R]

        # the fraction of writing that is being allocated in a new location
        alloc_gate = tf.reshape( tf.nn.sigmoid( alloc_gate ), [bs, 1] )  # 1

        # the amount of information to be written to memory
        write_gate = tf.reshape( tf.nn.sigmoid( write_gate ), [bs, 1] )  # 1

        # softmax distribution over the 3 read modes (forward, content lookup, backward)
        read_modes = tf.reshape( read_modes, [ bs, 3, self.R ] )  # [3,R]
        read_modes = tf.nn.softmax( read_modes, axis = 1 )

        return ( k_read, b_read, k_write, b_write, erase, write_v,
                free_gates, alloc_gate, write_gate, read_modes )

    def write(self, free_gates, alloc_gate, write_gate, k_write, b_write, erase, write_v, usage, W_read, W_write, M):

        bs = tf.shape( W_read )[0]
        
        """ Write to the memory matrix. """
        # memory retention vector represents by how much each location will not be freed by the free gates
        retention = tf.reduce_prod( 1 - free_gates * W_read, axis = -1 )
        retention = tf.reshape( retention, [ bs, self.N, 1 ] )  # [N,1]

        # update usage vector which is used to dynamically allocate memory
        usage = ( usage + W_write - usage * W_write ) * retention

        # compute allocation weights using dynamic memory allocation
        W_alloc = self.allocation_weighting( usage )  # [N,1]
        W_alloc = tf.reshape( W_alloc, [ bs, self.N ] )

        # apply content lookup for the write vector to figure out where to write to
        W_lookup = self.content_lookup( k_write, b_write, M )
        W_lookup = tf.reshape( W_lookup, [ bs, self.N ] )  # [N,1]

        # define our write weights now that we know how much space to allocate for them and where to write to
        W_write = write_gate * ( alloc_gate * W_alloc + ( 1 - alloc_gate ) * W_lookup )

        # update memory matrix: erase memory and write using the write weights and vector
        M = ( ( M * ( 1 - tf.matmul( W_write[:,:,tf.newaxis], erase ) ) ) + tf.matmul( W_write[:,:,tf.newaxis], write_v ) )

        return usage, W_write, M

    def read(self, k_read, b_read, read_modes, W_write, L, W_precedence, W_read, M):
        
        bs = tf.shape( W_read )[0]

        """ Read from the memory matrix. """
        # update memory link matrix used later for the forward and backward read modes
        W_write_cast = tf.matmul( W_write[:,:,tf.newaxis], tf.ones( [ bs, 1, self.N ] ) )  # [bs,N,N]        
        L = ( ( 1 - W_write_cast - tf.transpose( W_write_cast, perm=[0,2,1] ) ) * L + tf.matmul( W_write[:,:,tf.newaxis], W_precedence, transpose_b = True ) )  # [N,N]
        L *= ( tf.ones( [ bs, self.N, self.N ] ) - tf.tile( tf.constant( np.identity( self.N, dtype = np.float32 ) )[tf.newaxis,...], [ bs, 1, 1 ] ) )

        # update precedence vector which determines degree to which a memory row was written to at t-1
        W_precedence = ( ( 1 - tf.reduce_sum( W_write, axis = 1 ) )[:,tf.newaxis] * W_precedence[:,:,0] + W_write )[:,:,tf.newaxis]

        # apply content lookup for the read vector(s) to figure out where to read from
        W_lookup = self.content_lookup( k_read, b_read, M )
        #W_lookup = tf.reshape( W_lookup, [ bs, self.N, self.R ] )  # [N,R]

        # compute forward and backward read weights using the link matrix
        # forward weights recall information written in sequence and backward weights in reverse
        W_fwd = tf.matmul( L, W_read )  # [N,N]*[N,R] -> [N,R]
        W_bwd = tf.matmul( L, W_read, transpose_a = True)  # [N,R]

        # 3 modes: forward, backward and content lookup
        fwd_mode    = read_modes[:,2][:,tf.newaxis,:] * W_fwd
        lookup_mode = read_modes[:,1][:,tf.newaxis,:] * W_lookup
        bwd_mode    = read_modes[:,0][:,tf.newaxis,:] * W_bwd

        # read weights = backward + content lookup + forward mode weights
        W_read = bwd_mode + lookup_mode + fwd_mode  # [N,R]

        # create read vectors by applying read weights to memory matrix
        read_v = tf.transpose( tf.matmul( M, W_read, transpose_a = True ), perm = [ 0, 2, 1 ] )  # ([W,N]*[N,R])^T -> [R,W]

        return L, W_precedence, W_read, read_v

    def step_write(self, x, usage, W_read, W_write, M):

        interface = tf.matmul( x, self.W_interface )  # [1,Y+I] * [Y+I,I] -> [1,I]

        # partition the interface vector
        ( _, _, k_write, b_write, erase, write_v, free_gates, alloc_gate, write_gate, _ ) = self.partition_interface( interface )

        # write to memory
        n_usage, n_W_write, n_M = self.write( free_gates, alloc_gate, write_gate, k_write, b_write, erase, write_v, usage, W_read, W_write, M )

        return n_M, n_W_write, n_usage

    def step_read(self, x, W_write, L, W_precedence, W_read, M):

        interface = tf.matmul( x, self.W_interface )  # [1,Y+I] * [Y+I,I] -> [1,I]

        # partition the interface vector
        ( k_read, b_read, _, _, _, _, _, _, _, read_modes ) = self.partition_interface( interface )

        n_L, n_W_precedence, n_W_read, read_v = self.read( k_read, b_read, read_modes, W_write, L, W_precedence, W_read, M )

        return n_L, n_W_precedence, n_W_read, read_v

    def step(self, x_read, x_write, M, usage, L, W_precedence, W_read, W_write):
        
        bs = tf.shape(x_read)[0]

        """
        Update the controller, compute the output and interface vectors,
        write to and read from memory and compute the output.
        """
        # update controller
        if self.external_controller:
            controller_read_out = x_read
            controller_write_out = x_write
        else:
            controller_read_out = self.controller( x_read )
            controller_write_out = self.controller( x_write )

        # write to memory
        n_M, n_W_write, n_usage = self.step_write( controller_write_out, usage, W_read, W_write, M )

        # read from memory
        n_L, n_W_precedence, n_W_read, read_v = self.step_read( controller_read_out, n_W_write, L, W_precedence, W_read, n_M )
        
        # flatten read vectors and multiply them with W matrix before adding to controller output
        read_v_out = tf.matmul( tf.reshape( read_v, [ bs, self.R * self.W ] ), self.W_read_out )  # [1,RW]*[RW,Y] -> [1,Y]

        if not self.external_controller:
            
            # compute output vectors
            output_v = tf.matmul( controller_read_out, self.W_output )  # [1,Y+I] * [Y+I,Y] -> [1,Y]

            # compute output
            y = output_v + read_v_out
            
            return y, n_M, n_usage, n_L, n_W_precedence, n_W_read, n_W_write[:,:,tf.newaxis]
        
        return read_v_out, n_M, n_usage, n_L, n_W_precedence, n_W_read, n_W_write[:,:,tf.newaxis]

    def call(self, inputs, states):

        x_read, x_write = tf.unstack( inputs, axis = 1 )

        M, usage, L, W_precedence, W_read, W_write = states

        ( y_seq, n_M, n_usage, n_L, n_W_precedence, n_W_read, 
          n_W_write ) = self.step( x_read, x_write, M, usage, L, W_precedence, W_read, W_write )

        return y_seq, ( n_M, n_usage, n_L, n_W_precedence, n_W_read, n_W_write )


class dnc_v2(Layer):

    def __init__(self, output_dim, memory_shape=(100, 20), n_read=3, leaning_rate=0, memory_decay_steps=0):

        super(dnc_v2, self).__init__()

        # define output data size
        self.output_dim = output_dim  # Y
        
        # define size of memory matrix
        self.N, self.W = memory_shape  # N, W
        
        self.alpha = lambda x: leaning_rate * ( 1. / ( ( x / memory_decay_steps ) + 1 ) )
        
        # define number of read heads
        self.R = n_read  # R

        # size of output vector from controller that defines interactions with memory matrix:
        # R read keys + R read strengths + write key + write strength + erase vector +
        # write vector + R free gates + allocation gate + write gate + R read modes
        self.interface_dim = self.R * self.W + 3 * self.W + 5 * self.R + 3  # I

        # neural net output = output of controller + interface vector with memory
        self.controller_dim = self.output_dim + self.interface_dim  # Y+I

        # controller variables
        # initialize controller hidden state
        # self.fc1 = dense( self.controller_dim, 'fc1_dnc', kernel_initializer = tf.keras.initializers.RandomNormal() )
        # self.fc2 = dense( self.controller_dim, 'fc2_dnc', kernel_initializer = tf.keras.initializers.RandomNormal() )
        self.fc1 = GRUCell( self.controller_dim, name = 'fc1_dnc', activation = gelu )
        
        self.state_size = [ 
            tf.TensorShape( [ self.N, self.W ] ),
            tf.TensorShape( [ self.N, 1 ] ),
            tf.TensorShape( [ self.N, self.N ] ),
            tf.TensorShape( [ self.N, 1 ] ),
            tf.TensorShape( [ self.N, n_read ] ),
            tf.TensorShape( [ self.N, 1 ] ),
            tf.TensorShape( [ 2, self.controller_dim ] ),
         ]


    def build(self, input_shapes):

        # define and initialize weights for controller output and interface vectors
        
        self.W_interface = self.add_weight( name = 'dnc_interface_weights', shape = [ self.controller_dim, self.interface_dim ], 
                                            initializer = tf.keras.initializers.truncated_normal( stddev = 0.1 ) ) # [Y+I,I]

        self.W_read_out = self.add_weight( name = 'dnc_read_vector_weights', shape = [ self.R * self.W, self.output_dim ], 
                                           initializer = tf.keras.initializers.truncated_normal( stddev = 0.1 ) ) # [R*W,Y]

        # self.lr_controller = self.add_weight( 'dnc_learnig_rate', [ self.controller_dim, 8 ] , initializer = tf.keras.initializers.random_normal() )
        
    def reset(self, bs):

        # initialize memory matrix with zeros
        M = tf.zeros( [ bs, self.N, self.W ] )  # [N,W]

        # usage vector records which locations in the memory are used and which are free
        usage = tf.fill( [ bs, self.N, 1 ], 1e-6 )  # [N,1]

        # temporal link matrix L[i,j] records to which degree location i was written to after j
        L = tf.zeros( [ bs, self.N, self.N ] )  # [N,N]

        # precedence vector determines degree to which a memory row was written to at t-1
        W_precedence = tf.zeros( [ bs, self.N, 1 ] )  # [N,1]

        # initialize R read weights and vectors and write weights
        W_read = tf.fill( [ bs, self.N, self.R ], 1e-6 )  # [N,R]
        W_write = tf.fill( [ bs, self.N, 1 ], 1e-6 )  # [N,1]

        return M, usage, L, W_precedence, W_read, W_write, tf.zeros( [ bs, 2, self.controller_dim ] )
    
    def content_lookup(self, key, strength, M):
        """
        Attention mechanism: content based addressing to read from and write to the memory.
        Params
        ------
        key
            Key vector emitted by the controller and used to calculate row-by-row
            cosine similarity with the memory matrix.
        strength
            Strength scalar attached to each key vector (1x1 or 1xR).
        Returns
        -------
        Similarity measure for each row in the memory used by the read heads for associative
        recall or by the write head to modify a vector in memory.
        """
        # The l2 norm applied to each key and each row in the memory matrix
        norm_mem = tf.nn.l2_normalize( M, -1 )  # [N,W]
        norm_key = tf.nn.l2_normalize( key, -1 )     # [1,W] for write or [R,W] for read

        # get similarity measure between both vectors, transpose before multiplication
        # write: [N*W]*[W*1] -> [N*1]
        # read: [N*W]*[W*R] -> [N,R]
        sim = tf.matmul( norm_mem, norm_key, transpose_b = True )
        return tf.nn.softmax( sim * strength, 1 )  # [N,1] or [N,R]
    
    def allocation_weighting(self, usage):

        bs = tf.shape( usage )[0]

        """
        Memory needs to be freed up and allocated in a differentiable way.
        The usage vector shows how much each memory row is used.
        Unused rows can be written to. Usage of a row increases if
        we write to it and can decrease if we read from it, depending on the free gates.
        Allocation weights are then derived from the usage vector.
        Returns
        -------
        Allocation weights for each row in the memory.
        """
        # sort usage vector in ascending order and keep original indices of sorted usage vector
        sorted_usage, free_list = tf.nn.top_k( -1 * tf.transpose( usage, perm = [ 0, 2, 1 ] ), k = self.N )
        sorted_usage *= -1
        cumprod = tf.math.cumprod( sorted_usage, axis = -1, exclusive = True )
        unorder = ( 1 - sorted_usage ) * cumprod

        W_alloc = tf.zeros( [ bs, self.N ] )
        I = tf.tile( tf.constant( np.identity( self.N, dtype = np.float32 ) )[tf.newaxis,...], [ bs, 1, 1 ] )

        # for each usage vec
        for pos, idx in enumerate( tf.unstack( free_list[:,0,:], axis = -1 ) ):
            # flatten
            
            # como tem dimensao de batch, é nescessario usar onehot para filtrar a linha
            # m = tf.squeeze( tf.slice( I, [ idx , 0 ], [ 1, -1 ] ), axis = 1 )
            one_hot = tf.one_hot( idx, tf.shape( free_list )[-1] )[:,:,tf.newaxis]
            m = tf.reduce_sum( I * one_hot, axis = 1 )
            
            # add to allocation weight matrix
            W_alloc += m * unorder[ :, 0, pos ][:,tf.newaxis]

        # return the allocation weighting for each row in memory
        # return tf.reshape( W_alloc, [ bs, self.N, 1 ] )
        return W_alloc[:,:,tf.newaxis]
    
    def controller(self, x, h):
        x1, h1 = self.fc1( x, h )
        return x1, h1
    
    def partition_interface(self, interface):

        bs = tf.shape( interface )[0]

        """
        Partition the interface vector in the read and write keys and strengths,
        the free, allocation and write gates, read modes and erase and write vectors.
        """
        
        # convert interface vector into a set of read write vectors
        # partition = tf.constant( [ [0] * ( self.R * self.W ) + [1] * self.R +
        #                            [2] * self.W + [3] + [4] * self.W + [5] * self.W +
        #                            [6] * self.R + [7] + [8] + [9] * (self.R * 3 ) ],
        #                         dtype = tf.int32 )

        # ( k_read, b_read, k_write, b_write, erase, write_v, free_gates, alloc_gate,
        #  write_gate, read_modes ) = tf.dynamic_partition( interface[0], partition, 10 )

        partition = [ ( self.R * self.W ), self.R, self.W, 1, self.W, self.W, self.R, 1, 1, ( self.R * 3 ) ]
        ( k_read, b_read, k_write, b_write, erase, write_v, free_gates, alloc_gate,
          write_gate, read_modes ) = tf.split( interface, partition, axis = -1 )

        # R read keys and strengths
        k_read = tf.reshape( k_read, [ bs, self.R, self.W ] )       # [R,W]
        b_read = 1 + tf.nn.softplus( tf.expand_dims( b_read, 1 ) )  # [1,R]

        # write key, strength, erase and write vectors
        k_write = tf.expand_dims( k_write, 1 )                        # [1,W]
        b_write = 1 + tf.nn.softplus( tf.expand_dims( b_write, 1 ) )  # [1,1]
        erase = tf.nn.sigmoid( tf.expand_dims( erase, 1 ) )           # [1,W]
        write_v = tf.expand_dims( write_v, 1 )                        # [1,W]

        # the degree to which locations at read heads will be freed
        free_gates = tf.nn.sigmoid( tf.expand_dims( free_gates, 1 ) )  # [1,R]

        # the fraction of writing that is being allocated in a new location
        alloc_gate = tf.reshape( tf.nn.sigmoid( alloc_gate ), [bs, 1] )  # 1

        # the amount of information to be written to memory
        write_gate = tf.reshape( tf.nn.sigmoid( write_gate ), [bs, 1] )  # 1

        # softmax distribution over the 3 read modes (forward, content lookup, backward)
        read_modes = tf.reshape( read_modes, [ bs, 3, self.R ] )  # [3,R]
        read_modes = tf.nn.softmax( read_modes, axis = 1 )

        return ( k_read, b_read, k_write, b_write, erase, write_v,
                free_gates, alloc_gate, write_gate, read_modes )

    def write(self, free_gates, alloc_gate, write_gate, k_write, b_write, erase, write_v, usage, W_read, W_write, M, lr):

        bs = tf.shape( W_read )[0]
        
        """ Write to the memory matrix. """
        # memory retention vector represents by how much each location will not be freed by the free gates
        retention = tf.reduce_prod( 1 - free_gates * W_read, axis = -1 )
        retention = tf.reshape( retention, [ bs, self.N, 1 ] )  # [N,1]

        # update usage vector which is used to dynamically allocate memory
        usage = ( usage + W_write - usage * W_write ) * retention

        # compute allocation weights using dynamic memory allocation
        W_alloc = self.allocation_weighting( usage )  # [N,1]
        W_alloc = tf.reshape( W_alloc, [ bs, self.N ] )

        # apply content lookup for the write vector to figure out where to write to
        W_lookup = self.content_lookup( k_write, b_write, M )
        W_lookup = tf.reshape( W_lookup, [ bs, self.N ] )  # [N,1]

        # define our write weights now that we know how much space to allocate for them and where to write to
        W_write = write_gate * ( alloc_gate * W_alloc + ( 1 - alloc_gate ) * W_lookup )
        ww = W_write * lr

        # update memory matrix: erase memory and write using the write weights and vector
        m_erase = M * ( ( 1 - tf.matmul( W_write[:,:,tf.newaxis], erase ) ) ) 
        m_add = tf.matmul( W_write[:,:,tf.newaxis], write_v )

        m_erase = M * ( ( 1 - tf.matmul( ww[:,:,tf.newaxis], erase ) ) ) 
        m_add = tf.matmul( ww[:,:,tf.newaxis], write_v )

        # M = ( ( 1 - lr[:,:,tf.newaxis] ) * M ) + ( lr[:,:,tf.newaxis] * ( m_erase + m_add ) )
        M = m_erase + m_add

        return usage, W_write, M

    def read(self, k_read, b_read, read_modes, W_write, L, W_precedence, W_read, M):
        
        bs = tf.shape( W_read )[0]

        """ Read from the memory matrix. """
        # update memory link matrix used later for the forward and backward read modes
        W_write_cast = tf.matmul( W_write[:,:,tf.newaxis], tf.ones( [ bs, 1, self.N ] ) )  # [bs,N,N]
        L = ( ( 1 - W_write_cast - tf.transpose( W_write_cast, perm=[0,2,1] ) ) * L + tf.matmul( W_write[:,:,tf.newaxis], W_precedence, transpose_b = True ) )  # [N,N]
        L *= ( tf.ones( [ bs, self.N, self.N ] ) - tf.tile( tf.constant( np.identity( self.N, dtype = np.float32 ) )[tf.newaxis,...], [ bs, 1, 1 ] ) )

        # update precedence vector which determines degree to which a memory row was written to at t-1
        W_precedence = ( ( 1 - tf.reduce_sum( W_write, axis = 1 ) )[:,tf.newaxis] * W_precedence[:,:,0] + W_write )[:,:,tf.newaxis]

        # apply content lookup for the read vector(s) to figure out where to read from
        W_lookup = self.content_lookup( k_read, b_read, M )
        #W_lookup = tf.reshape( W_lookup, [ bs, self.N, self.R ] )  # [N,R]

        # compute forward and backward read weights using the link matrix
        # forward weights recall information written in sequence and backward weights in reverse
        W_fwd = tf.matmul( L, W_read )  # [N,N]*[N,R] -> [N,R]
        W_bwd = tf.matmul( L, W_read, transpose_a = True)  # [N,R]

        # 3 modes: forward, backward and content lookup
        fwd_mode    = read_modes[:,2][:,tf.newaxis,:] * W_fwd
        lookup_mode = read_modes[:,1][:,tf.newaxis,:] * W_lookup
        bwd_mode    = read_modes[:,0][:,tf.newaxis,:] * W_bwd

        # read weights = backward + content lookup + forward mode weights
        W_read = bwd_mode + lookup_mode + fwd_mode  # [N,R]

        # create read vectors by applying read weights to memory matrix
        read_v = tf.transpose( tf.matmul( M, W_read, transpose_a = True ), perm = [ 0, 2, 1 ] )  # ([W,N]*[N,R])^T -> [R,W]

        return L, W_precedence, W_read, read_v

    def step_write(self, x, usage, W_read, W_write, M, lr):

        interface = tf.matmul( x, self.W_interface )  # [1,Y+I] * [Y+I,I] -> [1,I]

        # partition the interface vector
        ( _, _, k_write, b_write, erase, write_v, free_gates, alloc_gate, write_gate, _ ) = self.partition_interface( interface )

        # write to memory
        n_usage, n_W_write, n_M = self.write( free_gates, alloc_gate, write_gate, k_write, b_write, erase, write_v, usage, W_read, W_write, M, lr )

        return n_M, n_W_write, n_usage

    def step_read(self, x, W_write, L, W_precedence, W_read, M):

        interface = tf.matmul( x, self.W_interface )  # [1,Y+I] * [Y+I,I] -> [1,I]

        # partition the interface vector
        ( k_read, b_read, _, _, _, _, _, _, _, read_modes ) = self.partition_interface( interface )

        n_L, n_W_precedence, n_W_read, read_v = self.read( k_read, b_read, read_modes, W_write, L, W_precedence, W_read, M )

        return n_L, n_W_precedence, n_W_read, read_v

    def step(self, x_read, x_write, M, usage, L, W_precedence, W_read, W_write, h_controller, lr):
        
        bs = tf.shape(x_read)[0]

        """
        Update the controller, compute the output and interface vectors,
        write to and read from memory and compute the output.
        """
        hc1, hc2 = tf.split( h_controller, 2, axis = 1 )
        controller_read_out, c1 = self.controller( x_read, hc1[:,0,...] )
        controller_write_out, c2 = self.controller( x_write, hc2[:,0,...] )

        hc = tf.stack( ( c1, c2 ), axis = 1 )

        # write to memory
        n_M, n_W_write, n_usage = self.step_write( controller_write_out, usage, W_read, W_write, M, lr )

        # read from memory
        n_L, n_W_precedence, n_W_read, read_v = self.step_read( controller_read_out, lr * n_W_write, L, W_precedence, W_read, n_M )
        
        # flatten read vectors and multiply them with W matrix before adding to controller output
        read_v_out = tf.matmul( tf.reshape( read_v, [ bs, self.R * self.W ] ), self.W_read_out )  # [1,RW]*[RW,Y] -> [1,Y]
        
        return read_v_out, n_M, n_usage, n_L, n_W_precedence, n_W_read, n_W_write[:,:,tf.newaxis], hc

    def call(self, inputs, states):

        x_read, x_write = tf.unstack( inputs, axis = 1 )

        M, usage, L, W_precedence, W_read, W_write, h_controller = states

        # lr controler
        # lr = tf.cast( self.alpha( step ), tf.float32 )
        lr = tf.constant( 1.0, tf.float32 )

        ( y_seq, n_M, n_usage, n_L, n_W_precedence, n_W_read, 
          n_W_write, h_controller ) = self.step( x_read, x_write, M, usage, L, W_precedence, W_read, W_write, h_controller, lr )

        # h_controller = tf.concat( [ n_usage[:,:,0], n_W_write[:,:,0] ], axis = -1 )

        return y_seq, ( n_M, n_usage, n_L, n_W_precedence, n_W_read, n_W_write, h_controller )


class tdnc(Layer):

    def __init__(self, output_dim, max_len_seq, memory_shape=(100, 20), num_blocks=1, n_read=3, n_att_heads=4):

        super(tdnc, self).__init__()

        self.output_dim = output_dim

        self.transformer = transformer_layer( output_dim, n_att_heads, num_blocks, max_len_seq, embeding_inputs = True )
        self.memory = dnc( output_dim, memory_shape, n_read, external_controller = True )
        self.rnn = RNN( self.memory, return_sequences = True, return_state = True, unroll = True )
        self.controller = dense( self.memory.controller_dim )

    def reset(self, bs):
        return self.memory.reset( bs )

    def call(self, x_read, x_write, past, states):
        
        bs = tf.shape( x_read )[0]
        
        # Controller
                        
        ## temporal encoder
        x = tf.concat( [ x_read, x_write ], axis = 0 )
        t, presents, msks = self.transformer( x, past )
        
        ## encoder
        xc = tf.keras.activations.tanh( self.controller( t ) )
        t_read, t_write = tf.split( xc, [ bs, bs ], axis = 0 )
        hrnn = tf.stack( [ t_read, t_write ], axis = 2 )
        
        # Dnc
        y, *p_state = self.rnn( hrnn, states )

        out = y + x_read[:,:,:self.output_dim]

        return out, [ presents ] + p_state, msks


class tdnc_v2(Layer):

    def __init__(self, output_dim, memory_shape=(100, 20), n_read=3, leaning_rate=1, memory_decay_steps=10000):

        super(tdnc_v2, self).__init__()

        self.output_dim = output_dim

        self.memory = dnc_v2( output_dim, memory_shape, n_read, leaning_rate = leaning_rate, memory_decay_steps = memory_decay_steps )
        self.rnn = RNN( self.memory, return_sequences = True, return_state = True, unroll = True )
        self.fc1 = dense( 128, 'fc1_memory', activation = gelu )

    def reset(self, bs):
        m, u, l, wp, wr, ww, hc = self.memory.reset( bs )
        return ( m, u, l, wp, wr, ww, hc )
        
    def call(self, x_read, x_write, memory_states, step):
        
        bs = tf.shape( x_read )[0]

        ## encoder
        h = tf.concat( [ x_read, x_write ], axis = 0 )
        h_read, h_write = tf.split( self.fc1( h ), [ bs, bs ], axis = 0 ) 

        # stack na 2 pra nao atravessar a dimensao da sequencia
        hrnn = tf.stack( [ h_read, h_write ], axis = 2 )
        
        # Dnc
        y, *p_state = self.rnn( hrnn, list( memory_states ) )

        out = y

        return out, p_state


class ReparameterizeVAE(Layer):
    
    def call(self, inputs):

        dtype = prec.global_policy().compute_dtype

        Z_mu, Z_logvar = inputs
        epsilon = tf.random.normal( tf.shape( Z_mu ), dtype = dtype )
        sigma = tf.math.exp( 0.5 * Z_logvar )
        return Z_mu + sigma * epsilon

        
class Adam(tf.keras.optimizers.Adam):
    
    def __init__(self, learning_rate, amsgrad = False, beta_1 = tf.Variable(0.9), beta_2 = tf.Variable(0.999), epsilon = tf.Variable(1e-7), decay = tf.Variable(0.0)):
        super(Adam, self).__init__( learning_rate, beta_1 = beta_1, beta_2 = beta_2, epsilon = epsilon, amsgrad = amsgrad )
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