import tensorflow as tf
from tensorflow.keras.layers import Layer, LayerNormalization, Dense, RNN, Dropout, BatchNormalization, Conv2DTranspose, LayerNormalization
from tensorflow.keras.activations import gelu, sigmoid, tanh, elu, relu
from tensorflow.keras.initializers import he_normal, orthogonal, random_normal, truncated_normal
import numpy as np
from tensorflow.keras.mixed_precision import LossScaleOptimizer
from functools import partial
from ann_utils import nalu_transform, tdnc_v2, dense, transformer_layer, nalu_transform_w, dense_w, norm
from math import ceil

class StateUnderstanding(Layer):

    def __init__(self, state_dim, f1, f2, train=False):

        super(StateUnderstanding, self).__init__()

        self.to_train = train
              
        self.fc1 = nalu_transform( f1, activation = elu, kernel_initializer = truncated_normal(), name = 'su1' )
        self.fc2 = nalu_transform( f2, activation = elu, kernel_initializer = truncated_normal(), name = 'su2' )

        self.dp1 = tf.keras.layers.Dropout( .25 )
        self.dp2 = tf.keras.layers.Dropout( .25 )

    def feature_extract(self, x, eval=True):
        
        x = x
        
        # Process non-linear tranformation and linear arithmetic
        x1 = self.fc1( x )
        x1 = self.dp1( x1 )
        if not eval: x1 = self.dp1( x1 )
        x2 = self.fc2( x1 )
        x2 = self.dp2( x2 )
        if not eval: x2 = self.dp2( x2 )
        
        return x2

    def call(self, state, eval=True):

        x = self.feature_extract( state, eval )
        return x


class StateUnderstanding_v2(Layer):

    def __init__(self, state_dim, f1, f2, train=False):

        super(StateUnderstanding_v2, self).__init__()

        self.to_train = train

        self.fc1e = nalu_transform( f1, activation = gelu, kernel_initializer = he_normal(), initializer = he_normal() )
        self.fc2e = nalu_transform( f1, activation = gelu, kernel_initializer = he_normal(), initializer = he_normal() )
        
        self.fc1 = nalu_transform( f1, activation = gelu, kernel_initializer = he_normal(), initializer = he_normal() )
        self.fc2 = nalu_transform( f2, activation = gelu, kernel_initializer = he_normal(), initializer = he_normal() )

        self.bn = BatchNormalization( )

        # especifico para cada entrada
        i_size = state_dim
        t_size = f1
        dvs = [ 0 ]
        w = np.zeros( [ state_dim, f1 ] )
        w1e = np.zeros( [ f1, f1 ] )
        for i in range( state_dim ):
            dv = t_size / i_size
            v = ceil( dv ) if dv - int( dv ) > .5 else int( dv )
            w[ i, np.sum( dvs ) : np.sum( dvs ) + v ] = 1            
            for j in range( np.sum( dvs ), np.sum( dvs ) + v ):
                w1e[ j, np.sum( dvs ) : np.sum( dvs ) + v ] = 1
            dvs.append( v )
            i_size -= 1
            t_size -= v
       
        self.w = tf.constant( w, tf.float32 )
        self.w1e = tf.constant( w1e, tf.float32 )

        self.dp1 = tf.keras.layers.Dropout( .25 )
        self.dp2 = tf.keras.layers.Dropout( .25 )

    def feature_extract(self, x, eval=True):

        # Process non-linear tranformation and linear arithmetic
        
        xa = tf.matmul( x, self.w )
        h = tf.stack( [ xa, xa ], axis = 1 )

        xa1 = self.fc1e( h, self.w1e )
        xa2 = self.fc2e( xa1, self.w1e )
        
        x1 = self.fc1( xa2 )
        if not eval: x1 = self.dp1( x1 )
        x2 = self.fc2( x1 )
        if not eval: x2 = self.dp2( x2 )

        return x2

    def call(self, state, pstate, eval=True):

        h = self.feature_extract( state, eval )
        if pstate is None: return h
        hprev = self.feature_extract( pstate, eval )
        return h, hprev


class StateUnderstandingMultiEnv(Layer):

    def __init__(self, envs, f1, f2):

        super(StateUnderstandingMultiEnv, self).__init__()

        self.fc1e = nalu_transform( f1, activation = gelu, kernel_initializer = he_normal(), initializer = he_normal() )
        self.fc2e = nalu_transform( f1, activation = gelu, kernel_initializer = he_normal(), initializer = he_normal() )
        
        self.fc1 = nalu_transform( f1, activation = gelu, kernel_initializer = he_normal(), initializer = he_normal() )
        self.fc2 = nalu_transform( f2, activation = gelu, kernel_initializer = he_normal(), initializer = he_normal() )

        self.bn = BatchNormalization( )
        
        # especifico para cada entrada
        self.envs = {}
        for e in envs:
            self.create_weigth_mask( e['name'], e['state_dim'], f1 )
        
        self.dp1 = tf.keras.layers.Dropout( .25 )
        self.dp2 = tf.keras.layers.Dropout( .25 )

    def create_weigth_mask(self, name, size, f1):

        i_size = size
        t_size = f1
        dvs = [ 0 ]
        w = np.zeros( [ size, f1 ] )
        we = np.zeros( [ f1, f1 ] )
        for i in range( size ):
            dv = t_size / i_size
            v = ceil( dv ) if dv - int( dv ) > .5 else int( dv )
            w[ i, np.sum( dvs ) : np.sum( dvs ) + v ] = 1            
            for j in range( np.sum( dvs ), np.sum( dvs ) + v ):
                we[ j, np.sum( dvs ) : np.sum( dvs ) + v ] = 1
            dvs.append( v )
            i_size -= 1
            t_size -= v
       
        w = tf.constant( w, tf.float32 )
        we = tf.constant( we, tf.float32 )

        self.envs[name] = { 'w': w, 'we': we }

    def feature_extract(self, x, env, eval=True):

        # Process non-linear tranformation and linear arithmetic
        
        # x = self.bn( x, training = not eval )

        xa = tf.matmul( x, self.envs[env]['w'] )
        h = tf.stack( [ xa, xa ], axis = 1 )

        xa1 = self.fc1e( h, self.envs[env]['we'] )
        xa2 = self.fc2e( xa1, self.envs[env]['we'] )
        
        x1 = self.fc1( xa2 )
        if not eval: x1 = self.dp1( x1 )
        x2 = self.fc2( x1 )
        if not eval: x2 = self.dp2( x2 )

        return x2

    def call(self, state, pstate, env, eval=True):

        h = self.feature_extract( state, env, eval )
        if pstate is None: return h
        hprev = self.feature_extract( pstate, env, eval )
        return h, hprev


class StateUnderstandingMultiEnv_v2(Layer):

    def __init__(self, envs, f1, f2, maxlen):

        super(StateUnderstandingMultiEnv_v2, self).__init__()

        self.fc1e = nalu_transform( f1, activation = gelu, kernel_initializer = he_normal(), initializer = he_normal(), name = 'su_e1' )
        self.fc2e = nalu_transform( f1, activation = gelu, kernel_initializer = he_normal(), initializer = he_normal(), name = 'su_e2' )
        
        self.fc1 = nalu_transform( f1, activation = gelu, kernel_initializer = he_normal(), initializer = he_normal(), name = 'su_e3' )
        
        self.fc2 = nalu_transform( f2, activation = gelu, kernel_initializer = he_normal(), initializer = he_normal(), name = 'su_non_recurrent' )

        self.fc3 = nalu_transform( f2, activation = gelu, kernel_initializer = he_normal(), initializer = he_normal(), name = 'su_recurrent_e' )
        self.rnn = transformer_layer( f2, 4, 2, maxlen, embeding_inputs = True, name = 'su_recurrent_t' )

        self.bn = BatchNormalization( )
        
        # especifico para cada entrada
        self.envs = {}
        for e in envs:
            self.create_weigth_mask( e['id'], e['state_dim'], f1, e['recurrent'] )
        
        self.dp1 = tf.keras.layers.Dropout( .25 )
        self.dp2 = tf.keras.layers.Dropout( .25 )

    def create_weigth_mask(self, name, size, f1, recurrent):

        i_size = size
        t_size = f1
        dvs = [ 0 ]
        w = np.zeros( [ size, f1 ] )
        we = np.zeros( [ f1, f1 ] )
        for i in range( size ):
            dv = t_size / i_size
            v = ceil( dv ) if dv - int( dv ) > .5 else int( dv )
            w[ i, np.sum( dvs ) : np.sum( dvs ) + v ] = 1            
            for j in range( np.sum( dvs ), np.sum( dvs ) + v ):
                we[ j, np.sum( dvs ) : np.sum( dvs ) + v ] = 1
            dvs.append( v )
            i_size -= 1
            t_size -= v
       
        w = tf.constant( w, tf.float32 )
        we = tf.constant( we, tf.float32 )

        self.envs[name] = { 'w': w, 'we': we, 'recurrent': tf.constant( recurrent, tf.float32 ) }

    def feature_extract(self, x, env, past=None, eval=True):

        # Process non-linear tranformation and linear arithmetic
        
        # encoding phase
        # x = self.bn( x, training = not eval )

        xa = tf.matmul( x, self.envs[env]['w'] )
        h = tf.stack( [ xa, xa ], axis = 1 )

        xa1 = self.fc1e( h, self.envs[env]['we'] )
        xa2 = self.fc2e( xa1, self.envs[env]['we'] )        

        x1 = self.fc1( xa2 )
        h1x1, h2x1 = tf.unstack( x1, axis = 1 )
        if not eval:
            h1x1 = self.dp1( h1x1 )
            h2x1 = self.dp1( h2x1 )

        # recurrent phase
        rth, present, mask = self.rnn( h2x1, past )
        rt = tf.stack( [ h1x1, rth ], axis = 1 )
        rt = self.fc3( rt )
        rt1, rt2 = tf.unstack( rt, axis = 1 )
        if not eval:
            rt1 = self.dp2( rt1 )
            rt2 = self.dp2( rt2 )

        # non recurrent phase
        x2 = self.fc2( x1 )
        h1x2, h2x2 = tf.unstack( x2, axis = 1 )
        if not eval:
            h1x2 = self.dp2( h1x2 )
            h2x2 = self.dp2( h2x2 )

        # recurrent vs non recurrent gate
        h = tf.stack( 
            [
                ( self.envs[env]['recurrent'] * rt1 ) + ( ( 1 - self.envs[env]['recurrent'] ) * h1x2 ),
                ( self.envs[env]['recurrent'] * rt2 ) + ( ( 1 - self.envs[env]['recurrent'] ) * h2x2 ),
            ], axis = 1
         )

        return h, present, mask

    def call(self, state, pstate, env, past=None, eval=True):

        h, present, mask = self.feature_extract( state, env, past, eval )
        
        if pstate is None: 
            return h, present, mask

        hprev, presentprev, _ = self.feature_extract( pstate, env, past, eval )
        
        return h, hprev, present, presentprev, mask


class StateUnderstandingMultiEnv_v3(Layer):

    def __init__(self, envs, f1, f2, maxlen):

        super(StateUnderstandingMultiEnv_v3, self).__init__()
        
        self.fc1e = dense_w( f1, activation = tanh, name = 'su_e1' )
        self.fc2e = dense_w( f1, activation = tanh, name = 'su_e2' )
        
        self.fc1 = nalu_transform( f1, activation = gelu, kernel_initializer = he_normal(), initializer = truncated_normal(), name = 'su_e3' )
        
        self.fc2 = nalu_transform( f2, activation = gelu, kernel_initializer = he_normal(), initializer = truncated_normal(), name = 'su_non_recurrent' )

        self.fc3 = nalu_transform( f2, activation = gelu, kernel_initializer = he_normal(), initializer = truncated_normal(), name = 'su_recurrent_e' )
        self.rnn = transformer_layer( f2, 4, 4, maxlen, embeding_inputs = True, name = 'su_recurrent_t' )

        self.ln1 = LayerNormalization()
        self.ln2 = LayerNormalization()

        self.mini_net_sizes = [ 3, 5, 7, 9, 11, 15, 19, 25, 35 ]
        self.mini_kn = {}
        for m in self.mini_net_sizes:
            self.mini_kn[ str( m ) ] = self.add_weight( 'kn_{}'.format( m ), [ m, m ], initializer = he_normal(), trainable = True )
        
        # especifico para cada entrada
        self.envs = {}
        for e in envs:
            self.create_weigth_mask( e['id'], e['state_dim'], f1, e['recurrent'] )
        
        self.dp1 = tf.keras.layers.Dropout( .25 )
        self.dp2 = tf.keras.layers.Dropout( .25 )

    def create_weigth_mask(self, name, size, f1, recurrent):

        i_size = size
        t_size = f1
        dvs = [ 0 ]
        w = np.zeros( [ size, f1 ] )
        we = np.zeros( [ f1, f1 ] )
        for i in range( size ):
            dv = t_size / i_size
            v = ceil( dv ) if dv - int( dv ) > .5 else int( dv )
            w[ i, np.sum( dvs ) : np.sum( dvs ) + v ] = 1            
            for j in range( np.sum( dvs ), np.sum( dvs ) + v ):
                we[ j, np.sum( dvs ) : np.sum( dvs ) + v ] = 1
            dvs.append( v )
            i_size -= 1
            t_size -= v
       
        w = tf.constant( w, tf.float32 )
        we = tf.constant( we, tf.float32 )

        mini_net_size = np.max( dvs[1:] )
        selected = np.argmin( np.abs( mini_net_size - np.array( self.mini_net_sizes ) ) )
        sz = self.mini_net_sizes[ selected ]

        self.envs[name] = { 'w': w, 'we': we, 'mini_net': str( sz ), 'sz': ceil( f1 / sz ), 'recurrent': tf.constant( recurrent, tf.float32 ) }
    
    def feature_extract(self, x, env, past=None, eval=True):

        # encoding phase - get a dynamic input size and features, process using mini weight matrix and normalize output        
        # xa = tf.matmul( x, self.envs[env]['w'] )

        # xa1 = self.fc1e( xa, self.envs[env]['sz'], self.mini_kn[self.envs[env]['mini_net']], self.envs[env]['we'] )
        # xa2 = self.fc2e( xa1, self.envs[env]['sz'], self.mini_kn[self.envs[env]['mini_net']], self.envs[env]['we'] )

        h = tf.stack( [ x, x ], axis = 1 )

        x1 = self.fc1( h )
        h1x1, h2x1 = tf.unstack( x1, axis = 1 )
        if not eval:
            h1x1 = self.dp1( h1x1 )
            h2x1 = self.dp1( h2x1 )

        # recurrent phase
        rth, present, mask = self.rnn( h2x1, past, eval )
        rt = tf.stack( [ h1x1, rth ], axis = 1 )
        rt = self.fc3( rt )
        rt1, rt2 = tf.unstack( rt, axis = 1 )
        if not eval:
            rt1 = self.dp2( rt1 )
            rt2 = self.dp2( rt2 )

        # non recurrent phase
        x2 = self.fc2( x1 )
        h1x2, h2x2 = tf.unstack( x2, axis = 1 )
        if not eval:
            h1x2 = self.dp2( h1x2 )
            h2x2 = self.dp2( h2x2 )

        # recurrent vs non recurrent gate
        is_recurrent = tf.stop_gradient( self.envs[env]['recurrent'] )
        not_is_recurrent = tf.stop_gradient( 1 - self.envs[env]['recurrent'] )
        h = tf.stack( 
            [
                ( is_recurrent * rt1 ) + ( not_is_recurrent * h1x2 ),
                ( is_recurrent * rt2 ) + ( not_is_recurrent * h2x2 ),
            ], axis = 1
         )

        return h, present, mask, ( x, h1x1, h2x1, rth, rt1, rt2, h1x2, h2x2 )

    def call(self, state, pstate, env, past=None, eval=True):

        h, present, mask, vls = self.feature_extract( state, env, past, eval )
        
        if pstate is None: 
            return h, present, mask, vls

        hprev, presentprev, _, _ = self.feature_extract( pstate, env, past, eval )
        
        return h, hprev, present, presentprev, mask, vls


class GlobalMemory(Layer):

    def __init__(self, action_dim, h_size, m, n, encoder_size, max_size, 
                 num_blocks=2, n_read=4, n_att_heads=4, lr=1, decay=1e10, train=False):
        
        super(GlobalMemory, self).__init__()
        
        self.to_train = train
        self.action_dim = action_dim
        self.encoder_size = encoder_size
        
        self.memory = tdnc_v2( h_size, max_size, ( m, n ), num_blocks, n_read, n_att_heads, lr, decay )
        self.t = dense( h_size, activation = gelu, initializer = he_normal() )

        self.dp1 = tf.keras.layers.Dropout( .25 )

        self.aw = tf.Variable( tf.random.normal( [ action_dim, encoder_size ] ) )
        self.dw = tf.Variable( tf.random.normal( [ 2, encoder_size ] ) )

    def reset(self, bs):
        return self.memory.reset( bs )

    def call(self, state, pstate, ac, rw, dn, x_w, step, params, past=None, eval=True):
        
        # action encoder
        ae = tf.gather( self.aw, tf.cast( ac, tf.int32 ) )

        # reward encoder
        re = tf.random.normal( shape = [ tf.shape( rw )[0], tf.shape( rw )[1], self.encoder_size ], mean = rw[:,:,tf.newaxis], stddev = 0.001 )

        # done encoder
        de = tf.gather( self.dw, tf.cast( dn, tf.int32 ) )

        # env reactions
        ard = ae + re + de

        # read and write features
        a_state, t_state = tf.unstack( state, axis = 1 )
        a_pstate, t_pstate = tf.unstack( pstate, axis = 1 )

        hread = t_state + self.t( a_state )
        hwrite = t_pstate + self.t( a_pstate )

        f_read = tf.concat( [ hread, tf.zeros_like( ard ) ], axis = -1 )
        f_write = tf.concat( [ hwrite, ard ], axis = -1 )

        ## model memory
        y, p_state, mask = self.memory( f_read, f_write, x_w, past, params, step )
        
        if not eval: return self.dp1( y ), p_state[:-1], mask 
        else: return y, p_state[:-1], mask


class FeatureCreator(Layer):

    def __init__(self, f1, f2, train=False):

        super(FeatureCreator, self).__init__()

        self.to_train = train
        
        self.fc1 = nalu_transform( f1, activation = gelu, kernel_initializer = he_normal() )
        self.fc2 = nalu_transform( f2, activation = gelu, kernel_initializer = he_normal() )

        self.dp1 = tf.keras.layers.Dropout( .25 )
        self.dp2 = tf.keras.layers.Dropout( .25 )
        self.dp3 = tf.keras.layers.Dropout( .25 )

    def build(self, input_shapes):
        
        self.wqk = self.add_weight( name = 'wqk', shape = [ input_shapes[-1] * 2, input_shapes[-1] * 2 ], initializer = tf.keras.initializers.RandomNormal() )
        self.wv = self.add_weight( name = 'wv', shape = [ input_shapes[-1], input_shapes[-1] ], initializer = tf.keras.initializers.RandomNormal() )

    def call(self, state, memory, eval=True):

        # unstack states
        s1, s2 = tf.unstack( state, axis = 1 )

        # search inside memory using self attention
        f = tf.concat( [ s1, memory ], axis = -1 )
        q, k = tf.split( tf.expand_dims( f @ self.wqk, axis = 3 ), [ tf.shape(self.wqk)[-1] // 2 ] * 2, axis = -2 )
        v = tf.expand_dims( memory @ self.wv, axis = 3 )

        w = tf.matmul( q, k, transpose_b = True )
        w = w * tf.math.rsqrt( tf.cast( v.shape[-1], w.dtype ) )
        
        w = tf.nn.softmax( w )
        recovered = tf.matmul( w, v )[:,:,:,0]

        # create features
        mt = tf.stack( [ recovered, recovered ], axis = 1 )
        mtf1, mtf2 = tf.unstack( self.fc1( mt ), axis = 1 )

        if not eval:
            h = tf.stack( [ self.dp1( tf.concat( [ s1, mtf1 ], axis = -1 ) ),
                            self.dp2( tf.concat( [ s2, mtf2 ], axis = -1 ) ) ], axis = 1 )
        else:
            h = tf.stack( [ tf.concat( [ s1, mtf1 ], axis = -1 ),
                            tf.concat( [ s2, mtf2 ], axis = -1 ) ], axis = 1 )

        # non linear
        x = self.fc2( h )

        if eval:
            return x
        else:
            return self.dp3( x )


class FakeActor(Layer):

    def __init__(self, f1, action_dim, atoms, train=False):

        super(FakeActor, self).__init__()

        self.to_train = train
        self.action_dim = action_dim
        self.atoms = atoms
        self.ag_out_size = action_dim * atoms
        self.tau = [ ( 2 * ( i - 1 ) + 1 ) / ( 2 * self.atoms ) for i in range( 1, self.atoms + 1 ) ]

        self.fc1 = Dense( f1, kernel_initializer = he_normal(), activation = gelu )
        self.fc2 = Dense( f1, kernel_initializer = he_normal(), activation = gelu )
        self.fc3 = Dense( self.ag_out_size, kernel_initializer = orthogonal() )
        
    def call(self, features, eval=True):

        h1, h2 = tf.unstack( features, axis = 1 )
        
        x1 = self.fc1( h1 )
        x2 = self.fc2( h2 )

        x3 = self.fc3( x1 + x2 )

        return tf.reshape( x3, [ tf.shape( h1 )[0], tf.shape( h1 )[1], self.action_dim, self.atoms ] ), None
       
    def quantile_huber_loss_no_sequence(self, target, pred, actions):

        one_hot_action = tf.one_hot( actions, self.action_dim )
        pred = tf.reduce_sum( pred * tf.expand_dims( one_hot_action, -1 ), axis = 1 )

        pred_tile = tf.tile( tf.expand_dims( pred, axis = 2 ), [ 1, 1, self.atoms ] )        
        target_tile = tf.cast( tf.tile( tf.expand_dims( target, axis = 1 ), [ 1, self.atoms, 1 ] ), tf.float32 )
        
        huber_loss = tf.compat.v1.losses.huber_loss( target_tile, pred_tile, reduction = tf.keras.losses.Reduction.NONE )
        
        tau = tf.cast( tf.reshape( np.array( self.tau ), [ 1, self.atoms ] ), tf.float32 )
        inv_tau = tf.cast( 1.0 - tau, tf.float32 )
        tau = tf.cast( tf.tile( tf.expand_dims( tau, axis = 1 ), [ 1, self.atoms, 1 ] ), tf.float32 )
        inv_tau = tf.cast( tf.tile( tf.expand_dims( inv_tau, axis = 1 ), [ 1, self.atoms, 1 ] ), tf.float32 )
        
        error_loss = tf.math.subtract( target_tile, pred_tile )
        loss = tf.where( tf.less( error_loss, 0.0 ), inv_tau * huber_loss, tau * huber_loss )
        loss = tf.reduce_mean( loss, axis = 2 )
        loss = tf.reduce_sum( loss, axis = 1 )
        
        return loss

    def quantile_huber_loss(self, target, pred, actions):

        one_hot_action = tf.one_hot( actions, self.action_dim )
        pred = tf.reduce_sum( pred * tf.expand_dims( one_hot_action, -1 ), axis = 2 )

        pred_tile = tf.tile( tf.expand_dims( pred, axis = 3 ), [ 1, 1, 1, self.atoms ] )        
        target_tile = tf.cast( tf.tile( tf.expand_dims( target, axis = 2 ), [ 1, 1, self.atoms, 1 ] ), tf.float32 )
        
        huber_loss = tf.compat.v1.losses.huber_loss( target_tile, pred_tile, reduction = tf.keras.losses.Reduction.NONE )
        
        tau = tf.cast( tf.reshape( np.array( self.tau ), [ 1, self.atoms ] ), tf.float32 )
        inv_tau = tf.cast( 1.0 - tau, tf.float32 )
        tau = tf.cast( tf.tile( tf.expand_dims( tau, axis = 1 ), [ 1, self.atoms, 1 ] ), tf.float32 )
        inv_tau = tf.cast( tf.tile( tf.expand_dims( inv_tau, axis = 1 ), [ 1, self.atoms, 1 ] ), tf.float32 )
        
        error_loss = tf.math.subtract( target_tile, pred_tile )
        loss = tf.where( tf.less( error_loss, 0.0 ), inv_tau * huber_loss, tau * huber_loss )
        loss = tf.reduce_mean( loss, axis = 3 )
        loss = tf.reduce_sum( loss, axis = 2 )
        
        return loss

    def quantile_huber_loss_v2(self, target, pred, actions):

        one_hot_action = tf.one_hot( actions, self.action_dim )
        positive = tf.ones( [ tf.shape( target )[0], tf.shape( target )[1], self.action_dim ] )
        negative = positive / self.action_dim
        p = one_hot_action * positive
        n = ( 1 - one_hot_action ) * negative

        pred_tile = tf.tile( tf.expand_dims( pred, axis = 4 ), [ 1, 1, 1, 1, self.atoms ] )        
        target_tile = tf.cast( tf.tile( tf.expand_dims( target, axis = 3 ), [ 1, 1, 1, self.atoms, 1 ] ), tf.float32 )
        
        huber_loss = tf.compat.v1.losses.huber_loss( target_tile, pred_tile, reduction = tf.keras.losses.Reduction.NONE )
        
        tau = tf.cast( tf.reshape( np.array( self.tau ), [ 1, self.atoms ] ), tf.float32 )
        inv_tau = tf.cast( 1.0 - tau, tf.float32 )
        tau = tf.cast( tf.tile( tf.expand_dims( tau, axis = 1 ), [ 1, self.atoms, 1 ] ), tf.float32 )
        inv_tau = tf.cast( tf.tile( tf.expand_dims( inv_tau, axis = 1 ), [ 1, self.atoms, 1 ] ), tf.float32 )
        
        error_loss = tf.math.subtract( target_tile, pred_tile )
        loss = tf.where( tf.less( error_loss, 0.0 ), inv_tau[:,tf.newaxis,:,:] * huber_loss, tau[:,tf.newaxis,:,:] * huber_loss )
        loss = tf.reduce_mean( loss, axis = 4 )
        loss = tf.reduce_sum( loss, axis = 3 )
        loss = tf.reduce_sum( loss * ( p + n ), axis = 2 )
        
        return loss


class Actor(Layer):

    def __init__(self, f1, action_dim, atoms, train=False):

        super(Actor, self).__init__()

        self.to_train = train
        self.action_dim = action_dim
        self.f1 = f1
        self.atoms = atoms
        self.ag_out_size = action_dim * atoms
        self.tau = [ ( 2 * ( i - 1 ) + 1 ) / ( 2 * self.atoms ) for i in range( 1, self.atoms + 1 ) ]

        self.fc1 = nalu_transform( f1, activation = gelu, kernel_initializer = he_normal(), name = 'a1' )
        self.fc2 = nalu_transform( f1, activation = gelu, kernel_initializer = he_normal(), name = 'a2' )
        self.fc3 = dense( f1, activation = gelu, kernel_initializer = he_normal() )
        self.fc4 = dense( self.ag_out_size, kernel_initializer = orthogonal() )

        self.dp1 = tf.keras.layers.Dropout( .25 )

    def call(self, features, eval=True):
        
        h = tf.stack( [ features, features ], axis = 1 )

        # calcular os valores do actor
        h1 = self.fc1( h )
        h2 = self.fc2( h1 )
        ha, ht = tf.unstack( h2, axis = 1 )
        v = ht + self.fc3( ha )

        # redimensionar o actor para saida de tamanho variavel
        o = self.fc4( v )
        o = tf.reshape( o, [ tf.shape( o )[0], tf.shape( o )[1], self.action_dim, self.atoms ] )

        return o

    def quantile_huber_loss(self, target, pred, actions):

        one_hot_action = tf.one_hot( actions, self.action_dim )
        pred = tf.reduce_sum( pred * tf.expand_dims( one_hot_action, -1 ), axis = 1 )

        pred_tile = tf.tile( tf.expand_dims( pred, axis = 2 ), [ 1, 1, self.atoms ] )        
        target_tile = tf.cast( tf.tile( tf.expand_dims( target, axis = 1 ), [ 1, self.atoms, 1 ] ), tf.float32 )
        
        huber_loss = tf.compat.v1.losses.huber_loss( target_tile, pred_tile, reduction = tf.keras.losses.Reduction.NONE )
        
        tau = tf.cast( tf.reshape( np.array( self.tau ), [ 1, self.atoms ] ), tf.float32 )
        inv_tau = tf.cast( 1.0 - tau, tf.float32 )
        tau = tf.cast( tf.tile( tf.expand_dims( tau, axis = 1 ), [ 1, self.atoms, 1 ] ), tf.float32 )
        inv_tau = tf.cast( tf.tile( tf.expand_dims( inv_tau, axis = 1 ), [ 1, self.atoms, 1 ] ), tf.float32 )
        
        error_loss = tf.math.subtract( target_tile, pred_tile )
        loss = tf.where( tf.less( error_loss, 0.0 ), inv_tau * huber_loss, tau * huber_loss )
        loss = tf.reduce_mean( loss, axis = 2 )
        loss = tf.reduce_sum( loss, axis = 1 )
        
        return loss


class Actorv2(Layer):

    def __init__(self, f1, action_dim, quantile_dim, atoms, train=False):

        super(Actorv2, self).__init__()

        self.to_train = train
        self.action_dim = action_dim
        self.f1 = f1
        self.atoms = atoms
        self.quantile_dim = quantile_dim

        self.fc1 = dense( f1, activation = elu, kernel_initializer = random_normal() )
        self.fc2 = dense( f1, activation = elu, kernel_initializer = random_normal() )

        self.phi = dense( quantile_dim, kernel_initializer = random_normal(), bias = False )
        self.phi_bias = tf.cast( tf.Variable( tf.zeros( quantile_dim ) ), tf.float32 )
        self.fc = dense( f1, kernel_initializer = random_normal(), activation = elu )
        self.fc_q = dense( action_dim, kernel_initializer = orthogonal() )
        
        self.dp1 = tf.keras.layers.Dropout( .25 )
        self.dp2 = tf.keras.layers.Dropout( .25 )

    def call(self, features, eval=True):
        
        h = features

        h1 = self.fc1( h )
        h1 = self.dp1( h1 )
        if not eval: h1 = self.dp1( h1 )
        v = self.fc2( h1 )
        v = self.dp2( v )
        if not eval: v = self.dp2( v )

        tau = np.random.rand( self.atoms, 1 )
        pi_mtx = tf.constant( np.expand_dims( np.pi * np.arange( 0, self.quantile_dim ), axis = 0 ) )
        cos_tau = tf.cos( tf.matmul( tau, pi_mtx ) )
        phi = elu( self.phi( cos_tau ) + tf.expand_dims( self.phi_bias, axis = 0 ) )
        phi = tf.expand_dims( phi, axis = 0 )
        x = tf.reshape( v, ( -1, v.shape[-1] ) )
        x = tf.expand_dims( x, 1 )
        x = x * phi
        x = self.fc( x )
        x = self.fc_q( x )
        q = tf.transpose( x, [ 0, 2, 1 ] )

        return q, tau

    def quantile_huber_loss(self, target, pred, actions, tau):

        one_hot_action = tf.one_hot( actions, self.action_dim )
        pred = tf.reduce_sum( pred * tf.expand_dims( one_hot_action, -1 ), axis = 1 )

        pred_tile = tf.tile( tf.expand_dims( pred, axis = 2 ), [ 1, 1, self.atoms ] )        
        target_tile = tf.cast( tf.tile( tf.expand_dims( target, axis = 1 ), [ 1, self.atoms, 1 ] ), tf.float32 )
        
        huber_loss = tf.compat.v1.losses.huber_loss( target_tile, pred_tile, reduction = tf.keras.losses.Reduction.NONE )
        
        tau = tf.cast( tf.reshape( np.array( tau ), [ 1, self.atoms ] ), tf.float32 )
        inv_tau = tf.cast( 1.0 - tau, tf.float32 )
        tau = tf.cast( tf.tile( tf.expand_dims( tau, axis = 1 ), [ 1, self.atoms, 1 ] ), tf.float32 )
        inv_tau = tf.cast( tf.tile( tf.expand_dims( inv_tau, axis = 1 ), [ 1, self.atoms, 1 ] ), tf.float32 )
        
        error_loss = tf.math.subtract( target_tile, pred_tile )
        loss = tf.where( tf.less( error_loss, 0.0 ), inv_tau * huber_loss, tau * huber_loss )
        loss = tf.reduce_sum( tf.reduce_mean( loss, axis = -1 ), axis = -1 )
        
        return loss


class ActorMultiEnv_v2(Layer):

    def __init__(self, envs, f1, atoms):

        super(ActorMultiEnv_v2, self).__init__()

        self.f1 = f1
        self.atoms = atoms
        self.tau = [ ( 2 * ( i - 1 ) + 1 ) / ( 2 * self.atoms ) for i in range( 1, self.atoms + 1 ) ]

        self.fc1e = dense_w( f1, activation = gelu, name = 'ac_e1' )
        self.fc2e = dense_w( f1, activation = gelu, name = 'ac_e2' )
        
        self.fc1 = nalu_transform( f1, activation = gelu, kernel_initializer = he_normal(), initializer = he_normal(), name = 'ac_e3' )

        self.mini_net_sizes = [ 3, 5, 7, 9, 11, 15, 19, 25, 35 ]
        self.mini_kn = {}
        for m in self.mini_net_sizes:
            self.mini_kn[ str( m ) ] = self.add_weight( 'kn_{}'.format( m ), [ m, m ], initializer = he_normal(), trainable = True )
        
        self.fc1 = nalu_transform( f1, activation = gelu, kernel_initializer = he_normal(), name = 'act_fc1' )
        self.fc2 = nalu_transform( f1, activation = gelu, kernel_initializer = he_normal(), name = 'act_fc2' )
        self.fc3 = dense( f1, initializer = orthogonal() )

        self.envs = {}
        for e in envs:
            self.create_weigth_mask( e['id'], e['action_dim'], f1 )

        self.dp1 = tf.keras.layers.Dropout( .25 )
        self.dp2 = tf.keras.layers.Dropout( .25 )

    def create_weigth_mask(self, name, size, f1):
        
        sz = size * self.atoms
        aux1 = sz
        aux2 = f1
        dvs = [ 0 ]
        w = np.zeros( [ f1, sz ] )
        we = np.zeros( [ f1, f1 ] )
        for i in range( sz ):
            dv = aux2 / aux1
            v = ceil( dv ) if dv - int( dv ) > .5 else int( dv )
            w[ np.sum( dvs ) : np.sum( dvs ) + v, i ] = ( 1 / v )
            for j in range( np.sum( dvs ), np.sum( dvs ) + v ):
                we[ j, np.sum( dvs ) : np.sum( dvs ) + v ] = 1
            dvs.append( v )
            aux1 -= 1
            aux2 -= v
       
        mini_net_size = np.max( dvs[1:] )
        selected = np.argmin( np.abs( mini_net_size - np.array( self.mini_net_sizes ) ) )
        sz = self.mini_net_sizes[ selected ]

        self.envs[name] = { 'w': w, 'we': we, 'mini_net': str( sz ), 'sz': ceil( f1 / sz ), 'action_dim': size }

    def call(self, features, env, eval=True):
                
        # calcular os valores hidden do actor
        h1 = self.fc1( features )
        h11, h12 = tf.unstack( h1, axis = 1 )
        if not eval:
            h11 = self.dp1( h1 )
            h12 = self.dp1( h2 )
        
        h2 = self.fc2( tf.stack( [ h11, h12 ], axis = 1 ) )
        ha, ht = tf.unstack( h2, axis = 1 )
        if not eval:
            ha = self.dp2( ha )
            ht = self.dp2( ht )
        h = tf.concat( [ ha, ht ], axis = -1 )

        # mini networks - redes criadas para adaptar a saida de tamanho variavel
        xa1 = self.fc1e( h, self.envs[env]['sz'], self.mini_kn[self.envs[env]['mini_net']], self.envs[env]['we'] )
        xa2 = self.fc2e( xa1, self.envs[env]['sz'], self.mini_kn[self.envs[env]['mini_net']], self.envs[env]['we'] )

        # linear transform
        v = self.fc3( xa2, self.envs[env]['we'] )
       
        # redimensionar o actor para saida de tamanho variavel
        o = tf.matmul( xa1, self.envs[env]['w'] )
        o = tf.reshape( o, [ tf.shape( o )[0], tf.shape( o )[1], self.envs[env]['action_dim'], self.atoms ] )

        return o, v

    def quantile_huber_loss(self, target, pred, actions):

        one_hot_action = tf.one_hot( actions, self.envs[env]['action_dim'] )
        pred = tf.reduce_sum( pred * tf.expand_dims( one_hot_action, -1 ), axis = 2 )

        pred_tile = tf.tile( tf.expand_dims( pred, axis = 3 ), [ 1, 1, 1, self.atoms ] )        
        target_tile = tf.cast( tf.tile( tf.expand_dims( target, axis = 2 ), [ 1, 1, self.atoms, 1 ] ), tf.float32 )
        
        huber_loss = tf.compat.v1.losses.huber_loss( target_tile, pred_tile, reduction = tf.keras.losses.Reduction.NONE )
        
        tau = tf.cast( tf.reshape( np.array( self.tau ), [ 1, self.atoms ] ), tf.float32 )
        inv_tau = tf.cast( 1.0 - tau, tf.float32 )
        tau = tf.cast( tf.tile( tf.expand_dims( tau, axis = 1 ), [ 1, self.atoms, 1 ] ), tf.float32 )
        inv_tau = tf.cast( tf.tile( tf.expand_dims( inv_tau, axis = 1 ), [ 1, self.atoms, 1 ] ), tf.float32 )
        
        error_loss = tf.math.subtract( target_tile, pred_tile )
        loss = tf.where( tf.less( error_loss, 0.0 ), inv_tau * huber_loss, tau * huber_loss )
        loss = tf.reduce_mean( loss, axis = 3 )
        loss = tf.reduce_sum( loss, axis = 2 )
        
        return loss


class Critic(Layer):

    def __init__(self, atoms, train=False):

        super(Critic, self).__init__()

        self.to_train = train
        self.atoms = atoms
        self.tau = [ ( 2 * ( i - 1 ) + 1 ) / ( 2 * self.atoms ) for i in range( 1, self.atoms + 1 ) ]

        self.fc1 = dense( self.atoms, initializer = tf.keras.initializers.orthogonal() )

        self.dp1 = tf.keras.layers.Dropout( .5 )
        self.dp2 = tf.keras.layers.Dropout( .5 )

    def build(self, input_shapes):
        
        self.wqk = self.add_weight( name = 'wqk', shape = [ input_shapes[-1] * 3, input_shapes[-1] * 2 ], initializer = tf.keras.initializers.RandomNormal() )
        self.wv = self.add_weight( name = 'wv', shape = [ input_shapes[-1], input_shapes[-1] ], initializer = tf.keras.initializers.RandomNormal() )
    
    def call(self, state, actorh, memory, eval=True):

        # unstack states
        _, s = tf.unstack( state, axis = 1 )

        # search inside memory using self attention
        f = tf.concat( [ s, actorh, memory ], axis = -1 )
        q, k = tf.split( tf.expand_dims( f @ self.wqk, axis = 3 ), [ tf.shape(self.wqk)[-1] // 2 ] * 2, axis = -2 )
        v = tf.expand_dims( memory @ self.wv, axis = 3 )

        w = tf.matmul( q, k, transpose_b = True )
        w = w * tf.math.rsqrt( tf.cast( v.shape[-1], w.dtype ) )
        
        w = tf.nn.softmax( w )
        recovered = tf.matmul( w, v )[:,:,:,0]

        # linear
        x = self.fc1( recovered )

        return x

    def quantile_huber_loss(self, target, pred):
        
        pred_tile = tf.tile( tf.expand_dims( pred, axis = 3 ), [ 1, 1, 1, self.atoms ] )        
        target_tile = tf.cast( tf.tile( tf.expand_dims( target, axis = 2 ), [ 1, 1, self.atoms, 1 ] ), tf.float32 )
        
        huber_loss = tf.compat.v1.losses.huber_loss( target_tile, pred_tile, reduction = tf.keras.losses.Reduction.NONE )
        
        tau = tf.cast( tf.reshape( np.array( self.tau ), [ 1, self.atoms ] ), tf.float32 )
        inv_tau = tf.cast( 1.0 - tau, tf.float32 )
        tau = tf.cast( tf.tile( tf.expand_dims( tau, axis = 1 ), [ 1, self.atoms, 1 ] ), tf.float32 )
        inv_tau = tf.cast( tf.tile( tf.expand_dims( inv_tau, axis = 1 ), [ 1, self.atoms, 1 ] ), tf.float32 )
        
        error_loss = tf.math.subtract( target_tile, pred_tile )
        loss = tf.where( tf.less( error_loss, 0.0 ), inv_tau[:,tf.newaxis,:,:] * huber_loss, tau[:,tf.newaxis,:,:] * huber_loss )
        loss = tf.reduce_sum( tf.reduce_mean( loss, axis = 3 ), axis = 2 )
        
        return loss



