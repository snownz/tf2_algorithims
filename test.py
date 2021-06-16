import tensorflow as tf
import numpy as np
import tensorflow as tf
from ann_utils import nalu

size = 5

l1 = nalu( size, 'n1' )
l2 = nalu( size, 'n1' )
l3 = nalu( size, 'n1' )
l4 = nalu( size, 'n1' )
l5 = nalu( size, 'n1' )
l6 = nalu( size, 'n1' )
l7 = nalu( size, 'n1' )

input = tf.Variable( np.random.randint( 0, 5, [ 1, size ] ), dtype = tf.float32 )

x1 = l1( input )
x2 = l2( x1 )
x3 = l3( x2 )
x4 = l4( x3 )
x5 = l5( x4 )
x6 = l6( x5 )
x7 = l7( x6 )

print('')
print( list( input[0].numpy() ) )
print('')
print( list( x1[0].numpy() ) )
print('')
print( list( x2[0].numpy() ) )
print('')
print( list( x3[0].numpy() ) )
print('')
print( list( x4[0].numpy() ) )
print('')
print( list( x5[0].numpy() ) )
print('')
print( list( x6[0].numpy() ) )
print('')
print( list( x7[0].numpy() ) )
print('')
