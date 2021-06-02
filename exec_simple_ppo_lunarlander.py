import os
import random
import gym
import numpy as np
import tensorflow as tf
# tf.compat.v1.disable_eager_execution() # usually using this for fastest performance
# import copy

from agents import NaluAdvancedPPOAgent

env = gym.make( "LunarLander-v2" )
ag = NaluAdvancedPPOAgent( env )
ag.run_multiprocesses( 6 )
#ag.test()