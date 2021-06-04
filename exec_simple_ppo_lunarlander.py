import os
import random
import gym
import numpy as np
import tensorflow as tf
# tf.compat.v1.disable_eager_execution() # usually using this for fastest performance
# import copy

from agents import SimplePPOAgent, NaluPPOAgent, NaluAdvancedPPOAgent, NaluAdvanced2PPOAgent, IQNNaluPPOAgent
from agents import SimpleRnnPPOAgent

env_name = "LunarLander-v2"
# env_name = "gym_rocketlander:rocketlander-v0"
env = gym.make( env_name )

# ag = SimplePPOAgent( env, env_name )
ag = SimpleRnnPPOAgent( env, env_name, 'g', 'sum_half', 16, 512 )

ag.run_multiprocesses( 12 )
ag.test( 20, disable_recurrence = False )
ag.test( 20, disable_recurrence = True )