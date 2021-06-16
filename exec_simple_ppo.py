import gym
from gym_minigrid.wrappers import *
from multi_env_warper import ImageEnv, floatEnv, miniGridEnv

from agents import SimplePPOAgent, NaluPPOAgent, NaluAdvancedPPOAgent, NaluAdvanced2PPOAgent, IQNNaluPPOAgent
from agents import SimpleRnnPPOAgent
from agents import SimpleConvPPOAgent

evs = [ "LunarLander-v2", "gym_rocketlander:rocketlander-v0" ]

# env = gym.make( evs[0] )
env = ImageEnv( gym.make( evs[0] ) ) # for pixel env

ag = SimpleConvPPOAgent( env, evs[0] )
# ag = SimpleRnnPPOAgent( env, env_name, 's', 'mean_half', 16, 128, sizes = [ 512, 256, 64 ] )

ag.run_multiprocesses( 6 )
# ag.test( 20, load = True )

# ag.test( 20, disable_recurrence = False )
# ag.test( 20, disable_recurrence = True )