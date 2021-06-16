import gym
import os
from gym_minigrid.wrappers import *
from multi_env_warper import miniGridEnv

from agents import SimplePPOAgent, NaluPPOAgent, NaluAdvancedPPOAgent, NaluAdvanced2PPOAgent, IQNNaluPPOAgent
from agents import SimpleRnnPPOAgent
from agents import SimpleConvPPOAgent

envs = [
    'MiniGrid-Empty-5x5-v0',           # 0
    'MiniGrid-Empty-Random-5x5-v0',    # 1
    'MiniGrid-Empty-6x6-v0',           # 2
    'MiniGrid-Empty-Random-6x6-v0',    # 3
    'MiniGrid-Empty-8x8-v0',           # 4
    'MiniGrid-Empty-16x16-v0',         # 5
    'MiniGrid-FourRooms-v0',           # 6
    'MiniGrid-DoorKey-5x5-v0',         # 7
    'MiniGrid-DoorKey-6x6-v0',         # 8
    'MiniGrid-DoorKey-8x8-v0',         # 9
    'MiniGrid-DoorKey-16x16-v0',       # 10
    'MiniGrid-MultiRoom-N2-S4-v0',     # 11
    'MiniGrid-MultiRoom-N4-S5-v0',     # 12
    'MiniGrid-MultiRoom-N6-v0',        # 13
]

env_name = envs[6]
env = miniGridEnv( gym.make( env_name ) )

ag = SimpleRnnPPOAgent( env, env_name, 'g', 'mean_half', 32, 128, sizes = [ 512, 256, 64 ] )
# ag = SimpleRnnPPOAgent( env, env_name )

# ag.restore_training( os.getcwd() + '/models/{}/{}'.format( ag.name, envs[6] ) )
ag.replay_count = 0
ag.run_multiprocesses( 12, miniGridEnv )
ag.test( 20, load = False )
