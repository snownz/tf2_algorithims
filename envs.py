import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
import gym
from env import CustomMarioReward, CustomSkipFrame, CustomAtariReward

def create_mario_train_env(world, stage, actions, output_path=None, random=False):
    if random:
        env = gym_super_mario_bros.make('SuperMarioBrosRandomStages-v0')
    else:
        # env = gym_super_mario_bros.make("SuperMarioBros-{}-{}-v0".format(world, stage))
        env = gym_super_mario_bros.make("SuperMarioBros-v0")
   
    env = JoypadSpace(env, actions)
    env = CustomMarioReward(env)
    env = CustomSkipFrame(env)
    return env

def create_atari_train_env(game, actions):

    env = gym.make(game)   
    # env = JoypadSpace(env, actions)
    env = CustomAtariReward(env)
    return env
