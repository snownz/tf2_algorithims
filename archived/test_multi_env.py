import os, json
import gym
import numpy as np
import tensorflow as tf
from tqdm import trange
from multi_env_warper import make_mp_envs

env = make_mp_envs( [ 'gym_rocketlander:rocketlander-v0', 'LunarLander-v2' ], 15 )

state = env.reset()

# sample action            
actions = env.sample()

# remove the batch_size dimension if batch_size == 1
next_state, reward, is_terminal, info = env.step( actions )

# check if game is terminated to decide how to update state, episode_steps,
# episode_rewards
if is_terminal:
    state = np.float32(env.reset())
else:
    state = next_state