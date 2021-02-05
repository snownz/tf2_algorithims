"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

from gym.spaces import Box
from gym import Wrapper
import cv2
import numpy as np
import subprocess as sp

def process_frame(frame):

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (84, 84))[:,:,np.newaxis] / 255.
    return frame

class CustomMarioReward(Wrapper):
    
    def __init__(self, env=None):
        super(CustomMarioReward, self).__init__(env)
        self.curr_score = 0
        
    def step(self, action):
        state_, reward, done, info = self.env.step(action)
        state = process_frame(state_)
        reward += (info["score"] - self.curr_score) / 40.
        self.curr_score = info["score"]
        if done:
            if info["flag_get"]: reward += 50
            else: reward -= 50
        return state, state_, reward / 10., done, info

    def reset(self):
        self.curr_score = 0
        im = self.env.reset()
        return process_frame( im ), im

class CustomAtariReward(Wrapper):
    
    def __init__(self, env=None):
        super(CustomAtariReward, self).__init__(env)
        self.curr_score = 0
        
    def step(self, action):
        state_, reward, done, info = self.env.step(action)
        state = cv2.resize( state_, (84, 84)) / 255.
        if done:
            _, state_ = self.reset()
            return cv2.resize( state_, (84, 84)) / 255., state_, reward, done, info
        return state, state_, reward, done, info

    def reset(self):
        self.curr_score = 0
        im = self.env.reset()
        return cv2.resize( im, (84, 84)) / 255., im

class CustomSkipFrame(Wrapper):
    def __init__(self, env, skip=4):
        super(CustomSkipFrame, self).__init__(env)
        self.skip = skip
        self.states = np.zeros((84, 84, skip), dtype=np.float32)

    def step(self, action):
        total_reward = 0
        last_states = []
        for i in range(self.skip):
            state, state_, reward, done, info = self.env.step(action)
            total_reward += reward
            if i >= self.skip / 2:
                last_states.append(state)
            if done:
                _, state_ = self.reset()
                return self.states[:, :, :].astype(np.float32), state_, total_reward, done, info
        max_state = np.max(np.concatenate(last_states, -1), -1)
        self.states[:,:,:-1] = self.states[:,:,1:]
        self.states[:,:,-1] = max_state
        return self.states[:, :, :].astype(np.float32), state_, total_reward, done, info

    def reset(self):
        state, state_ = self.env.reset()
        self.states = np.concatenate( [ state for _ in range( self.skip ) ], -1 )
        return self.states[ :, :, :].astype(np.float32), state_


