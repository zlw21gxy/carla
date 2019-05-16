import gym
import tensorflow as tf
import numpy as np
from gym.spaces import Box, Discrete, Tuple


class GYMENV(gym.Env):

    def __init__(self, render_size=(96, 96, 3)):
        self._env = gym.make("CartPole-v1")
        self._render_size = render_size

        self.observation_space = Box(
            -1,
            1,
            shape=render_size,
            dtype=np.float32)

        self.observation_space = gym.spaces.Dict({'state': self.observation_space})
    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs = {'state': obs}
        return obs, reward, done, info

    def reset(self):
        return {'state': self._env.reset()}

    def render(self):
        obs = self.env.render(mode="rgb_array", shape=self._render_size)
        return obs
