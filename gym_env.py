import gym
import tensorflow as tf
import numpy as np
from gym.spaces import Box, Discrete, Tuple
import cv2

class GYMEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array']}
    def __init__(self, *args):
        # self._render_size = render_size

        self.image_space = Box(
            -1,
            1,
            shape=(96, 96, 2),
            dtype=np.float32)
        # print("KKKKKKKKKK\n" * 10)
        # env.observation_space
        # Out[11]: Box(4, )
        # self._env = gym.make("BipedalWalker-v2")
        self._env = gym.make("CartPole-v1")

        self.observation_space = gym.spaces.Tuple([self.image_space, self._env.observation_space, self._env.observation_space])
        self.action_space = self._env.action_space
        self.obs_list = []

    def step(self, action):
        # print(">>>>>>>>>>>>>>>>>>>>>>>")
        state, reward, done, info = self._env.step(action)
        obs = (self.render(), state, state)
        # if done:
        #     self._env.close()
        return (obs, reward, done, info)

    def reset(self):
        # print("<<<<<<<<<<<<<<<<<<<<<<<<")
        state = self._env.reset()
        obs = (self.render(), state, state)
        return obs

    def render(self):
        obs = self._env.render(mode="rgb_array")
        obs = cv2.resize(
            obs, (96, 96),
            interpolation=cv2.INTER_AREA)
        # obs = (obs - 128)/128
        gray = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
        gray = (gray.astype(np.float32) - 128)/128
        self.obs_list.append(gray)
        if len(self.obs_list) > 1:
            obs_ = np.stack(self.obs_list[-1:-3:-1], -1)
        else:
            obs_ = np.stack([gray, gray], -1)
        if len(self.obs_list) > 32:
            self.obs_list.pop(0)
        return obs_

if __name__ == "__main__":
    for _ in range(4):
        env = GYMEnv()
        # env = gym.make("CartPole-v1")
        obs = env.reset()
        # print(obs[0].shape)

        done = False
        i = 0
        total_reward = 0.0
        while not done:
            i += 1
            obs, reward, done, info = env.step(0)
            print(obs[0].shape)
            # print(i)
            # env.render()
        # env.close()
