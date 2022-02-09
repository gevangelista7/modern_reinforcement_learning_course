import numpy as np
import gym
from collections import deque
import cv2


class RepeatActionAndMaxFrame(gym.wrappers):
    def __init__(self, env: gym.Env, repeat):
        super(RepeatActionAndMaxFrame, self).__init__()
        self.env = env
        self.repeat = repeat
        self.init_frame = np.zeros(shape=env.observation_space * 2)
        self.frame_buffer = deque()
        self.frame_buffer.append(self.init_frame)

    def step(self, action):
        reward = 0
        done = False
        for i in range(self.repeat):
            obs, reward_i, done, info = self.env.step(action)
            reward += reward_i
            self.frame_buffer.append(obs)
            if done:
                break
        max_frame = cv2.max(*self.frame_buffer)
        return max_frame, reward, done, info

    def reset(self):
        obs = self.env.reset()
        self.frame_buffer.clear()
        self.frame_buffer.append(self.init_frame)
        return obs

class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, new_shape):
        super(PreprocessFrame, self).__init__(env=env)
        self.env = env
        self.new_shape = new_shape

    def observation(self, observation):
        observation = np.array(observation, dtype=np.uint8)
        # observation = cv2.adaptiveThreshold(observation, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
        observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        observation = cv2.resize(observation, dsize=self.new_shape)
        observation = np.array(observation).transpose([2, 1, 0])
        observation /= 255
        return observation

class StackFrames(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, ):
        super(StackFrames, self).__init__()






