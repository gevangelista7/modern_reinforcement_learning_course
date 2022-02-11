import numpy as np
import gym
from collections import deque
import cv2
import random as rd


class RepeatActionAndMaxFrame(gym.Wrapper):
    def __init__(self, env: gym.Env, repeat):
        super(RepeatActionAndMaxFrame, self).__init__(env)
        self.env = env
        self.repeat = repeat
        self.frame_buffer = deque(maxlen=2)
        # self.frame_buffer = np.zeros_like((2,self.env.observation_space.low))

    def step(self, action):
        reward = 0
        done = False
        for i in range(self.repeat):
            obs, reward_i, done, info = self.env.step(action)
            reward += reward_i
            self.frame_buffer.append(obs)
            if done:
                break
        max_frame = np.maximum(*self.frame_buffer)
        return max_frame, reward, done, info

    def reset(self):
        obs = self.env.reset()
        self.frame_buffer.clear()
        self.frame_buffer = np.zeros_like((2,self.env.observation_space.low))
        self.frame_buffer[0] = obs
        return obs


class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, new_shape):
        super(PreprocessFrame, self).__init__(env=env)
        self.env = env
        self.shape = (new_shape[2], new_shape[0], new_shape[1])
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=self.shape, dtype=np.float32)

    def observation(self, observation):
        observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        observation = cv2.resize(observation, dsize=self.new_shape, shape=self.shape[1:], interpolation=cv2.INTER_AREA)
        observation = np.array(observation).reshape(self.shape)
        observation /= 255.0
        return observation


class StackFrames(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, stack_size=4):
        super(StackFrames, self).__init__(env)
        self.env = env
        self.observation_space = gym.spaces.Box(low=env.observation_space.low.repeat(stack_size, axis=0),
                                             high=env.observation_space.high.repeat(stack_size, axis=0),
                                             dtype=np.float32)

        self.shape = env.observation_space.low.shape
        self.frame_stack = deque(maxlen=stack_size)

    def reset(self):
        self.frame_stack.clear()
        obs = self.env.reset()
        for i in range(self.stack_size):
            self.frame_stack.append(obs)
        return np.array(self.frame_stack).reshape(self.shape)

    def observation(self, obs):
        self.frame_stack.append(obs)
        return np.array(self.frame_stack).reshape(self.shape)


def make_env(env_name, shape=(84,84,1), repeat=4):
    env = gym.make(env_name)
    env = RepeatActionAndMaxFrame(env, repeat)
    env = PreprocessFrame(env, shape)
    env = StackFrames(env, repeat)

    return env


class ExperienceBuffer:
    def __init__(self, memory_len):
        self.states = deque(maxlen=memory_len)
        self.actions = deque(maxlen=memory_len)
        self.rewards = deque(maxlen=memory_len)
        self.states_ = deque(maxlen=memory_len)
        self.dones = deque(maxlen=memory_len)

    def all_memory_zipped(self):
        return tuple(zip(self.states, self.actions, self.rewards, self.states_, self.dones))

    def insert(self, state, action, reward, state_,done):
        assert (state, action, reward, state_) not in self.all_memory_zipped()
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.states_.append(state_)
        self.dones.append(done)

    def sample(self):
        rand_exp = rd.choice(self.all_memory_zipped())
        np_exp = np.array(rand_exp, dtype=np.float32)
        return np_exp
