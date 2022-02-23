import numpy as np
import gym
from collections import deque
import cv2
import random as rd
import matplotlib.pyplot as plt


def plot_learning_curve(x, scores, epsilons, filename="performance.png", lines=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t - 20):(t + 1)])

    ax2.scatter(x, running_avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.show()
    # plt.savefig(filename)


#######################################################################
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
            self.frame_buffer[i % 2] = obs
            # self.frame_buffer.append(obs)
            if done:
                break
        max_frame = np.maximum(*self.frame_buffer)
        return max_frame, reward, done, info

    def reset(self):
        obs = self.env.reset()
        self.frame_buffer = deque(maxlen=2)
        self.frame_buffer = np.zeros_like((2, self.env.observation_space.low))
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
        observation = cv2.resize(observation, dsize=self.shape[1:], interpolation=cv2.INTER_AREA)
        observation = np.array(observation, dtype=np.uint8).reshape(self.shape)
        observation = observation/255.0
        return observation


# stolen from Phil Tabor
class StackFrames(gym.ObservationWrapper):
    def __init__(self, env, repeat):
        super(StackFrames, self).__init__(env)
        self.observation_space = gym.spaces.Box(
                            env.observation_space.low.repeat(repeat, axis=0),
                            env.observation_space.high.repeat(repeat, axis=0),
                            dtype=np.float32)
        self.stack = deque(maxlen=repeat)

    def reset(self):
        self.stack.clear()
        observation = self.env.reset()
        for _ in range(self.stack.maxlen):
            self.stack.append(observation)

        return np.array(self.stack).reshape(self.observation_space.low.shape)

    def observation(self, observation):
        self.stack.append(observation)

        return np.array(self.stack).reshape(self.observation_space.low.shape)


def make_env(env_name, shape=(84, 84, 1), repeat=4):
    env = gym.make(env_name)
    env = RepeatActionAndMaxFrame(env, repeat)
    env = PreprocessFrame(env, shape)
    env = StackFrames(env, repeat)

    return env

#####################################################################
# stolen from Phil Tabor
class ExperienceBuffer:
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size

        self.states = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.states_ = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.actions = np.zeros(self.mem_size, dtype=np.int64)
        self.rewards = np.zeros(self.mem_size, dtype=np.float32)
        self.dones = np.zeros(self.mem_size, dtype=np.bool)

        self.mem_idx = 0

    def insert(self, state, action, reward, state_, done):
        idx = self.mem_idx % self.mem_size
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.states_[idx] = state_
        self.dones[idx] = done
        self.mem_idx += 1

    def sample(self, batch_size):
        max_mem = min(self.mem_idx, self.mem_size)
        batch_idx = np.random.choice(max_mem, batch_size, replace=False)
        b_states = self.states[batch_idx]
        b_actions = self.actions[batch_idx]
        b_rewards = self.rewards[batch_idx]
        b_states_ = self.states_[batch_idx]
        b_dones = self.dones[batch_idx]
        return b_states, b_actions, b_rewards, b_states_, b_dones


###########################################################################
### ORIGINAL VERSION FROM PHIL TABOR:
# class RepeatActionAndMaxFrame(gym.Wrapper):
#     """ modified from:
#         https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/blob/master/Chapter06/lib/wrappers.py
#     """
#
#     def __init__(self, env=None, repeat=4, clip_reward=False,
#                  no_ops=0, fire_first=False):
#         super(RepeatActionAndMaxFrame, self).__init__(env)
#         self.repeat = repeat
#         self.shape = env.observation_space.low.shape
#         self.frame_buffer = np.zeros_like((2, self.shape))
#         self.clip_reward = clip_reward
#         self.no_ops = 0
#         self.fire_first = fire_first
#
#     def step(self, action):
#         t_reward = 0.0
#         done = False
#         for i in range(self.repeat):
#             obs, reward, done, info = self.env.step(action)
#             if self.clip_reward:
#                 reward = np.clip(np.array([reward]), -1, 1)[0]
#             t_reward += reward
#             idx = i % 2
#             self.frame_buffer[idx] = obs
#             if done:
#                 break
#
#         max_frame = np.maximum(self.frame_buffer[0], self.frame_buffer[1])
#         return max_frame, t_reward, done, info
#
#     def reset(self):
#         obs = self.env.reset()
#         no_ops = np.random.randint(self.no_ops) + 1 if self.no_ops > 0 else 0
#         for _ in range(no_ops):
#             _, _, done, _ = self.env.step(0)
#             if done:
#                 self.env.reset()
#
#         if self.fire_first:
#             assert self.env.unwrapped.get_action_meanings()[1] == 'FIRE'
#             obs, _, _, _ = self.env.step(1)
#
#         self.frame_buffer = np.zeros_like((2, self.shape))
#         self.frame_buffer[0] = obs
#         return obs
#
#
# class PreprocessFrame(gym.ObservationWrapper):
#     def __init__(self, shape, env=None):
#         super(PreprocessFrame, self).__init__(env)
#         self.shape = (shape[2], shape[0], shape[1])
#         self.observation_space = gym.spaces.Box(low=0, high=1.0,
#                                                 shape=self.shape, dtype=np.float32)
#
#     def observation(self, obs):
#         new_frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
#         resized_screen = cv2.resize(new_frame, self.shape[1:],
#                                     interpolation=cv2.INTER_AREA)
#         new_obs = np.array(resized_screen, dtype=np.uint8).reshape(self.shape)
#         new_obs = new_obs / 255.0
#         return new_obs
#
#
# class StackFrames(gym.ObservationWrapper):
#     def __init__(self, env, repeat):
#         super(StackFrames, self).__init__(env)
#         self.observation_space = gym.spaces.Box(
#             env.observation_space.low.repeat(repeat, axis=0),
#             env.observation_space.high.repeat(repeat, axis=0),
#             dtype=np.float32)
#         self.stack = deque(maxlen=repeat)
#
#     def reset(self):
#         self.stack.clear()
#         observation = self.env.reset()
#         for _ in range(self.stack.maxlen):
#             self.stack.append(observation)
#
#         return np.array(self.stack).reshape(self.observation_space.low.shape)
#
#     def observation(self, observation):
#         self.stack.append(observation)
#         obs = np.array(self.stack).reshape(self.observation_space.low.shape)
#
#         return obs
#
#
# def make_env(env_name, shape=(84, 84, 1), repeat=4, clip_rewards=False,
#              no_ops=0, fire_first=False):
#     env = gym.make(env_name)
#     env = RepeatActionAndMaxFrame(env, repeat, clip_rewards, no_ops, fire_first)
#     env = PreprocessFrame(shape, env)
#     env = StackFrames(env, repeat)
#
#     return env
