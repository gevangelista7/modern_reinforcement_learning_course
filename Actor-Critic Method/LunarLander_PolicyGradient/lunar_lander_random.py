import gym
import numpy as np


if __name__ == '__main__':
    env = gym.make("LunarLander-v2")
    obs = env.reset()
    done = False
    while not done:
        action = np.random.choice(env.action_space.n)
        obs_, reward, done, _ = env.step(action)
        obs = obs_
        env.render()

