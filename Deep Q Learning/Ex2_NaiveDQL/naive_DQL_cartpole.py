from Agent import Agent
from NNEstimator import NNEstimator
import numpy as np
import gym
import torch
from plot_learning_curve import plot_learning_curve

WIN_UPDATE = 100
N_GAMES = 10000

env = gym.make("CartPole-v1")

if __name__ == '__main__':
    actions_n = env.action_space.n
    state_dim = env.observation_space.shape
    device = "cuda" if torch.cuda.is_available() else "cpu"

    agent = Agent(actions_n=actions_n, state_dim=state_dim, lr=0.0001)
    win_rate_list = []
    scores = []
    eps_history = []
    exp_buffer = {}

    for i in range(N_GAMES):
        obs = env.reset()
        score = 0
        done = False
        while not done:
            action = agent.choose_action(obs)
            next_obs, reward, done, _ = env.step(action)
            agent.learn(obs, action, reward, next_obs)
            score += reward
            obs = next_obs
        scores.append(score)
        eps_history.append(agent.epsilon)

        if i % WIN_UPDATE == 0:
            win_rate = np.mean(scores[-WIN_UPDATE:])
            win_rate_list.append(win_rate)
            print("episode: {} \twin rate: {:.3} \tepsilon: {:.5}".format(i, win_rate, agent.epsilon))
    x = [i+1 for i in range(N_GAMES)]
    plot_learning_curve(x, scores, eps_history)




