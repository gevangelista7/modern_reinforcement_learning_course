from ql_agent import Agent
import numpy as np
import matplotlib.pyplot as plt
import gym

WIN_UPDATE = 1000
N_GAMES = 500000

env = gym.make("FrozenLake-v1")

if __name__ == '__main__':
    agent = Agent(lr=0.001, gamma=0.9, eps_end=0.01, eps_start=1, eps_decay=0.9999995, n_actions=4, n_states=16)
    win_rate_list = []
    scores = []

    for i in range(N_GAMES):
        state = env.reset()
        score = 0
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state=state, action=action, next_state=next_state, reward=reward )
            score += reward
            state = next_state
        scores.append(score)
        if i % WIN_UPDATE == 0:
            win_rate = np.mean(scores[-WIN_UPDATE:])
            win_rate_list.append(win_rate)
            print("episode: {} \twin rate: {:.3} \tepsilon: {:.5}".format(i, win_rate, agent.epsilon))
    plt.plot(win_rate_list)
    plt.show()


