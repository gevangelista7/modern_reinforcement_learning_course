import numpy as np
import gym
import matplotlib.pyplot as plt

"""

"SFFF" 
"FHFH" 
"FFFH" 
"HFFG"

left:   0
down:   1
right:  2
up:     3
"""
policy = {0: 1, 1: 2, 2: 1, 3: 0, 4: 1, 6: 1, 8: 2, 9: 1, 10: 1, 13: 2, 14: 2}

env = gym.make('FrozenLake-v1')
done = False
state = env.reset()
actions = []
win_rate_list = []
scores = []
score = 0
N_GAMES = 100000
WIN_RATE_UPDATE = 100

if __name__ == "__main__":
    for i in range(N_GAMES):
        score = 0
        state = env.reset()
        done = False

        while not done:
            action = policy[state]
            state, reward, done, _ = env.step(action)
            actions.append(action)
            score += reward

            scores.append(score)

        if i % WIN_RATE_UPDATE == 0:
            win_rate_list.append(np.mean(scores[-WIN_RATE_UPDATE:]))

    print("win rate: ", win_rate_list)
    print(max(win_rate_list))
    plt.plot(win_rate_list)
    plt.show()

