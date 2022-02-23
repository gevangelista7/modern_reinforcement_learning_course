import numpy as np
import numpy.random as rd


class Agent:
    def __init__(self, lr, gamma, n_actions, n_states, eps_start, eps_end, eps_decay):
        self.lr = lr

        self.eps_start = eps_start
        self.epsilon = self.eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.n_actions = n_actions
        self.n_states = n_states
        self.gamma = gamma
        self.Q = {}

        self.Q_init()

    def Q_init(self):
        for state in range(self.n_states):
            for action in range(self.n_actions):
                self.Q[(state, action)] = 0.0

    def decrement_epsilon(self):
        self.epsilon = max(self.epsilon * self.eps_decay, self.eps_end)

    def choose_action(self, state):
        if rd.rand() < self.epsilon:
            action = rd.choice(range(self.n_actions))
        else:
            action = self.best_action(state)

        return action

    def best_action(self, state):
        actions = np.array([self.Q[state, i] for i in range(self.n_actions)])
        return np.argmax(actions)

    def learn(self, state, action, next_state, reward):
        best_next_action = self.best_action(next_state)

        self.Q[(state, action)] = (1 - self.lr) * self.Q[(state, action)] + self.lr * (
                    reward + self.gamma * self.Q[(next_state, best_next_action)])

        self.decrement_epsilon()


# agt = Agent(lr=0.001, gamma=0.9, eps_end=0.01, eps_start=1, eps_decay=0.99999, n_actions=4, n_states=16)
# agt.best_action(5), agt.choose_action(3)