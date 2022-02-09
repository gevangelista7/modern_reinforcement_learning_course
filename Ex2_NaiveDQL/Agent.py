import numpy as np
import numpy.random as rd
import random
from NNEstimator import NNEstimator
import torch

# N_GAMES = 10000
# gamma =


class Agent():
    def __init__(self, lr, actions_n, state_dim, gamma=0.99,
                 eps_start=1.0, eps_end=0.01, eps_dec=1e-5):

        self.gamma = gamma
        self.action_n = actions_n
        self.state_dim = state_dim
        self.epsilon = eps_start
        self.eps_min = eps_end
        self.eps_dec = eps_dec

        self.Q = NNEstimator(lr=lr, actions_n=actions_n, state_dim=state_dim)
        self.action_space = [i for i in range(self.action_n)]

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min

    def choose_action(self, state):
        if rd.rand() < self.epsilon:
            action = rd.choice(range(self.action_n))
        else:
            action = self.best_action(state)

        return action

    def best_action(self, state):
        state_t = torch.tensor(state, dtype=torch.float32, device=self.Q.device)
        actions = self.Q.forward(state_t)
        action = actions.argmax().item()
        # est_rwd = actions[action].item()

        return action

    def learn(self, state, action, reward, next_state):
        self.zero_grad()

        state_t = torch.tensor(state, dtype=torch.float32, device=self.Q.device)
        next_state_t = torch.tensor(next_state, dtype=torch.float32, device=self.Q.device)
        reward_t = torch.tensor(reward, dtype=torch.float32, device=self.Q.device)

        q_pred = self.Q.forward(state_t)[action]
        q_next = self.Q.forward(next_state_t).max()

        tgt = reward_t + self.gamma * q_next
        loss = self.Q.loss(q_pred, tgt)

        loss.backward()
        self.step()
        self.decrement_epsilon()

    def zero_grad(self):
        self.Q.optimizer.zero_grad()

    def step(self):
        self.Q.optimizer.step()

    def forward(self, data):
        return self.Q.forward(data)

