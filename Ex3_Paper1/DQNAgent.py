import torch
import numpy.random as rd
from QFunctionNN import QFunctionNN
from copy import deepcopy


class DQNAgent:
    def __init__(self, lr, actions_n, state_shape, gamma=0.99,
                 eps_start=1.0, eps_end=0.01, eps_dec=1e-5):

        self.actions_n = actions_n
        self.state_shape = state_shape
        self.gamma = gamma
        self.epsilon = eps_start
        self.eps_dec = eps_dec
        self.eps_min = eps_end

        self.Q = QFunctionNN(lr=lr, state_shape=state_shape, actions_n=actions_n)
        self.Q_tgt = deepcopy(self.Q)

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def choose_action(self, state):
        if rd.rand() < self.epsilon:
            action = rd.choice(range(self.actions_n))
        else:
            action = self.best_action(state)

        return action

    def update_Q_tgt(self):
        self.Q_tgt = deepcopy(self.Q)
        # self.Q_tgt.load_state_dict(self.Q.state_dict()) # does work??

    def best_action(self, state):
        state_t = torch.tensor(state, dtype=torch.float32, device=self.Q.device)
        actions = self.Q.forward(state_t)
        action = actions.argmax().item()
        # est_rwd = actions[action].item()

        return action

    def save_model(self):
        self.Q.save_checkpoint()

    def load_model(self, ):
        self.Q.load_checkpoint()

    def learn(self, state, action, reward, next_state, done):
        self.Q.zero_grad()

        state_t = torch.tensor(state, dtype=torch.float32, device=self.Q.device)
        next_state_t = torch.tensor(next_state, dtype=torch.float32, device=self.Q.device)
        reward_t = torch.tensor(reward, dtype=torch.float32, device=self.Q.device)

        q_pred = self.Q.forward(state_t)[action]
        q_next = self.Q_tgt.forward(next_state_t).max()

        tgt = reward_t + self.gamma * q_next
        loss = self.Q.loss(q_pred, tgt)

        loss.backward()
        self.Q.optimizer.step()
        self.decrement_epsilon()

        self.save_model()
