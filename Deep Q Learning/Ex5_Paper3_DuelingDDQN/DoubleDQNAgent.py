import numpy as np
import torch
import numpy.random as rd
from QFunctionNN import QFunctionNN
from copy import deepcopy
from utils import ExperienceBuffer


class DoubleDQNAgent:
    def __init__(self, lr, actions_n, state_shape,
                 mem_size, batch_size,
                 gamma=0.99, q_update=1000,
                 eps_start=1.0, eps_end=0.01, eps_dec=1e-5):

        self.actions_n = actions_n
        self.state_shape = state_shape
        self.gamma = gamma
        self.epsilon = eps_start
        self.eps_dec = eps_dec
        self.eps_min = eps_end

        self.Q_eval = QFunctionNN(lr=lr, state_shape=state_shape, actions_n=actions_n, batch_size=batch_size)
        self.Q_tgt = QFunctionNN(lr=lr, state_shape=state_shape, actions_n=actions_n, batch_size=batch_size)

        self.memory = ExperienceBuffer(mem_size, state_shape)
        self.batch_size = batch_size
        self.learn_step_count = 0
        self.q_tgt_update_period = q_update

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def choose_action(self, state):
        if rd.rand() < self.epsilon:
            action = rd.choice(range(self.actions_n))
        else:
            action = self.best_action(state)

        return action

    def best_action(self, state):
        if len(state.shape) == 3:
            state = np.expand_dims(state, 0)

        state = torch.tensor(state, dtype=torch.float32, device=self.Q_eval.device)
        actions = self.Q_eval.forward(state)
        actions = actions.argmax(dim=1)

        return actions

    def update_Q_tgt(self):
        self.Q_tgt.load_state_dict(self.Q_eval.state_dict())

    def insert_memory(self, state, action, reward, state_, done):
        self.memory.insert(state, action, reward, state_, done)

    def sample_memory(self):
        states, actions, rewards, states_, dones = self.memory.sample(self.batch_size)

        states = torch.tensor(states, device=self.device)
        actions = torch.tensor(actions, device=self.device)
        rewards = torch.tensor(rewards, device=self.device)
        states_ = torch.tensor(states_, device=self.device)
        dones = torch.tensor(dones, device=self.device)

        return states, actions, rewards, states_, dones

    def save_model(self):
        print(" === === === SAVING CHECKPOINT === === ===")
        self.Q_eval.save_checkpoint()

    def load_model(self, ):
        print(" === === === LOADING CHECKPOINT === === ===")
        self.Q_eval.load_checkpoint()

    def learn(self):
        if self.memory.mem_idx < self.batch_size:
            # print("skip learning...")
            return

        self.Q_eval.optimizer.zero_grad()

        if self.learn_step_count % self.q_tgt_update_period == 0:
            self.update_Q_tgt()

        states, actions, rewards, states_, dones = self.sample_memory()
        index = np.arange(self.batch_size)

        q_pred = self.Q_eval.forward(states)[index, actions]
        best_actions_ = self.best_action(states_)
        q_next = self.Q_tgt.forward(states_)[index, best_actions_]
        # q_eval = self.Q_eval.forward(states_)
        # max_actions = torch.argmax(q_eval, dim=1)
        # q_next = self.Q_tgt.forward(states_)[index, max_actions]

        q_next[dones] = 0.0
        q_tgt = rewards + self.gamma * q_next

        loss = self.Q_eval.loss(q_tgt, q_pred).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.decrement_epsilon()
        self.learn_step_count += 1

