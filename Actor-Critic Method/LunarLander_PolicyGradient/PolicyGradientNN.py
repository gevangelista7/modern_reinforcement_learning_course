import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

class PolicyGradientNN():
    def __init__(self, lr, obs_shape, actions_n):
        super(PolicyGradientNN, self).__init__()
        self.obs_shape = obs_shape
        self.actions_n = actions_n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = nn.Sequential(
            nn.Linear(obs_shape, 128),
            nn.ReLU(),
            nn.Linear(128, actions_n)
        ).to(self.device)

        self.optimizer = optim.Adam(params=self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self. to(self.device)

    def forward(self, obs):
        return self.model.forward(obs)



