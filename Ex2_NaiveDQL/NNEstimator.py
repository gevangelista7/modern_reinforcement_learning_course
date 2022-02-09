import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch


class NNEstimator(nn.Module):
    def __init__(self, lr, state_dim, actions_n):
        super(NNEstimator, self).__init__()

        self.fc1 = nn.Linear(*state_dim, 128)
        self.fc3 = nn.Linear(128, actions_n)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state_action):
        layer1 = F.relu(self.fc1(state_action))
        out = self.fc3(layer1)
        return out
