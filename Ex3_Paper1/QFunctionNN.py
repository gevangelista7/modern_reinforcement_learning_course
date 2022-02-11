import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch


class QFunctionNN(nn.Module):
    def __init__(self, lr, state_shape, actions_n):
        super(QFunctionNN, self).__init__()
        self.state_shape = state_shape
        self.actions_n = actions_n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=state_shape[-1], out_channels=32, kernel_size=(8, 8), stride=(4, 4)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU()
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64, 512),
            nn.ReLU(),
            nn.Linear(512, actions_n),
            nn.ReLU()
        )

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.to(self.device)

    def size_conv_out(self):
        zeros = torch.zeros(self.state_shape)
        conv_out = self.conv_layers(zeros)
        return conv_out

    def forward(self, state_action):
        layer1 = F.relu(self.fc1(state_action))
        out = self.fc3(layer1)
        return out
