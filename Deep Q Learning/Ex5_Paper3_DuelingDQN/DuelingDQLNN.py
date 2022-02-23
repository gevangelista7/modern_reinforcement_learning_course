import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import os


class DuelingDQLNN(nn.Module):
    def __init__(self, lr, state_shape, actions_n, batch_size , name="model-v0.pt", checkpoint_dir="./saved_models"):
        super(DuelingDQLNN, self).__init__()
        self.state_shape = state_shape
        self.actions_n = actions_n
        self.batch_size = batch_size

        self.conv_in_channels = state_shape[0]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.conv1 = nn.Conv2d(in_channels=self.conv_in_channels, out_channels=32, kernel_size=(8, 8), stride=(4, 4)).to(self.device)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(2, 2)).to(self.device)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1)).to(self.device)

        self.fc_input_size = self.get_fc_input_size()

        self.fcl1_v = nn.Linear(self.fc_input_size, 512).to(self.device)
        self.fcl2_v = nn.Linear(512, 1).to(self.device)

        self.fcl1_a = nn.Linear(self.fc_input_size, 512).to(self.device)
        self.fcl2_a = nn.Linear(512, actions_n).to(self.device)

        self.optimizer = optim.RMSprop(params=self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.to(self.device)

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

    def get_fc_input_size(self):
        zrs = torch.zeros(1, *self.state_shape).to(self.device)
        conv_out = self.conv(zrs)
        # conv_out = conv_out.view(conv_out.size()[0], -1)
        return int(np.prod(conv_out.shape))

    def conv(self, state):
        layer1 = F.relu(self.conv1(state))
        layer2 = F.relu(self.conv2(layer1))
        layer3 = F.relu(self.conv3(layer2))
        return layer3

    def value_stream(self, conv_out):
        layer4_v = F.relu((self.fcl1_v(conv_out)))
        out_layer = self.fcl2_v(layer4_v)
        return out_layer

    def advantage_stream(self, conv_out):
        layer4_a = F.relu((self.fcl1_a(conv_out)))
        out_layer = self.fcl2_a(layer4_a)
        return out_layer

    def forward(self, state):
        convoluted = F.relu(self.conv(state))
        convoluted_f = convoluted.view(convoluted.size()[0], -1)
        advantage = self.advantage_stream(convoluted_f)
        value = self.value_stream(convoluted_f)
        return value, advantage

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(self.checkpoint_file)
