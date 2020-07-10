import torch 
import torch.nn as nn
import torch.nn.functional as F

import config

class DQN_net(nn.Module):
    def __init__(self, num_action, network_name):
        super(DQN_net, self).__init__()
        input_channel = config.state_size[2]*config.stack_frame
        self.conv1 = nn.Conv2d(in_channels=input_channel, out_channels=32, kernel_size=8, stride=4, padding=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=1)
        self.fc1 = nn.Linear(64*int(config.state_size[0]/8)*int(config.state_size[1]/8), 512)
        self.fc2 = nn.Linear(512, num_action)

    def forward(self, x):
        x = (x-(255.0/2))/(255.0/2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64*int(config.state_size[0]/8)*int(config.state_size[1]/8))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

