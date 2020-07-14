# Model
import torch
import torch.nn as nn
import torch.nn.functional as F

import math, random

class DQN(nn.Module):
    def __init__(self, model_name, action_size, in_channel):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=32, kernel_size=8, stride=4, padding=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=1)
        self.fc1   = nn.Linear(64*10*10, 512)
        self.fc2   = nn.Linear(512, action_size)

    def forward(self, x):
        x = (x-(255.0/2)) / (255.0/2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x)) # [bs:32, 64, 10, 10]
        x = x.view(x.size(0), -1) # [32, 6400]
        x = F.relu(self.fc1(x))
        out = self.fc2(x)
        return out


class DuelingDQN(nn.Module):
    def __init__(self, model_name, action_size, in_channel):
        super(DuelingDQN, self).__init__()
        self.action_size = action_size
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=32, kernel_size=8, stride=4, padding=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=1)
        self.fc1   = nn.Linear(64*10*10, 512)
        self.fc2_a = nn.Linear(512, action_size)
        self.fc2_v = nn.Linear(512, 1)

    def forward(self, x):
        x = (x-(255.0/2)) / (255.0/2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x)) # [bs:32, 64, 10, 10]
        x = x.view(x.size(0), -1) # [32, 6400]
        x = F.relu(self.fc1(x))
        # A stream : action advantage
        x_a = self.fc2_a(x) # [32, 4]
        x_a_mean = x_a.mean(dim=1).unsqueeze(1) # [32, 1]
        x_a = x_a - x_a_mean.repeat(1, self.action_size) # [32, 4]
        # V stream : state value
        x_v = self.fc2_v(x) # [32, 1]
        x_v = x_v.repeat(1, self.action_size) # [32, 4]
        out = x_a + x_v # [32, 4]
        return out

#
# class NoisyDQN(nn.Module):
#     def __init__(self, model_name, action_size, in_channel, train_mode):
#         super(NoisyDQN, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=32, kernel_size=8, stride=4, padding=4)
#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=2)
#         self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=1)
#         self.noisylinear1 = NoisyLinear(64*10*10, 512, train_mode)
#         self.noisylinear2 = NoisyLinear(512, action_size, train_mode)
#
#     def forward(self, x):
#         x = (x-(255.0/2)) / (255.0/2)
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x)) # [bs:32, 64, 10, 10]
#         x = x.view(x.size(0), -1) # [32, 6400]
#         x = F.relu(self.noisylinear1(x))
#         out = self.noisylinear2(x)
#         return out
#
# class NoisyLinear(nn.Module):
#     def __init__(self, in, out, train_mode, std_init=0.017):
#         super(NoisyLienar, self).__init__()
#         self.in = in
#         self.out = out
#         self.std_init = std_init
#         # weight parameters
#         self.w_mu = nn.Parameter(torch.FloatTensor(self.in, self.out))
#         self.w_sigma = nn.Parameter(torch.FloatTensor(self.in, self.out))
#         self.w_eps = torch.FloatTensor(self.in, self.out)
#         # bias parameters
#         self.b_mu = nn.Parameter(torch.FloatTensor(self.out))
#         self.b_sigma = nn.Parameter(torch.FloatTensor(self.out))
#         self.b_eps = torch.FloatTensor(self.out)
#
#     def forward(self, x):
#         out = x
#         return out
#
#     def init_param(self, mu, sigma, in):
#         self.mu.uniform_(-torch.sqrt(3/in.size()), torch.sqrt(3/in.size()))
#         self.sigma = self.std_init
