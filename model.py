import torch
import torch.nn as nn
import torch.nn.functional as F

import config

import numpy as np
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, num_action, network_name):
        super(DQN, self).__init__()
        input_channel = config.state_size[2]*config.stack_frame
        self.conv1 = nn.Conv2d(in_channels=input_channel, out_channels=32, kernel_size=8, stride=4, padding=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64*int(config.state_size[0]/8)*int(config.state_size[1]/8), 512)
        self.fc2 = nn.Linear(512, num_action)

    def forward(self, x):
        x = (x-(255.0/2))/(255.0/2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(-1, 64*int(config.state_size[0]/8)*int(config.state_size[1]/8))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DuelingDQN(nn.Module):
    def __init__(self, num_action, network_name):
        super(DuelingDQN, self).__init__()
        self.num_action = num_action
        input_channel = config.state_size[2]*config.stack_frame
        self.conv1 = nn.Conv2d(in_channels=input_channel, out_channels=32, kernel_size=8, stride=4, padding=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=1)
        self.fc1 = nn.Linear(64*int(config.state_size[0]/8)*int(config.state_size[1]/8), 512)
        self.fc2_a = nn.Linear(512, num_action)
        self.fc2_v = nn.Linear(512, 1)

    def forward(self, x):
        x = (x-(255.0/2))/(255.0/2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        # A stream : action advantage
        x_a = self.fc2_a(x) # [bs, 4]
        x_a_mean = x_a.mean(dim=1).unsqueeze(1) # [bs, 1]
        x_a = x_a - x_a_mean.repeat(1, self.num_action) # [bs, 4]
        # V stream : state value
        x_v = self.fc2_v(x) # [bs, 1]
        x_v = x_v.repeat(1, self.num_action) # [bs, 4]
        out = x_a + x_v # [bs, 4]
        return out


class NoisyDQN(nn.Module):
    def __init__(self, num_action, model_name, device):
        super(NoisyDQN, self).__init__()
        input_channel = config.state_size[2]*config.stack_frame
        self.conv1 = nn.Conv2d(in_channels=input_channel, out_channels=32, kernel_size=8, stride=4, padding=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=1)
        self.noisylinear1 = NoisyLinear(64*int(config.state_size[0]/8)*int(config.state_size[1]/8), 512, device)
        self.noisylinear2 = NoisyLinear(512, num_action, device)

    def forward(self, x, is_train):
        x = (x-(255.0/2))/(255.0/2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x)) # [bs:32, 64, 10, 10]
        x = x.view(x.size(0), -1) # [32, 6400]
        x = F.relu(self.noisylinear1(x, is_train))
        out = self.noisylinear2(x, is_train)
        return out

class NoisyLinear(nn.Module):
    def __init__(self, in_nl, out_nl):
        super(NoisyLinear, self).__init__()
        self.in_nl = in_nl
        self.out_nl = out_nl
        self.std_init = 0.017

        # weight related variables
        self.w_mu = nn.Parameter(torch.empty(self.out_nl, self.in_nl)).to(device)
        self.w_sigma = nn.Parameter(torch.empty(self.out_nl, self.in_nl)).to(device)
        self.w_eps = torch.empty(self.out_nl, self.in_nl).to(device)

        # bias related variables
        self.b_mu = nn.Parameter(torch.empty(self.out_nl)).to(device)
        self.b_sigma = nn.Parameter(torch.empty(self.out_nl)).to(device)
        self.b_eps = torch.empty(self.out_nl).to(device)

        self.init_params()
        self.init_noise()

    def forward(self, x, is_train):
        self.init_noise()
        # print(self.w_sigma.data[0][0])

        if is_train:
            w = self.w_mu.data + torch.mul(self.w_sigma.data, self.w_eps)
            b = self.b_mu.data + torch.mul(self.b_sigma.data, self.b_eps)
        else:
            w = self.w_mu.data
            b = self.b_mu.data
        return F.linear(x, w, b)


    def init_params(self):
        mu_dist_range = math.sqrt(3/self.in_nl)
        self.w_mu.data.uniform_(-mu_dist_range, mu_dist_range)
        self.b_mu.data.uniform_(-mu_dist_range, mu_dist_range)
        self.w_sigma.data.fill_(self.std_init)
        self.b_sigma.data.fill_(self.std_init)

    def init_noise(self):
        self.w_eps = torch.normal(mean=0.0, std=1.0, size=self.w_mu.size()).to(device)
        self.b_pes = torch.normal(mean=0.0, std=1.0, size=self.b_mu.size()).to(device)


#
# class NoisyDQN(nn.Module):
#     def __init__(self, num_action, network_name):
#         super(NoisyDQN, self).__init__()
#         input_channel = config.state_size[2]*config.stack_frame
#         self.conv1 = nn.Conv2d(in_channels=input_channel, out_channels=32, kernel_size=8, stride=4, padding=4)
#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=2)
#         self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=1)
#
#         flat_size = 64*int(config.state_size[0]/8)*int(config.state_size[1]/8)
#         self.mu_w1 = nn.init.uniform_(torch.empty(flat_size, 512), -np.sqrt(3/flat_size), np.sqrt(3/flat_size)).to(device)
#         self.sig_w1 = nn.init.constant_(torch.empty(flat_size, 512), 0.017).to(device)
#         self.mu_b1 = nn.init.uniform_(torch.empty(512), -np.sqrt(3/flat_size), np.sqrt(3/flat_size)).to(device)
#         self.sig_b1 = nn.init.constant_(torch.empty(512), 0.017).to(device)
#
#         self.mu_w2= nn.init.uniform_(torch.empty(512, num_action), -np.sqrt(3/512), np.sqrt(3/512)).to(device)
#         self.sig_w2 = nn.init.constant_(torch.empty(512, num_action), 0.017).to(device)
#         self.mu_b2 = nn.init.uniform_(torch.empty(num_action), -np.sqrt(3/512), np.sqrt(3/512)).to(device)
#         self.sig_b2 = nn.init.constant_(torch.empty(num_action), 0.017).to(device)
#
#     def forward(self, x, is_train):
#         x = (x-(255.0/2))/(255.0/2)
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = x.view(-1, 64*int(config.state_size[0]/8)*int(config.state_size[1]/8))
#
#         x = F.relu(self.noisy_dense(x, self.mu_w1, self.sig_w1, self.mu_b1, self.sig_b1, is_train))
#         x = self.noisy_dense(x, self.mu_w2, self.sig_w2, self.mu_b2, self.sig_b2, is_train)
#         return x
#
#     def noisy_dense(self, x, mu_w, sig_w, mu_b, sig_b, is_train):
#         if is_train:
#             eps_w = torch.randn(sig_w.size()).to(device)
#             eps_b = torch.randn(sig_b.size()).to(device)
#         else:
#             eps_w = torch.zeros_like(sig_w).to(device)
#             eps_b = torch.zeros_like(sig_b).to(device)
#
#         w_fc = mu_w + (sig_w * eps_w)
#         b_fc = mu_b + (sig_b * eps_b)
#
#         return torch.matmul(x, w_fc) + b_fc
