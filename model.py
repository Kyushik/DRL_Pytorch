import torch 
import torch.nn as nn
import torch.nn.functional as F

import config

import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, num_action, network_name):
        super(DQN, self).__init__()
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

class NoisyDQN(nn.Module):
    def __init__(self, num_action, network_name):
        super(NoisyDQN, self).__init__()
        input_channel = config.state_size[2]*config.stack_frame
        self.conv1 = nn.Conv2d(in_channels=input_channel, out_channels=32, kernel_size=8, stride=4, padding=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=1)
        
        flat_size = 64*int(config.state_size[0]/8)*int(config.state_size[1]/8)
        self.mu_w1 = nn.init.uniform_(torch.empty(flat_size, 512), -np.sqrt(3/flat_size), np.sqrt(3/flat_size)).to(device)
        self.sig_w1 = nn.init.constant_(torch.empty(flat_size, 512), 0.017).to(device)
        self.mu_b1 = nn.init.uniform_(torch.empty(512), -np.sqrt(3/flat_size), np.sqrt(3/flat_size)).to(device)
        self.sig_b1 = nn.init.constant_(torch.empty(512), 0.017).to(device)
        
        self.mu_w2= nn.init.uniform_(torch.empty(512, num_action), -np.sqrt(3/512), np.sqrt(3/512)).to(device)
        self.sig_w2 = nn.init.constant_(torch.empty(512, num_action), 0.017).to(device)
        self.mu_b2 = nn.init.uniform_(torch.empty(num_action), -np.sqrt(3/512), np.sqrt(3/512)).to(device)
        self.sig_b2 = nn.init.constant_(torch.empty(num_action), 0.017).to(device)

    def forward(self, x, is_train):
        x = (x-(255.0/2))/(255.0/2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64*int(config.state_size[0]/8)*int(config.state_size[1]/8))
        
        x = F.relu(self.noisy_dense(x, self.mu_w1, self.sig_w1, self.mu_b1, self.sig_b1, is_train))
        x = self.noisy_dense(x, self.mu_w2, self.sig_w2, self.mu_b2, self.sig_b2, is_train)
        return x

    def noisy_dense(self, x, mu_w, sig_w, mu_b, sig_b, is_train):
        if is_train:
            eps_w = torch.randn(sig_w.size()).to(device)
            eps_b = torch.randn(sig_b.size()).to(device)
        else:
            eps_w = torch.zeros_like(sig_w).to(device)
            eps_b = torch.zeros_like(sig_b).to(device)

        w_fc = mu_w + (sig_w * eps_w)
        b_fc = mu_b + (sig_b * eps_b)

        return torch.matmul(x, w_fc) + b_fc

        



