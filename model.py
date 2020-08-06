import torch
import torch.nn as nn
import torch.nn.functional as F

import config

import numpy as np
import math
from collections import OrderedDict

device = config.device

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


class DuelingDQN(nn.Module):
    def __init__(self, num_action, network_name):
        super(DuelingDQN, self).__init__()
        self.num_action = num_action
        input_channel = config.state_size[2]*config.stack_frame

        self.conv1 = nn.Conv2d(in_channels=input_channel, out_channels=32, kernel_size=8, stride=4, padding=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=1)

        self.fc1_a = nn.Linear(64*int(config.state_size[0]/8)*int(config.state_size[1]/8), 512)
        self.fc1_v = nn.Linear(64*int(config.state_size[0]/8)*int(config.state_size[1]/8), 512)

        self.fc2_a = nn.Linear(512, num_action)
        self.fc2_v = nn.Linear(512, 1)

    def forward(self, x):
        x = (x-(255.0/2))/(255.0/2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        x_a = F.relu(self.fc1_a(x))
        x_v = F.relu(self.fc1_v(x))

        # A stream : action advantage
        x_a = self.fc2_a(x_a) # [bs, num_action]
        x_a_mean = x_a.mean(dim=1).unsqueeze(1) # [bs, 1]
        x_a = x_a - x_a_mean.repeat(1, self.num_action) # [bs, num_action]

        # V stream : state value
        x_v = self.fc2_v(x_v) # [bs, 1]
        x_v = x_v.repeat(1, self.num_action) # [bs, num_action]

        out = x_a + x_v # [bs, num_action]
        return out

class NoisyLinearHay(nn.Module):
    def __init__(self, n_in, n_out, use_cuda=True):
        super(NoisyLinearHay, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.use_cuda = use_cuda

        self.w_mu  = -np.sqrt(3/n_out) + torch.rand(n_out, n_in, requires_grad=True) * 2 * np.sqrt(3/n_out)
        self.w_sig = torch.ones((n_out, n_in), requires_grad=True) * 0.017

        self.b_mu  = -np.sqrt(3/n_out) + torch.rand(n_out, requires_grad=True) * 2 * np.sqrt(3/n_out)
        self.b_sig = torch.ones((n_out), requires_grad=True) * 0.017

        if use_cuda:
            self.w_mu = self.w_mu.cuda()
            self.w_sig = self.w_sig.cuda()
            self.b_mu = self.b_mu.cuda()
            self.b_sig = self.b_sig.cuda()

        self.w_mu = nn.Parameter(self.w_mu)
        self.w_sig = nn.Parameter(self.w_sig)
        self.b_mu = nn.Parameter(self.b_mu)
        self.b_sig = nn.Parameter(self.b_sig)

    def forward(self, x, train):
        if self.use_cuda:
            w_eps = torch.randn((self.n_out, self.n_in)).cuda()
            b_eps = torch.randn((self.n_out)).cuda()
        else:
            w_eps = torch.randn((self.n_out, self.n_in))
            b_eps = torch.randn((self.n_out))

        if train:
            w = self.w_mu + F.relu(self.w_sig) * w_eps
            b = self.b_mu + F.relu(self.b_sig) * b_eps
        else:
            w = self.w_mu
            b = self.b_mu

        return F.linear(x, w, b)

class NoisyDQNHay(nn.Module):
    def __init__(self, num_action, use_cuda=True):
        super(NoisyDQNHay, self).__init__()
        self.use_cuda = use_cuda

        input_channel = config.state_size[2] * config.stack_frame
        self.conv1 = nn.Conv2d(in_channels=input_channel, out_channels=32, kernel_size=8, stride=4, padding=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=1)

        self.linear1 = NoisyLinearHay(
            n_in=64*int(config.state_size[0]/8)*int(config.state_size[1]/8),
            n_out=512,
            use_cuda=use_cuda
        )
        self.linear2 = NoisyLinearHay(n_in=512, n_out=num_action, use_cuda=use_cuda)

        if use_cuda:
            self.conv1.cuda()
            self.conv2.cuda()
            self.conv3.cuda()

    def forward(self, x, train=True):
        x = (x-(255.0/2))/(255.0/2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        # noisy linear
        x = F.relu(self.linear1(x, train))

        return self.linear2(x, train)

class ICM(nn.Module):
    def __init__(self, num_action, name):
        super(ICM, self).__init__()
        input_channel = config.state_size[2]*config.stack_frame
        self.conv1 = nn.Conv2d(in_channels=input_channel, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1)

        # forward model
        self.fc1_fm = nn.Linear(32*int(config.state_size[0]/16)*int(config.state_size[1]/16) + 1, 512)
        self.fc2_fm = nn.Linear(512 + 1, 32*int(config.state_size[0]/16)*int(config.state_size[1]/16))

        # inverse model
        self.fc1_im = nn.Linear(2*32*int(config.state_size[0]/16)*int(config.state_size[1]/16), 256)
        self.fc2_im = nn.Linear(256, num_action)

    def forward(self, x_now, x_next, a_now):
        # current state
        x_now = (x_now-(255.0/2))/(255.0/2)
        x_now = F.elu(self.conv1(x_now))
        x_now = F.elu(self.conv2(x_now))
        x_now = F.elu(self.conv3(x_now))
        x_now = F.elu(self.conv4(x_now))
        x_now_encode = x_now.view(-1, 32*int(config.state_size[0]/16)*int(config.state_size[1]/16)) # encoding vector of current state

        # next state
        x_next = (x_next-(255.0/2))/(255.0/2)
        x_next = F.elu(self.conv1(x_next))
        x_next = F.elu(self.conv2(x_next))
        x_next = F.elu(self.conv3(x_next))
        x_next = F.elu(self.conv4(x_next))
        x_next_encode = x_next.view(-1, 32*int(config.state_size[0]/16)*int(config.state_size[1]/16)) # encoding vector of next state

        # forward model
        x_fm = torch.cat([x_now_encode, a_now.unsqueeze(1)], dim=1)
        x_fm = F.relu(self.fc1_fm(x_fm))
        x_fm = torch.cat([x_fm, a_now.unsqueeze(1)], dim=1)
        x_fm = self.fc2_fm(x_fm) # predicted encoding vector of next state

        # inverse model
        x_encode = torch.cat([x_now_encode, x_next_encode], dim=1)
        x_im = F.relu(self.fc1_im(x_encode))
        x_im = F.softmax(self.fc2_im(x_im), dim=1) # (bs, 3)

        return x_next_encode, x_fm, x_im


class RND(nn.Module):
    def __init__(self, num_action, name):
        super(RND, self).__init__()
        input_channel = config.state_size[2]*config.stack_frame

        self.model_active = nn.Sequential(OrderedDict([
            ('conv1' , nn.Conv2d(in_channels=input_channel, out_channels=32, kernel_size=4, stride=2, padding=1)),
            ('activ1', nn.ELU()),
            ('conv2' , nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1)),
            ('activ2', nn.ELU()),
            ('conv3' , nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1)),
            ('activ3', nn.ELU()),
            ('conv4' , nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1)),
            ('activ4', nn.ELU()),
        ]))

        self.model_frozen = nn.Sequential(OrderedDict([
            ('conv1' , nn.Conv2d(in_channels=input_channel, out_channels=32, kernel_size=4, stride=2, padding=1)),
            ('activ1', nn.ELU()),
            ('conv2' , nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1)),
            ('activ1', nn.ELU()),
            ('conv3' , nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1)),
            ('activ1', nn.ELU()),
            ('conv4' , nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1)),
            ('activ1', nn.ELU()),
        ]))

    def forward(self, x):
        # current state
        x_next = (x-(255.0/2))/(255.0/2)
        x_next = self.model_active(x_next)
        x_next_encode = x_next.view(-1, 32*int(config.state_size[0]/16)*int(config.state_size[1]/16)) # encoding vector of next state

        x_next_t = (x-(255.0/2))/(255.0/2)
        x_next_t = self.model_frozen(x_next_t)
        x_next_encode_t = x_next_t.view(-1, 32*int(config.state_size[0]/16)*int(config.state_size[1]/16)) # predicted encoding vector of next state

        return x_next_encode, x_next_encode_t


class Actor(nn.Module):
    def __init__(self, num_action, name):
        super(Actor, self).__init__()
        input_size = config.state_size
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        policy = F.tanh(self.fc3(x))
        return policy

class Critic(nn.Module):
    def __init__(self, num_action, name):
        super(Critic, self).__init__()
        input_size = config.state_size
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128 + num_action, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x, a):
        x = F.relu(self.fc1(x))
        x = torch.cat([x, a.squeeze(1)], dim=1)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.fc4(x)
        return q_value

class ActorSAC(nn.Module):
    def __init__(self, num_action, name):
        super(ActorSAC, self).__init__()
        input_size = config.state_size
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3_mu = nn.Linear(128, num_action)
        self.fc3_std = nn.Linear(128, num_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.fc3_mu(x)
        ln_std = self.fc3_std(x)
        std = ln_std.exp()
        return mu, std

class CriticSAC(nn.Module):
    def __init__(self, num_action, name):
        super(CriticSAC, self).__init__()
        input_size = config.state_size
        # Q1 architecture
        self.fc1_Q1 = nn.Linear(input_size + num_action, 128)
        self.fc2_Q1 = nn.Linear(128, 128)
        self.fc3_Q1 = nn.Linear(128, 1)

        # Q2 architecture
        self.fc1_Q2 = nn.Linear(input_size + num_action, 128)
        self.fc2_Q2 = nn.Linear(128, 128)
        self.fc3_Q2 = nn.Linear(128, 1)

    def forward(self, x, a):
        x = torch.cat([x, a.squeeze(1)], dim=1)

        x1 = F.relu(self.fc1_Q1(x))
        x1 = F.relu(self.fc2_Q1(x1))
        q1 = F.relu(self.fc3_Q1(x1))

        x2 = F.relu(self.fc1_Q2(x))
        x2 = F.relu(self.fc2_Q2(x2))
        q2 = F.relu(self.fc3_Q2(x2))

        return q1, q2
