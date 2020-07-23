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
            w = self.w_mu + self.w_sig * w_eps
            b = self.b_mu + self.b_sig * b_eps
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


def init(module, weight_init, bias_init, gain=1, mode=None, nonlinearity='relu'):
    if mode is not None:
        weight_init(module.weight.data, mode=mode, nonlinearity=nonlinearity)
    else:
        weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class C51(nn.Module):
    def __init__(self, num_inputs, hidden_size=512, num_actions=4,
                       atoms=51, vmin=-10, vmax=10, use_cuda=True):
        super(C51, self).__init__()
        self.atoms = atoms
        self.vmin  = vmin
        self.vmax  = vmax
        self.delta_z = (vmax - vmin) / (atoms - 1)
        self.num_actions   = num_actions
        self.use_cuda = use_cuda


        # init_ = lambda m: init(m,
        #                        nn.init.kaiming_uniform_,
        #                        lambda x: nn.init.constant_(x, 0),
        #                        nonlinearity='relu',
        #                        mode='fan_in')
        # init2_ = lambda m: init(m,
        #                        nn.init.kaiming_uniform_,
        #                        lambda x: nn.init.constant_(x, 0),
        #                        nonlinearity='relu',
        #                        mode='fan_in')
        #
        #
        # self.conv1 = init_(nn.Conv2d(num_inputs, 32, 8, stride=4))
        # self.conv2 = init_(nn.Conv2d(32, 64, 4, stride=2))
        # self.conv3 = init_(nn.Conv2d(64, 32, 3, stride=1))

        self.conv1 = nn.Conv2d(num_inputs, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=1)

        self.fc1 = nn.Linear(192,      hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_actions*atoms)

        if use_cuda:
            self.conv1.cuda()
            self.conv2.cuda()
            self.conv3.cuda()
            self.fc1.cuda()
            self.fc2.cuda()


    def forward(self, x):
        x = (x-(255.0/2))/(255.0/2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x_batch = x.view(-1, self.num_actions, self.atoms)
        y = F.log_softmax(x_batch, dim=2).exp() # y is of shape [batch x action x atoms]

        return y
