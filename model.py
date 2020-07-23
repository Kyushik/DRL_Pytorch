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
        w_eps = torch.randn((self.n_out, self.n_in))
        b_eps = torch.randn((self.n_out))

        if self.use_cuda:
            w_eps = w_eps.cuda()
            b_eps = b_eps.cuda()

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
#
#
# class NoisyDQN(nn.Module):
#     def __init__(self, num_action, model_name):
#         super(NoisyDQN, self).__init__()
#         input_channel = config.state_size[2]*config.stack_frame
#         self.conv1 = nn.Conv2d(in_channels=input_channel, out_channels=32, kernel_size=8, stride=4, padding=4)
#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=2)
#         self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=1)
#         self.noisylinear1 = NoisyLinear(64*int(config.state_size[0]/8)*int(config.state_size[1]/8), 512)
#         self.noisylinear2 = NoisyLinear(512, num_action)
#
#     def forward(self, x, is_train):
#         x = (x-(255.0/2))/(255.0/2)
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x)) # [bs:32, 64, 10, 10]
#         x = x.view(x.size(0), -1) # [32, 6400]
#         x = F.relu(self.noisylinear1(x, is_train))
#         out = self.noisylinear2(x, is_train)
#         return out
# class NoisyLinear(nn.Module):
#     def __init__(self, in_nl, out_nl):
#         super(NoisyLinear, self).__init__()
#         self.in_nl = in_nl
#         self.out_nl = out_nl
#         self.std_init = 0.017
#
#         # weight related variables
#         self.w_mu = nn.Parameter(torch.empty(self.out_nl, self.in_nl), requires_grad=True).to(device)
#         self.w_sigma = nn.Parameter(torch.empty(self.out_nl, self.in_nl), requires_grad=True).to(device)
#         self.w_eps = torch.empty(self.out_nl, self.in_nl).to(device)
#
#         # bias related variables
#         self.b_mu = nn.Parameter(torch.empty(self.out_nl), requires_grad=True).to(device)
#         self.b_sigma = nn.Parameter(torch.empty(self.out_nl), requires_grad=True).to(device)
#         self.b_eps = torch.empty(self.out_nl).to(device)
#
#         self.init_params()
#         self.init_noise()
#
#     def forward(self, x, is_train):
#         self.init_noise()
#         # print(self.w_sigma.data[0][0])
#
#         if is_train:
#             w = self.w_mu.data + torch.mul(self.w_sigma.data, self.w_eps)
#             b = self.b_mu.data + torch.mul(self.b_sigma.data, self.b_eps)
#         else:
#             w = self.w_mu.data
#             b = self.b_mu.data
#         return F.linear(x, w, b)
#
#     def init_params(self):
#         mu_dist_range = math.sqrt(3/self.in_nl)
#         self.w_mu.data.uniform_(-mu_dist_range, mu_dist_range)
#         self.b_mu.data.uniform_(-mu_dist_range, mu_dist_range)
#         self.w_sigma.data.fill_(self.std_init)
#         self.b_sigma.data.fill_(self.std_init)
#
#     def init_noise(self):
#         self.w_eps = torch.normal(mean=0.0, std=1.0, size=self.w_mu.size()).to(device)
#         self.b_pes = torch.normal(mean=0.0, std=1.0, size=self.b_mu.size()).to(device)


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
