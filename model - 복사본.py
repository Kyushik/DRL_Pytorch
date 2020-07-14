# Model
import math, random

import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, model_name, action_size, in_channel):
        super(DQN, self).__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=32, kernel_size=8, stride=4, padding=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=1),
            nn.ReLU()
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(64*10*10, 512),
            nn.ReLU(),
            nn.Linear(512, action_size)
        )

    def forward(self, x):
        x = (x-(255.0/2)) / (255.0/2)
        x = self.conv_layer(x) # [32, 64, 10, 10]
        x = x.view(x.size(0), -1) # [32, 6400]
        out = self.fc_layer(x)
        return out



class DuelingDQN(nn.Module):
    def __init__(self, model_name, action_size, in_channel):
        super(DuelingDQN, self).__init__()
        self.action_size = action_size
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=32, kernel_size=8, stride=4, padding=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=1),
            nn.ReLU()
        )
        # A stream : action advantage
        self.fc_layer_a = nn.Sequential(
            nn.Linear(64*10*10, 512),
            nn.ReLU(),
            nn.Linear(512, self.action_size)
        )
        # V stream : state value
        self.fc_layer_v = nn.Sequential(
            nn.Linear(64*10*10, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = (x-(255.0/2)) / (255.0/2)
        x = self.conv_layer(x) # [32, 64, 10, 10]
        x = x.view(x.size(0), -1) # [32, 6400]
        out_a = self.fc_layer_a(x) # [32, 4]
        out_a = out_a - out_a.mean(dim=1) # [32, 4]

        out_v = self.fc_layer_v(x) # [32, 1]
        out_v = out_v.repeat(1, self.action_size) # [32, 4]

        out = out_a + out_v # [32, 4]
        return out

# 
# class DQN(nn.Module):
#     def __init__(self, model_name, action_size, in_channel):
#         super(DQN, self).__init__()
#
#         self.conv_layer = nn.Sequential(
#             nn.Conv2d(in_channels=in_channel, out_channels=32, kernel_size=8, stride=4, padding=4),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=2),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=1),
#             nn.ReLU()
#         )
#
#         self.fc_layer = nn.Sequential(
#             nn.Linear(64*10*10, 512),
#             nn.ReLU(),
#             nn.Linear(512, action_size)
#         )
#
#     def forward(self, x):
#         x = (x-(255.0/2)) / (255.0/2)
#         x = self.conv_layer(x) # [32, 64, 10, 10]
#         x = x.view(x.size(0), -1) # [32, 6400]
#         out = self.fc_layer(x)
#         return out
#
#
#
# class DuelingDQN(nn.Module):
#     def __init__(self, model_name, action_size, in_channel):
#         super(DuelingDQN, self).__init__()
#         self.action_size = action_size
#         self.conv_layer = nn.Sequential(
#             nn.Conv2d(in_channels=in_channel, out_channels=32, kernel_size=8, stride=4, padding=4),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=2),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=1),
#             nn.ReLU()
#         )
#         # A stream : action advantage
#         self.fc_layer_a = nn.Sequential(
#             nn.Linear(64*10*10, 512),
#             nn.ReLU(),
#             nn.Linear(512, self.action_size)
#         )
#         # V stream : state value
#         self.fc_layer_v = nn.Sequential(
#             nn.Linear(64*10*10, 512),
#             nn.ReLU(),
#             nn.Linear(512, 1)
#         )
#
#     def forward(self, x):
#         x = (x-(255.0/2)) / (255.0/2)
#         x = self.conv_layer(x) # [32, 64, 10, 10]
#         x = x.view(x.size(0), -1) # [32, 6400]
#         out_a = self.fc_layer_a(x) # [32, 4]
#         out_a = out_a - out_a.mean(dim=1) # [32, 4]
#
#         out_v = self.fc_layer_v(x) # [32, 1]
#         out_v = out_v.repeat(1, self.action_size) # [32, 4]
#
#         out = out_a + out_v # [32, 4]
#         return out
#
#
# class NoisyDQN(nn.Module):
#     def __init__(self, model_name, action_size, in_channel):
#         super(NoisyDQN, self).__init__()
#
#         self.conv_layer = nn.Sequential(
#             nn.Conv2d(in_channels=in_channel, out_channels=32, kernel_size=8, stride=4, padding=4),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=2),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=1),
#             nn.ReLU()
#         )
#
#
#     def forward(self, x):
#         x = (x-(255.0/2)) / (255.0/2)
#         x = self.conv_layer(x) # [32, 64, 10, 10]
#         x = x.view(x.size(0), -1) # [32, 6400]
#         out = self.NoisyLinear(x)
#         return out
#
# class NoisyLinear(nn.Module):
#     def __init__(self, in, out, std_init):
#         super(NoisyLienar, self).__init__()
