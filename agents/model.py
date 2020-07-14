# Model
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
# import torch.nn.functional as F

class DQNModel(nn.Module):
    def __init__(self, in_channel, model_name, action_size):
        super(DQNModel, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.conv2d(in_channels=in_channel, out_channels=4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.conv2d(4, 8, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(4*4*8, action_size)
        )

    def forward(self, x):
        out = (x-(255.0/2)) / (255.0/2)
        out = self.conv_layer(out)
        out = out.view(out.size(0), -1)
        out = self.fc_layer(out)
        return out
