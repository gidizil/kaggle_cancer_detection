import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class BasicTwoLayerNet(nn.Module):
    def __init__(self):
        super(BasicTwoLayerNet).__init__()
        #TODO: Add config options
        self.conv1 = nn.conv2d(3, 16, 5)
        self.conv1_bn_2d = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(16, 128, 5)
        self.conv2_bn_2d = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 1. Conv part of network
        x = self.conv1(x)
        x = self.conv1_bn_2d(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.conv2_bn_2d(x)
        x = F.relu(X)
        x = self.pool(x)

        # 2. Feed Forward part of network
        x = x.view(-1, 128 * 5 * 5)
        x = F.relu(x)
        
