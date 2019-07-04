import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_utils import ModelUtils
from collections import OrderedDict


class Net(nn.Module, ModelUtils):
    def __init__(self, img_dims=None):

        net_dict = OrderedDict([('conv1', ((5, 5), )),
                                ('pool1', ((2, 2), 2)),
                                ('conv2', ((5, 5), )),
                                ('pool2', ((2, 2), 2))
                                ])
        # Inherit from parent classes
        super(Net, self).__init__()
        utils = ModelUtils(img_dims, net_dict)
        utils.get_final_feature_map_dims()
        self.final_f_map_dims = utils.final_f_maps_dims


        #TODO: Add config options
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.conv1_bn_2d = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 128, 5)
        self.conv2_bn_2d = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * self.final_f_map_dims, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1) # For bce loss only one value is needed


    def forward(self, x):
        # 1. Conv part of network
        x = self.conv1(x)
        x = self.conv1_bn_2d(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.conv2_bn_2d(x)
        x = F.relu(x)
        x = self.pool(x)

        # 2. Feed Forward part of network
        x = x.view(-1, 128 * self.final_f_map_dims)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.sigmoid(x) # F.sigmoid is soon deprecated
        return x





