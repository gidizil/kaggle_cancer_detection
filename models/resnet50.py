import torch
from torch import nn
import torchvision


""" ====================================== """
""" Using the pretrained model 'resnet 50' """
""" This serves for 2 purposes:            """
""" 1. Making sure the data pipeline is OK """
""" 2. Setting a baseline on results       """
""" ====================================== """


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.resnet50 = torchvision.models.resnet50()
        self.resnet50.fc = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=2048,
                out_features=1
            ),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.resnet50.forward(x)



