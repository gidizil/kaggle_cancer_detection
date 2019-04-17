import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from models.basic_two_layers_model import Net
import configparser

"""==========================================="""
""" Create a class for Training a classifier  """
""" based on architectures defined in /models """
""" use config.txt to define hyperparameters  """
"""==========================================="""

