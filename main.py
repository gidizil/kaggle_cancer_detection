import torch
import torchvision
import data_utils
import configparser
from models.basic_two_layers_model import Net
from models.train_classifier import Classifier

pickle_files = '/Users/gzilbar/msc/side_projects/data/kaggle_1_data/data/pickle_files'
train_set = data_utils.CancerDataset(data_utils)
train_loader = torch.utils.data.DataLoader()