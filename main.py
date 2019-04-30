import torch
import torchvision
from data_utils import CancerDataset
import configparser
from models.basic_two_layers_model import Net
from models.train_classifier import Classifier

# 1. Get pickled data path
config = configparser.ConfigParser()
config.read_file(open(r'config.txt'))
PICKLE_TRAIN_PATH = config.get('PATHS', 'PICKLE_TRAIN_PATH')

train_set = CancerDataset(PICKLE_TRAIN_PATH)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64,
                                           shuffle=True, num_workers=2)

net = Net()
net_classifier = Classifier(net)

net_classifier.fit(train_loader)