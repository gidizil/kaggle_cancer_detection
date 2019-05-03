import torch
import torchvision
from data_utils_v2 import CancerDataset
from data_utils_v2 import PickleImageData
import configparser
from models.basic_two_layers_model import Net
from models.train_classifier import Classifier
import os
from general_utils import GeneralUtils

# 1. Get pickled data path
config = configparser.ConfigParser()
config.read_file(open(r'config.txt'))
PICKLE_S_TRAIN_PATH_V2 = config.get('PATHS', 'PICKLE_S_TRAIN_PATH_V2')

# 2. get pickled data objects (unpickled)
images_list_path = os.path.join(PICKLE_S_TRAIN_PATH_V2, 'images_names_list.pickle')
images_list = PickleImageData.unpickle_file(images_list_path)
labels_dict_path = os.path.join(PICKLE_S_TRAIN_PATH_V2, 'labels_dict.pickle')
labels_dict = PickleImageData.unpickle_file(labels_dict_path)

# 3. setting seeds:
GeneralUtils.seed_torch()


train_set = CancerDataset(PICKLE_S_TRAIN_PATH_V2, labels_dict, images_list)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64,
                                           shuffle=True, num_workers=2,
                                           worker_init_fn=GeneralUtils._init_fn)

net = Net()
net_classifier = Classifier(net)

net_classifier.fit(train_loader)