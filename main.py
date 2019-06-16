import torch
import torchvision
from data_utils_v2 import CancerDataset
from data_utils_v2 import PickleImageData
import configparser
from models.basic_two_layers_model import Net
from models.train_classifier import Classifier
import os
from general_utils import SetSeeds, GPUConfig, HyperParamsConfig
from transformations import Transformations

# 1. Get pickled data path
config = configparser.ConfigParser()
config.read_file(open(r'config.txt'))
class_config = GPUConfig()
path_dict = class_config.get_paths_dict()
h_params_config = HyperParamsConfig()
h_params_dict = h_params_config.params_dict

# 2A. get pickled data objects (unpickled) for training
images_list_path = os.path.join(path_dict['pickle_train'], 'images_names_list.pickle')
tr_images_list = PickleImageData.unpickle_file(images_list_path)
labels_dict_path = os.path.join(path_dict['pickle_train'], 'labels_dict.pickle')
tr_labels_dict = PickleImageData.unpickle_file(labels_dict_path)

# 2B. get pickled data objects (unpickled) for validation
images_list_path = os.path.join(path_dict['pickle_val'], 'images_names_list.pickle')
val_images_list = PickleImageData.unpickle_file(images_list_path)
labels_dict_path = os.path.join(path_dict['pickle_val'], 'labels_dict.pickle')
val_labels_dict = PickleImageData.unpickle_file(labels_dict_path)

# 3. setting seeds:
SetSeeds.seed_torch()

# 4A. Set transformers, then build Dataset and DataLoader for training
transformers = Transformations(crop_size=h_params_dict['center_crop'])
center_crop = transformers.center_crop

train_set = CancerDataset(path_dict['pickle_train'],
                          tr_labels_dict, tr_images_list,
                          img_transform=center_crop)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=h_params_dict['batch_size'],
                                           shuffle=True, num_workers=h_params_dict['num_workers'],
                                           worker_init_fn=SetSeeds._init_fn,
                                           )

# 4B. Build Dataset and DataLoader for validation
val_set = CancerDataset(path_dict['pickle_val'],
                        val_labels_dict, val_images_list,
                        img_transform=center_crop)

val_loader = torch.utils.data.DataLoader(val_set, batch_size=h_params_dict['batch_size'],
                                         shuffle=True, num_workers=h_params_dict['num_workers'],
                                         worker_init_fn=SetSeeds._init_fn)

# 5. Train Network
net = Net(h_params_dict['center_crop'])
net_classifier = Classifier(net)

net_classifier.fit_and_eval(train_loader, val_loader)

#6. Test on unseen data
# TODO: finish the rest to be able to make predictions on new data.

