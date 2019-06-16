import torch
import torchvision.transforms as transforms
import configparser
from general_utils import GPUConfig
import numpy as np
import sys
import pickle

""" ================================= """
""" Class to hold everything that is  """
""" related to pre-processing of the  """
""" data and augmentations as well.   """
""" Methods for training/testing with """
""" augmentation will also be here    """
""" ================================= """


class Transformations:
    """ 1. Holding all the different Trasnformations.   """
    """ 2. setting default values for certain params.   """
    """ Use this as a helper of for the Dataset         """
    """ class. There is base transform - normalize      """
    """ All other, normalize and do other transforms    """

    # TODO: work with kwargs
    def __init__(self, **kwargs):
        # 1. Basic initialization of given params
        self.__dict__.update(kwargs)
        self.params = self.__dict__.copy()
        config_class = GPUConfig()
        self.path_dict = config_class.get_paths_dict()

        self.base_transform = None
        self.center_crop = None

        # 2. set required params
        self.set_base_transform()
        self.set_center_crop_transform()

    # 3. Initialize all transformers
    def get_channels_mean(self):
        pickle_means = open(self.path_dict['means'], 'rb')
        if sys.version_info[0] == 2:
            np_means = pickle.load(pickle_means)
        else:
            np_means = pickle.load(pickle_means, encoding='latin1')

        return np_means

    # TODO: allow to work with stds
    def set_base_transform(self):
        if self.params.get('means', None) is None:
            self.params['means'] = self.get_channels_mean()

        self.base_transform = transforms.Compose([
            transforms.Normalize(self.params['means'], (1, 1, 1))
        ])

    def set_center_crop_transform(self):
        if self.params.get('crop_size', None) is not None:
            self.center_crop = transforms.Compose([
                transforms.Normalize(self.params['means'], (1, 1, 1)),  # TODO: add params of mean
                transforms.ToPILImage(),
                transforms.CenterCrop(size=self.params['crop_size']),
                transforms.ToTensor()
            ])




#Working with **kwargs
#
# class TryKwargs:
#     def __init__(self):
#         pass
#
#     def intro(**data):
#         print("\nData type of argument:",type(data))
#
#         if data.get('means', None) is None:
#             data['means'] = 'abcs!!!!'
#             print(data['means'])
#
#         for key, value in data.items():
#             print("{} is {}".format(key,value))
#
#     intro(Firstname="Sita", Lastname="Sharma", Age=22, Phone=1234567890)
#     intro(Firstname="John", Lastname="Wood", Email="johnwood@nomail.com", Country="Wakanda", Age=25, Phone=9876543210)