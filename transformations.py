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
        self.resize = None
        self.crop_resize = None
        self.basic_augment = None
        self.rand_augment = None

        # 2. set required params
        self._set_base_transform()
        self._set_center_crop_transform()
        self._set_resize_transform()
        self._set_crop_resize_transform()
        self._set_basic_augmentations_transform()
        self._randomly_augment_transform()

        self._set_transform_dict()

    # 3. Initialize all transformers
    def _get_channels_mean(self):
        pickle_means = open(self.path_dict['means'], 'rb')
        if sys.version_info[0] == 2:
            np_means = pickle.load(pickle_means)
        else:
            np_means = pickle.load(pickle_means, encoding='latin1')

        return np_means

    # TODO: allow to work with stds
    def _set_base_transform(self):
        """ Normalize image on;t"""
        if self.params.get('means', None) is None:
            self.params['means'] = self._get_channels_mean()

        self.base_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _set_center_crop_transform(self):
        """ Crop central part of image and normalize"""
        if self.params.get('crop_size', None) is not None:
            if self.params.get('crop_size', None) is not None:
                self.center_crop = transforms.Compose([
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),  # TODO: add params of mean
                    transforms.ToPILImage(),
                    transforms.CenterCrop(size=self.params['crop_size']),
                    transforms.ToTensor()
                ])

    def _set_resize_transform(self):
        """ Resize and normalize image """
        if self.params.get('resize', None) is not None:
            self.resize = transforms.Compose([
                transforms.ToPILImage(),  # Convert np array to PILImage
                transforms.Resize(size=self.params['resize']),  # 224x224 is standard for most models
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def _set_crop_resize_transform(self):
        """ Center crop and resize """
        is_resize = self.params.get('resize', None) is not None
        is_center_crop = self.params.get('crop_size', None) is not None
        if is_resize and is_center_crop:
            self.crop_resize = transforms.Compose([
                transforms.ToPILImage(),
                transforms.CenterCrop(size=self.params['crop_size']),
                transforms.Resize(size=self.params['resize']),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def _set_basic_augmentations_transform(self):
        """ A series of random augmentations to the image. then normalize """
        is_resize = self.params.get('resize', None) is not None
        is_center_crop = self.params.get('crop_size', None) is not None

        # Consider all cases of available params
        if (not is_resize) and (not is_center_crop):
            self.basic_augment = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ColorJitter(hue=.05, saturation=.05),
                transforms.RandomRotation(degrees=(-45, 45)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
                
            ])

        elif (not is_resize) and is_center_crop:
            self.basic_augment = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ColorJitter(hue=.05, saturation=.05),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=(-45, 45)),
                transforms.CenterCrop(size=self.params['crop_size']),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

            ])

        elif is_resize and (not is_center_crop):
            self.basic_augment = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ColorJitter(hue=.05, saturation=.05),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=(-45, 45)),
                transforms.Resize(size=self.params['resize']),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

            ])

        else:
            self.basic_augment = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=(-45, 45)),
                transforms.CenterCrop(size=self.params['crop_size']),
                transforms.ColorJitter(hue=.05, saturation=.05),
                transforms.Resize(size=self.params['resize']),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

            ])

    def _randomly_augment_transform(self):
        is_resize = self.params.get('resize', None) is not None
        is_center_crop = self.params.get('crop_size', None) is not None
        assert (is_resize and is_center_crop), \
            'Class instance must have resize and center_crop args'

        self.rand_augment = transforms.Compose([
            transforms.RandomChoice([
                self.crop_resize,
                self.basic_augment,
                self.basic_augment
            ])
        ])

    def _set_transform_dict(self):
        """ Create a dictionary with all of the transforms """
        self.transform_dict = {
            'base_transform': self.base_transform,
            'center_crop': self.center_crop,
            'resize': self.resize,
            'crop_resize': self.crop_resize,
            'basic_augment': self.basic_augment,
            'rand_augment': self.rand_augment
        }

    def set_transform(self, transform_name):
        """ select a transform given it's name """
        assert (transform_name is not None), 'Transform name must be provided'

        return self.transform_dict[transform_name]


# Working with **kwargs
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
