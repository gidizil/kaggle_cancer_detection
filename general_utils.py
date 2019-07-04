import random
import os
import numpy as np
import torch
import configparser

""" =================================== """
""" Collection of miscellaneous methods """
""" to help with general problems       """
""" encountered along the way           """
""" =================================== """


class SetSeeds:
    """Collection of useful methods"""

    @staticmethod
    def seed_torch(seed=1029):
        """Setting a fixed seed for torch and it's dependencies"""
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    @staticmethod
    def _init_fn(worker_id):
        """ Setting a fixed seed for each worker in the DataLoader"""
        np.random.seed(12 + worker_id)



class GPUConfig:
    """Different configurations for the case of GPU/CPU"""

    def __init__(self):
        self.has_gpu = torch.cuda.is_available()
        # TODO: init config file here

    def set_num_workers(self):
        config = configparser.ConfigParser()
        config.read_file(open(r'config.txt'))
        if self.has_gpu:
            return int(config.get('GPU', 'NUM_WORKERS'))
        else:
            return int(config.get('CPU', 'NUM_WORKERS'))

    def get_paths_dict(self):
        """Get a dict of the paths for the case of cpu/gpu"""
        config = configparser.ConfigParser()
        config.read_file(open(r'config.txt'))
        path_dict = {}
        if self.has_gpu:
            path_dict['train'] = config.get('GPU', 'TRAIN_PATH')
            path_dict['val'] = config.get('GPU', 'VAL_PATH')
            path_dict['test'] = config.get('GPU', 'TEST_PATH')
            path_dict['pickle_train'] = config.get('GPU', 'PICKLE_TRAIN_PATH')
            path_dict['pickle_val'] = config.get('GPU', 'PICKLE_VAL_PATH')
            path_dict['pickle_test'] = config.get('GPU', 'PICKLE_TEST_PATH')
            path_dict['labels'] = config.get('GPU', 'LABELS_PATH')
            path_dict['plots'] = config.get('GPU', 'PLOTS_PATH')
            path_dict['means'] = config.get('GPU', 'CHANNELS_MEAN')
            path_dict['epochs_num'] = int(config.get('GPU', 'EPOCHS'))
        else:
            path_dict['train'] = config.get('CPU', 'SMALL_TRAIN_PATH')
            path_dict['val'] = config.get('CPU', 'SMALL_VAL_PATH')
            path_dict['test'] = config.get('CPU', 'TEST_PATH')
            path_dict['pickle_train'] = config.get('CPU', 'PICKLE_S_TRAIN_PATH_V2')
            path_dict['pickle_val'] = config.get('CPU', 'PICKLE_S_VAL_PATH_V2')
            # this one doesn't really exist. only for compatability
            path_dict['pickle_test'] = config.get('CPU', 'PICKLE_S_TEST_PATH_V2')
            path_dict['labels'] = config.get('CPU', 'LABELS_PATH')
            path_dict['plots'] = config.get('CPU', 'PLOTS_PATH')
            path_dict['means'] = config.get('CPU', 'CHANNELS_MEAN')
            path_dict['epochs_num'] = int(config.get('CPU', 'EPOCHS'))

        return path_dict


class HyperParamsConfig:
    """ Set hyper params config. Conisder CPU/GPU"""
    def __init__(self):
        self.has_gpu = torch.cuda.is_available()
        self.config = configparser.ConfigParser()

        self.set_config_path()
        self.config.read_file(open(self.config_path))
        self.params_dict = {}

        self.get_params_dict()

    def set_config_path(self):
        """ Set config.txt path for the case of GPU/CPU"""
        if self.has_gpu:
            self.config_path = '/home/gzilbar/cancer_detection/config.txt'
        else:
            self.config_path = '/Users/gzilbar/msc/side_projects/kaggle_1/config.txt'

    def get_params_dict(self):
        """ Set all hyper params based on cpu/gpu"""

        if self.has_gpu:
            self.params_dict['num_workers'] = int(self.config.get('GPU_H_PARAMS', 'NUM_WORKERS'))
            self.params_dict['num_epochs'] = int(self.config.get('GPU_H_PARAMS', 'EPOCHS'))
            self.params_dict['batch_size'] = int(self.config.get('GPU_H_PARAMS', 'BATCH_SIZE'))
            self.params_dict['center_crop'] = int(self.config.get('GPU_H_PARAMS', 'CENTER_CROP'))
            self.params_dict['lr'] = float(self.config.get('GPU_H_PARAMS', 'LR'))
            self.params_dict['orig_img_size'] = int(self.config.get('GPU_H_PARAMS', 'ORIG_IMG_SIZE'))
            self.params_dict['resize'] = int(self.config.get('GPU_H_PARAMS', 'IMG_RESIZE'))
            self.params_dict['loss_func'] = self.config.get('GPU_H_PARAMS', 'LOSS')

        else:
            self.params_dict['num_workers'] = int(self.config.get('CPU_H_PARAMS', 'NUM_WORKERS'))
            self.params_dict['num_epochs'] = int(self.config.get('CPU_H_PARAMS', 'EPOCHS'))
            self.params_dict['batch_size'] = int(self.config.get('CPU_H_PARAMS', 'BATCH_SIZE'))
            self.params_dict['center_crop'] = int(self.config.get('CPU_H_PARAMS', 'CENTER_CROP'))
            self.params_dict['lr'] = float(self.config.get('CPU_H_PARAMS', 'LR'))
            self.params_dict['orig_img_size'] = int(self.config.get('CPU_H_PARAMS', 'ORIG_IMG_SIZE'))
            self.params_dict['resize'] = int(self.config.get('CPU_H_PARAMS', 'IMG_RESIZE'))
            self.params_dict['loss_func'] = self.config.get('CPU_H_PARAMS', 'LOSS')
