import  random
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
        else:
            path_dict['train'] = config.get('CPU', 'SMALL_TRAIN_PATH')
            path_dict['val'] = config.get('CPU', 'SMALL_VAL_PATH')
            # this will be empty at the moment
            path_dict['test'] = config.get('CPU', 'TEST_PATH')
            path_dict['pickle_train'] = config.get('CPU', 'PICKLE_S_TRAIN_PATH_V2')
            path_dict['pickle_val'] = config.get('CPU', 'PICKLE_S_VAL_PATH_V2')
            # this will be empty at the moment
            path_dict['pickle_test'] = config.get('CPU', 'PICKLE_TEST_PATH')
            path_dict['labels'] = config.get('CPU', 'LABELS_PATH')

        return path_dict


