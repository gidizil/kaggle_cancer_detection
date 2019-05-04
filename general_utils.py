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

    @staticmethod
    def set_num_workers(has_gpu):
        config = configparser.ConfigParser()
        config.read_file(open(r'config.txt'))
        if has_gpu:
            return int(config.get('GPU', 'NUM_WORKERS'))
        else:
            return int(config.get('CPU', 'NUM_WORKERS'))
        





