import  random
import os
import numpy as np
import torch

""" =================================== """
""" Collection of miscellaneous methods """
""" to help with general problems       """
""" encountered along the way           """
""" =================================== """


class GeneralUtils:
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


