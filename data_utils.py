import torch
from torch.utils import data
import pandas as pd
import os
import configparser
import numpy as np
import random
from PIL import Image
import csv
import pickle
"""======================================================"""
""" Two classes: one for pickling images and their labels"""
""" Second class is building the Dataset object using the"""
""" pickled data                                         """
"""======================================================"""

# 1. get the names of the files (images) and their labels we will work with
config = configparser.ConfigParser()
config.read_file(open(r'config.txt'))
TRAIN_PATH = config.get('PATHS', 'TRAIN_PATH')
SMALL_TRAIN_PATH = config.get('PATHS', 'SMALL_TRAIN_PATH')
SMALL_VAL_PATH = config.get('PATHS', 'SMALL_VAL_PATH')
TEST_PATH = config.get('PATHS', 'TEST_PATH')
LABELS_PATH = config.get('PATHS', 'LABELS_PATH')
PICKLE_FILES_PATH = config.get('PATHS', 'PICKLE_FILES_PATH')

class PickleImageData():
    """
    Produce dictionaries of relevant data
    dictionary contains:
    'batch_label': string. number of pickled batch
    'labels': list. list of labels. size of pickled batch
    'data': np.array. all images. possibly flattned
    'filenames': list. name of files

    Dictionary is then pickled
    """

    # TODO: (optinal) add a config interface
    # TODO: (optinal) add a way to insert labels dictionary
    def __init__(self, image_path, labels_path, is_train, pickle_size=5000):
        self.image_path = image_path
        self.labels_path = labels_path
        self.is_train = is_train
        self.pickle_size = pickle_size

        self.all_files_list = None
        self.shuffled_indices = None
        self.img_dims = None
        self.num_of_Batches = None

        self.get_all_file_names()
        self.shuffle_file_indices()
        self.labels_csv_to_dict()
        self.get_img_dims()
        self.get_num_of_batches()

    def get_img_dims(self):
        """Get image dimensions"""
        frst_img_path = os.path.join(self.image_path, self.all_files_list[0])
        frst_img = Image.open(frst_img_path)
        frst_img_np = np.array(frst_img)
        self.img_dims = frst_img_np.shape

    def get_all_file_names(self):
        """Get the name of images files"""
        all_files_list = [f for f in os.listdir(self.image_path)
                          if os.path.isfile(os.path.join(self.image_path, f))]
        self.all_files_list = np.array(all_files_list) # wrap with np for ease of use in 'build_pickle_dicts'

    def shuffle_file_indices(self):
        """Shuffle indices for future shuffle of files"""
        img_num = len(self.all_files_list)
        self.shuffled_indices = random.sample(range(img_num), img_num)

    def extract_file_names_from_indices(self, indices):
        """Get correspnding files_names to the their relevant indices"""
        files_list = [self.all_files_list[idx] for idx in indices]
        return files_list

    #TODO: Think about storing data as tensors
    def extract_data_from_files(self, files_list, flatten=False):
        """Extract images into numpy array"""
        if flatten:
            batch_data = np.zeros(len(files_list), np.prod(self.img_dims))
        else:
            batch_data = np.zeros((len(files_list), *self.img_dims))
        for idx, file_name in enumerate(files_list):
            path = os.path.join(self.image_path, file_name)
            np_img = np.array(Image.open(path))
            np_img = np_img[:, :, ::-1]  # Convert image from BGR to RGB
            if flatten:
                batch_data[idx, :] = np.reshape(np_img, newshape=[1, -1])[0]  # flatten image
            else:
                batch_data[idx, :, :, :] = np_img

        return batch_data

    def labels_csv_to_dict(self):
        """create dictionary of id (key) and corresponding label (value)"""
        labels_dict = {}
        with open(self.labels_path) as labels_file:
            reader = csv.reader(labels_file, delimiter=',')
            self.labels_dict = dict(reader)

    def extract_labels_from_files(self, files_list):
        """Create s list of labels based on corresponding images (files)"""
        labels_list = []
        for file in files_list:
            id = file.split('.')[0]
            labels_list.append(self.labels_dict[id])

        return labels_list

    def get_num_of_batches(self):
        """Understand how many pickled dict we are going to have"""
        self.num_of_batches = (len(self.all_files_list) // self.pickle_size) + 1

    #TODO: Adjust to train or test
    def set_batch_label(self, idx):
        """Name the pickled batch"""
        batch_label = 'training batch {0} of {1}'.format(idx, self.num_of_batches)
        return batch_label

    def build_single_dict(self, idx, files_list):
        """Build a single dictionary that is going to be pickled later"""
        batch_label = self.set_batch_label(idx)
        labels = self.extract_labels_from_files(files_list)
        data = self.extract_data_from_files(files_list, flatten=False)
        file_names = files_list # Redundant, but makes it more readable

        single_dict = {'batch_label': batch_label,
                       'labels': labels,
                       'data': data,
                       'file_names': file_names}
        return single_dict
    #TODO: Add a no lables version for test time
    def build_pickled_dicts(self, pickle_name):
        pickle_size = self.pickle_size
        num_batchs = self.num_of_batches
        for i in range(num_batchs):
            start_pos = i * pickle_size
            end_pos = min((i + 1) * pickle_size, len(self.all_files_list))
            tmp_shuf_indices = self.shuffled_indices[start_pos: end_pos]
            files_list = self.all_files_list[tmp_shuf_indices]
            single_dict = self.build_single_dict(idx=i+1, files_list=files_list)

            # pickle dictionary
            pickle_file_name = pickle_name + '_' + str(i+1)
            pickle_path = os.path.join(PICKLE_FILES_PATH, pickle_file_name)
            out_file = open(pickle_path, 'wb')
            pickle.dump(single_dict, out_file)
            out_file.close()



test_instance = PickleImageData(image_path=SMALL_TRAIN_PATH, labels_path=LABELS_PATH,is_train=False, pickle_size=300)
test_instance.build_pickled_dicts('small_train_data')
#print(test_instance.shuffle_file_indices())
