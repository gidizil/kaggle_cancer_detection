import torch
from PIL import Image
import configparser
import os
import numpy as np
import csv
import pickle

"""==========================================="""
""" Another versions of the PickleImage class """
""" and the CancerDataset class. Based on the """
""" Stanford Tutorial.                        """
"""==========================================="""

config = configparser.ConfigParser()
config.read_file(open(r'config.txt'))
SMALL_TRAIN_PATH = config.get('PATHS', 'SMALL_TRAIN_PATH')
SMALL_VAL_PATH = config.get('PATHS', 'SMALL_VAL_PATH')
PICKLE_S_TRAIN_PATH_V2 = config.get('PATHS', 'PICKLE_S_TRAIN_PATH_V2')
PICKLE_S_VAL_PATH_V2 = config.get('PATHS', 'PICKLE_S_VAL_PATH_V2')
LABELS_PATH = config.get('PATHS', 'LABELS_PATH')


class PickleImageData:
    """ ========================================================== """
    """ Produces three things:                                     """
    """ 1. Directory with images pickled as tensors                """
    """ 2. list with name of files (list) in each directory        """
    """ 3. Dictionary. label (value) of each image file_name (key) """
    """ ========================================================== """

    def __init__(self, images_path, pickle_path, labels_path):
        self.images_path = images_path
        self.pickle_path = pickle_path
        self.labels_path = labels_path

        self.images_list = None
        self.labels_dict = None

        self.get_images_name()

    def get_images_name(self):
        """Create a list of all images names"""
        files_list = [f for f in os.listdir(self.images_path)
                      if os.path.isfile(os.path.join(self.images_path, f))]
        self.images_list = np.array(files_list)

    def set_labels_dict(self):
        """Create a dict of image names (key) and their label (value)"""
        self.labels_dict = {}
        with open(self.labels_path) as labels_file:
            reader = csv.reader(labels_file, delimiter=',')
            self.labels_dict = dict(reader)

    def image_to_pil(self, file_name):
        """open the image as a PIL image (BGR)"""
        img_path = os.path.join(self.images_path, file_name)
        pil_img = Image.open(img_path)
        return pil_img

    def pickle_pil_image(self, pil_img, file_name):
        """Convert a file to pickle file"""
        pickle_file_name = file_name.split('.')[0] + '.pickle'
        img_pickle_path = os.path.join(self.pickle_path, pickle_file_name)
        output_file = open(img_pickle_path, 'wb')
        pickle.dump(pil_img, output_file)
        output_file.close()

    def pickle_all_images(self):
        """ Pickle all the files in images list"""
        for img_file_name in self.images_list:
            pil_img = self.image_to_pil(img_file_name)
            self.pickle_pil_image(pil_img, img_file_name)

    




test = PickleImageData(SMALL_TRAIN_PATH, PICKLE_S_TRAIN_PATH_V2, LABELS_PATH)
test.pickle_all_images()

