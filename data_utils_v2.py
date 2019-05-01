from PIL import Image
import configparser
import os
import numpy as np
import csv
import pickle
import torch
from torch.utils import data
from torchvision import transforms
import pickle
import sys

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

    def __init__(self, images_path, pickle_path, labels_path,
                 img_transform=None, target_transform=None):
        self.images_path = images_path
        self.pickle_path = pickle_path
        self.labels_path = labels_path
        self.transform = img_transform
        self.target_transform = target_transform

        self.images_list = None
        self.labels_dict = None

        self.get_images_name()
        self.set_labels_dict()

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
            labels_dict = dict(reader)
        self.labels_dict = {k: int(labels_dict[k]) for k in labels_dict.keys() if k != 'id'}

    def image_to_pil(self, file_name):
        """open the image as a PIL image (BGR)"""
        img_path = os.path.join(self.images_path, file_name)
        pil_img = Image.open(img_path)
        return pil_img

    def pil_image_to_tensor(self, pil_img):
        if self.transform is not None:
            img_tensor = self.transform(pil_img)
        else:
            transform = transforms.Compose(
                [transforms.ToTensor()])
            img_tensor = transform(pil_img)

        return img_tensor

    def pickle_tensor_image(self, tensor_img, file_name):
        """Convert a file to pickle file"""
        pickle_file_name = file_name.split('.')[0] + '.pt'
        img_pickle_path = os.path.join(self.pickle_path, pickle_file_name)
        # output_file = open(img_pickle_path, 'wb')
        torch.save(tensor_img, img_pickle_path)
        # output_file.close()

    def pickle_all_images(self):
        """ Pickle all the files in images list"""
        for img_file_name in self.images_list:
            pil_img = self.image_to_pil(img_file_name)
            tensor_img = self.pil_image_to_tensor(pil_img)
            self.pickle_tensor_image(tensor_img, img_file_name)

    def pickle_images_list(self):
        """Pickle the list of images name"""
        images_names_list_path = os.path.join(self.pickle_path, 'images_names_list.pickle')
        output_file = open(images_names_list_path, 'wb')
        images_list = [img_name.split('.')[0] for img_name in self.images_list]
        pickle.dump(images_list, output_file)
        output_file.close()

    def pickle_labels_dict(self):
        """ Pickle the dictionary holding image and it's corresponding label"""
        lables_pickle_path = os.path.join(self.pickle_path, 'labels_dict.pickle')
        output_file = open(lables_pickle_path, 'wb')
        pickle.dump(self.labels_dict, output_file)
        output_file.close()

    def pickle_everything(self):
        self.pickle_all_images()
        self.pickle_images_list()
        self.pickle_labels_dict()

    @staticmethod
    def unpickle_file(file_path):
        file_open = open(file_path, 'rb')

        # Handling unpickling in differnt versions
        # result of entry is the dict from PickleImageData() class
        if sys.version_info[0] == 2:
            data_object = pickle.load(file_open)
        else:
            data_object = pickle.load(file_open, encoding='latin1')

        return data_object


test = PickleImageData(SMALL_TRAIN_PATH, PICKLE_S_TRAIN_PATH_V2, LABELS_PATH)
test.pickle_everything()


class CancerDataset(data.Dataset):
    """ ====================================== """
    """ Implementing a pytorch Dataset object. """
    """ X should be a transformed and pickled  """
    """ tensor. y is an integer                """
    """ ====================================== """

    def __init__(self, data_path,labels_dict, images_list,
                 img_transform=None, target_transform=None):
        self.data_path = data_path
        self.labels_dict = labels_dict
        self.images_list = images_list
        self.img_transform = img_transform
        self.target_transform = target_transform

    def __len__(self):
        """get the number of images samples in the dataset"""
        return len(self.images_list)

    def __getitem__(self, index):
        """Generate one sample data tensor and it's label"""
        img_name = self.images_list[index]
        X = torch.load(os.path.join(self.data_path, img_name + '.pt'))
        y = torch.tensor(self.labels_dict[img_name], dtype=torch.long)

        return X, y


