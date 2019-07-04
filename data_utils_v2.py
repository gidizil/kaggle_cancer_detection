from PIL import Image
import configparser
import os
import numpy as np
import csv
import torch
from torch.utils import data
from torchvision import transforms
import pickle
import sys
from sklearn.model_selection import train_test_split
import pandas as pd
import shutil
from general_utils import GPUConfig

""" =========================================== """
""" Another versions of the PickleImage class   """
""" and the CancerDataset class. Based on the   """
""" Stanford Tutorial.                          """
""" =========================================== """

# Some directories configuration
# TODO: add GPUConfig inside each class
config = configparser.ConfigParser()
config.read_file(open(r'config.txt'))
config_class = GPUConfig()
path_dict = config_class.get_paths_dict()


class PickleImageData:
    """ ========================================================== """
    """ Produces three things:                                     """
    """ 1. Directory with images pickled as tensors/pil images     """
    """ 2. list with name of files (list) in each directory        """
    """ 3. Dictionary. label (value) of each image file_name (key) """
    """ ========================================================== """

    def __init__(self, images_path, pickle_path, labels_path,
                 img_type='tensor', img_transform=None, target_transform=None):
        self.images_path = images_path
        self.pickle_path = pickle_path
        self.labels_path = labels_path
        self.img_type = img_type
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

    def pickle_pil_image(self, pil_img, file_name):
        pickle_file_name = file_name.split('.')[0] + '.pickle'
        img_pickle_path = os.path.join(self.pickle_path, pickle_file_name)
        with open(img_pickle_path, 'wb') as fp:
            pickle.dump(pil_img, fp)

    def pickle_tensor_image(self, tensor_img, file_name):
        """Convert a file to pickle file"""
        pickle_file_name = file_name.split('.')[0] + '.pt'
        img_pickle_path = os.path.join(self.pickle_path, pickle_file_name)
        torch.save(tensor_img, img_pickle_path)

    def pickle_all_images(self, img_type):
        """ Pickle all the files in images list. can pickle tensors/pil images"""
        for img_file_name in self.images_list:
            pil_img = self.image_to_pil(img_file_name)
            if img_type == 'tensor':
                tensor_img = self.pil_image_to_tensor(pil_img)
                self.pickle_tensor_image(tensor_img, img_file_name)
            elif img_type == 'pil':
                self.pickle_pil_image(pil_img, img_file_name)
            else:
                print('Invalid image pickling type')

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
        self.pickle_all_images(img_type=self.img_type)
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

""" Pickle all files"""
# test = PickleImageData(path_dict['train'], path_dict['pickle_train'], path_dict['labels'])
# test.pickle_everything()


class CancerDataset(data.Dataset):
    """ ====================================== """
    """ Implementing a pytorch Dataset object. """
    """ X should be a transformed and pickled  """
    """ tensor. y is an integer                """
    """ ====================================== """

    def __init__(self, data_path, labels_dict, images_list,
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
        """Generate one sample data tensor and it's label"""
        img_name = self.images_list[index]
        X = torch.load(os.path.join(self.data_path, img_name + '.pt'))
        if self.img_transform is not None:
            X = self.img_transform(X)

        y = torch.tensor(self.labels_dict[img_name], dtype=torch.long)

        return X, y


class GeneralDataUtils:
    """ ================================= """
    """ Collection of useful methods      """
    """ that are data related but do not  """
    """ Under the category of the classes """
    """ above. possible even as one-of's  """
    """ ================================= """

    def __init__(self):
        pass

    def _get_files_in_dir(self, dir_path, suffix=None):
        """ get list of files in dir. specific suffix - optional"""
        if suffix is not None:
            images_list = [f.split('.')[0] for f in os.listdir(dir_path) if f.endswith(suffix)]
        else:
            images_list = [f.split('.')[0] for f in os.listdir(dir_path)]

        return images_list

    def _get_relevant_images_and_labels(self, images_list, labels_path):
        """extract relevant images, labels for a given images list"""
        labels_df = pd.read_csv(labels_path)
        relevant_labels_df = labels_df[labels_df.id.isin(images_list)]

        rel_images = relevant_labels_df.id.values
        rel_labels = relevant_labels_df.label.values

        return rel_images, rel_labels

    def train_val_split(self, train_path, val_path, labels_path, val_rate=0.2, special_suffix=None):
        """ Moves random files from train to val dir. specific suffix - optional """

        # 1. Create a list of all images in directory:
        images_list = self._get_files_in_dir(train_path, special_suffix)

        # 2. extract files to be in val directory
        images, labels = self._get_relevant_images_and_labels(images_list, labels_path)

        train_images, val_images, y_tr, y_val = \
            train_test_split(images, labels, test_size=val_rate, stratify=labels)

        # 3. moves files to val directory:
        for img in val_images:
            train_img_path = os.path.join(train_path, img + '.tif')
            val_img_path = os.path.join(val_path, img + '.tif')
            shutil.move(train_img_path, val_img_path)

    def get_bgr_channels_mean(self, images_path, save_path):
        """Return the mean valus in each channel across all pictures"""
        images_list = self._get_files_in_dir(images_path) # set path of original images - not tensors

        # 1. Understand image dims:
        img_path = os.path.join(images_path, images_list[0] + '.tif')
        bgr_img = Image.open(img_path)
        bgr_np = np.array(bgr_img)
        n_dims = len(bgr_np.shape)
        bgr_mean = np.zeros(shape=(bgr_np.shape[-1],))
        bgr_std = np.zeros(shape=(bgr_np.shape[-1],))

        for image in images_list:
            img_path = os.path.join(images_path, image + '.tif')
            bgr_img = Image.open(img_path)
            bgr_np = np.array(bgr_img)

            #print(bgr_np.shape)
            # add the mean in each channel - image format is HxWxC
            bgr_mean += np.mean(bgr_np, axis=tuple(range(0, n_dims-1)))
            bgr_std += np.std(bgr_np, axis=tuple(range(0, n_dims-1)))

        # 2. Divide each channel by the number of images
        bgr_mean /= len(images_list)

        # 3. Store image means as pickle to to the pickled train directory
        channel_mean_pickle_path = os.path.join(save_path, 'channels_mean.pickle')
        output_file = open(channel_mean_pickle_path, 'wb')
        pickle.dump(bgr_mean, output_file)
        output_file.close()

    # def get_bgr_channels_mean_std(self, ):
    #     """ Compute channels mean and std """
    #     for i, (images, labels) in enumerate(train_loader, 0):
    #         # set params and all others to train mode
    #         # TODO: is possible to set classifier outside training loop
    #         self.classifier.train()
    #         # pass data gpu (if exists)
    #         images = images.to(self.device)
    #         labels = labels.to(self.device)


"""Move the images to val"""
# move_files_to_val = GeneralDataUtils()
# move_files_to_val.train_val_split('/Users/gzilbar/msc/side_projects/data/kaggle_1_data/train',
#                                   path_dict['train'],
#                                   path_dict['labels'],
#                                   val_rate=0.001)

# Get images mean per channel

# gen = GeneralDataUtils()
# gen.get_bgr_channels_mean(path_dict['train'], path_dict['pickle_train'])
