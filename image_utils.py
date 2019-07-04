import numpy as np
import cv2
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from sklearn.utils import shuffle
from tqdm import tqdm_notebook
import configparser
import random
import torch
import torchvision.transforms as transforms
from general_utils import GPUConfig

config_class = GPUConfig()
path_dict = config_class.get_paths_dict()

"""Methods to visualize all the images"""

# shuffled_labels = shuffle(labels_df)
#
# fig, ax = plt.subplots(2, 5, figsize=(20, 12))
# fig.suptitle('Histopathologic scans of lymph node sections',
#              fontsize=20)
# # Negative samples
# for i, idx in enumerate(shuffled_labels[shuffled_labels['label'] == 0]['id'][0:5]):
#     img_path = os.path.join(TRAIN_PATH, idx)
#     read_image(img_path + '.tif')
#     ax[0, i].imshow(read_image(img_path + '.tif'))
#     box = patches.Rectangle((32, 32), 32, 32, linewidth=4, edgecolor='b',
#                             facecolor='none', linestyle=':', capstyle='round')
#     ax[0, i].add_patch(box)
# ax[0, 0].set_ylabel('Negative samples', size='large')
# # positive samples
# for i, idx in enumerate(shuffled_labels[shuffled_labels['label'] == 1]['id'][0:5]):
#     img_path = os.path.join(TRAIN_PATH, idx)
#     read_image(img_path + '.tif')
#     ax[1, i].imshow(read_image(img_path + '.tif'))
#     box = patches.Rectangle((32, 32), 32, 32, linewidth=4,
#                             edgecolor='b', linestyle=':',
#                             facecolor='none', capstyle='round')
#     ax[1, i].add_patch(box)
# ax[1, 0].set_ylabel('Positive samples', fontsize='large')
# plt.show()


class ReviewImages:
    """ ======================================== """
    """ A class with utilities to review images  """
    """ Reason is to try and understand the data """
    """ Before jumping in to all the Deep Stuff  """
    """ ======================================== """

    def __init__(self, images_path, labels_path):
        self.images_path = images_path
        self.labels_path = labels_path

        self.img_df = None

        # self.build_img_data_frames()
        # self.apply_all_methods()

    def build_img_data_frames(self):
        """Build initial DataFrame of img_name and their label"""
        self.img_df = pd.read_csv(self.labels_path)

    def read_image(self, img_path):
        """ read img to RGB format, given a path of an image"""
        bgr_img = cv2.imread(img_path)
        # flip to rgb
        b, g, r = cv2.split(bgr_img)
        rgb_img = cv2.merge([r, g, b])
        return rgb_img

    def compare_images(self,  neg_img_list, pos_img_list):
        """Present 5 different images from each class
           given a list of img names from each class """
        fig, ax = plt.subplots(2, 5, figsize=(20, 12))
        fig.suptitle('Histopathologic scans of lymph node sections',
                     fontsize=20)

        # Negative samples
        for i, img in enumerate(neg_img_list):
            img_path = os.path.join(self.images_path, img + '.tif')
            ax[0, i].imshow(self.read_image(img_path))
            box = patches.Rectangle((32, 32), 32, 32, linewidth=4, edgecolor='b',
                                    facecolor='none', linestyle=':', capstyle='round')
            ax[0, i].add_patch(box)
        ax[0, 0].set_ylabel('Negative Samples', size='large')

        # positive samples
        for i, img in enumerate(pos_img_list):
            img_path = os.path.join(path_dict['train'], img + '.tif')
            ax[1, i].imshow(self.read_image(img_path))
            box = patches.Rectangle((32, 32), 32, 32, linewidth=4,
                                    edgecolor='b', linestyle=':',
                                    facecolor='none', capstyle='round')
            ax[1, i].add_patch(box)
        ax[1, 0].set_ylabel('Positive Samples', fontsize='large')
        plt.show()

    def get_image_array(self, img_name):
        img_path = os.path.join(self.images_path, img_name + '.tif')
        img = self.read_image(img_path)
        img_arr = np.array(img)
        return img_arr

    def get_image_energy(self, img_arr):
        """Calc the energy of image from cv2 image"""
        img_energy = np.sum(np.power(img_arr, 2))
        return img_energy

    def get_R_channel_energy(self, img_arr):
        """calc the energy in channel"""
        img_r_energy = np.sum(np.power(img_arr[:, :, 0], 2))
        return img_r_energy

    def get_G_channel_energy(self, img_arr):
        """calc the energy in channel"""
        img_g_energy = np.sum(np.power(img_arr[:, :, 1], 2))
        return img_g_energy

    def get_B_channel_energy(self, img_arr):
        """calc the energy in channel"""
        img_b_energy = np.sum(np.power(img_arr[:, :, 2], 2))
        return img_b_energy

    def get_image_mean(self,img_arr):
        """calc the mean of the image"""
        img_mean  = np.mean(img_arr)
        return img_mean

    def get_image_std(self,img_arr):
        """calc the std of the image"""
        img_std = np.std(img_arr)
        return img_std

    def calc_all_statistics(self, img_name):
        img_arr = self.get_image_array(img_name)
        img_energy = self.get_image_energy(img_arr)
        img_r_energy = self.get_R_channel_energy(img_arr)
        img_g_energy = self.get_G_channel_energy(img_arr)
        img_b_energy = self.get_B_channel_energy(img_arr)
        img_mean = self.get_image_mean(img_arr)
        img_std = self.get_image_std(img_arr)

        return pd.Series((img_energy, img_r_energy,
                          img_g_energy, img_b_energy,
                          img_mean, img_std))

    def apply_all_methods(self):
        self.img_df[['energy', 'r_energy', 'g_energy',
                     'b_energy', 'mean', 'std']] = \
            self.img_df.apply(lambda row: self.calc_all_statistics(row['id']),
                              axis=1)

    @staticmethod
    def visualize_before_net(tensor_img_path, img_transform):
        """ Visualize the data that is fed to the network"""
        # 1. Select a random image
        tensor_images_list = [im for im in os.listdir(tensor_img_path)
                              if os.path.isfile(os.path.join(tensor_img_path, im))]

        rand_int = random.randint(0, len(tensor_images_list))
        tnsr_img_name = tensor_images_list[rand_int]
        tnsr_img_path = os.path.join(tensor_img_path, tnsr_img_name)

        # 2. Apply transformations
        tnsr_img = torch.load(tnsr_img_path)
        trans_img = img_transform(tnsr_img)

        # 3. Show image
        pil_transform = transforms.Compose([transforms.ToPILImage()])
        pil_img = pil_transform(trans_img)
        pil_img.show()


#test_instance = ReviewImages(TRAIN_PATH, LABELS_PATH)

#
# plt.hist(test_instance.img_df.energy[test_instance.img_df.label == 0], bins=150)
# plt.hist(test_instance.img_df.energy[test_instance.img_df.label == 1], bins=150)
# plt.show()