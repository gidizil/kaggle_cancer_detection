import numpy as np
import cv2
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from sklearn.utils import shuffle
from tqdm import tqdm_notebook


DATA_PATH = '/Users/gzilbar/msc/side_projects/data/kaggle_1_data/data'
TRAIN_PATH = os.path.join(DATA_PATH, 'train')
labels_df = pd.read_csv(os.path.join(DATA_PATH, 'train_labels.csv'))

"""Methods to visualzie all the images"""
def read_image(img_path):
    bgr_img = cv2.imread(img_path)
    # flip to rgb
    b, g, r = cv2.split(bgr_img)
    rgb_img = cv2.merge([r, g, b])
    return rgb_img


shuffled_labels = shuffle(labels_df)

fig, ax = plt.subplots(2, 5, figsize=(20, 12))
fig.suptitle('Histopathologic scans of lymph node sections',
             fontsize=20)
# Negative samples
for i, idx in enumerate(shuffled_labels[shuffled_labels['label'] == 0]['id'][0:5]):
    img_path = os.path.join(TRAIN_PATH, idx)
    read_image(img_path + '.tif')
    ax[0, i].imshow(read_image(img_path + '.tif'))
    box = patches.Rectangle((32, 32), 32, 32, linewidth=4, edgecolor='b',
                            facecolor='none', linestyle=':', capstyle='round')
    ax[0, i].add_patch(box)
ax[0, 0].set_ylabel('Negative samples', size='large')
# positive samples
for i, idx in enumerate(shuffled_labels[shuffled_labels['label'] == 1]['id'][0:5]):
    img_path = os.path.join(TRAIN_PATH, idx)
    read_image(img_path + '.tif')
    ax[1, i].imshow(read_image(img_path + '.tif'))
    box = patches.Rectangle((32, 32), 32, 32, linewidth=4,
                            edgecolor='b', linestyle=':',
                            facecolor='none', capstyle='round')
    ax[1, i].add_patch(box)
ax[1, 0].set_ylabel('Positive samples', fontsize='large')
plt.show()


