# ==================================================
# DEPRECATED
# ==================================================

import os
import pathlib
import random
import cv2
import numpy as np

data_path = "dataset/split/"

def load_x_y(data_path):
    data_root = pathlib.Path(data_path)

    all_image_paths = list(data_root.glob('*/*'))
    all_image_paths = [str(path) for path in all_image_paths]
    random.shuffle(all_image_paths)

    label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
    label_to_index = dict((name, index) for index,name in enumerate(label_names))
    all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                    for path in all_image_paths]

    labels_size = len(all_image_labels)
    num_classes = len(set(all_image_labels))
    all_image_labels_hot = np.zeros((labels_size, num_classes))
    all_image_labels_hot[np.arange(labels_size), all_image_labels] = 1

    all_images = [cv2.imread(file) for file in all_image_paths]
    # all_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in all_images]
    # all_images = [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) for img in all_images]
    all_images = np.concatenate([arr[np.newaxis] for arr in all_images]).astype('float32')

    print(label_to_index)
    return all_images, all_image_labels_hot, num_classes

def training():
    return load_x_y(data_path+"train")
def validation():
    return load_x_y(data_path+"val")
def test():
    return load_x_y(data_path+"test")