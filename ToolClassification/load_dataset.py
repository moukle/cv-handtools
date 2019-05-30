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

    all_images = [cv2.imread(file) for file in all_image_paths]

    return all_images, all_image_labels

def training():
    return load_x_y(data_path+"train")
def validation():
    return load_x_y(data_path+"val")
def test():
    return load_x_y(data_path+"test")