import os
import split_folders

IN_PATH = "dataset/unsorted"
OUT_PATH = "dataset/split"
split_folders.ratio(input=IN_PATH, output=OUT_PATH, seed=1337, ratio=(.8, .1, .1))