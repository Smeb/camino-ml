"""divide_dataset.py
    Divides a generated dataset into test and train sets for reproducibility
"""
from __future__ import print_function

import os
import shutil

import numpy as np
from sklearn.model_selection import train_test_split

from src.config import MODELS
from src.datasets.dataset_factory import get_dataset_data_path, get_model_name
from src.routes import make_path_ignoring_existing
def compare_fnames(fname_a, fname_b):
    """Sorts filenames as numbers, removing the .float extension"""
    a_name = int(os.path.splitext(fname_a)[0])
    b_name = int(os.path.splitext(fname_b)[0])
    return 1 if a_name > b_name else 0 if a_name == b_name else -1

def divide_dataset():
  divide_model(MODELS[0])
  # for model in MODELS:
  #   divide_model(model)

def divide_model(model):
    name = get_model_name(model)
    data_path = get_dataset_data_path(model)

    float_path = data_path + "/float"
    float_train_path = float_path + '/train'
    float_test_path = float_path + '/test'

    bfloat_path = data_path + "/raw"
    bfloat_test_path = bfloat_path + '/test'

    float_files = os.listdir(float_path)
    float_files.sort(compare_fnames)
    bfloat_files = os.listdir(bfloat_path)
    bfloat_files.sort(compare_fnames)

    float_train, float_test, _, bfloat_test= train_test_split(
        float_files, bfloat_files, test_size=0.2)

    # Sorting insures that the ground truth is correctly ordered for each file
    float_train.sort(compare_fnames)
    float_test.sort(compare_fnames)
    bfloat_test.sort(compare_fnames)

    train_ground_truth = []
    test_ground_truth = []

    make_path_ignoring_existing(float_train_path)
    make_path_ignoring_existing(float_test_path)
    make_path_ignoring_existing(bfloat_test_path)

    with open("{}/{}.params".format(data_path, name)) as f:
      ground_truth = f.readlines()
      with open("{}/{}.params".format(float_train_path, name), 'w') as f_train:
        for float_file in float_train:
            index = int(os.path.splitext(float_file)[0])
            print(ground_truth[index], file=f_train, end="")
            shutil.move("{}/{}".format(float_path, float_file),
                        "{}/{}".format(float_train_path, float_file))
      with open("{}/{}.params".format(float_test_path, name), 'w') as f_test:
        for float_file, bfloat_file in zip(float_test, bfloat_test):
            index = int(os.path.splitext(float_file)[0])
            print(ground_truth[index], file=f_test, end="")
            shutil.move("{}/{}".format(float_path, float_file),
                        "{}/{}".format(float_test_path, float_file))
            shutil.move("{}/{}".format(bfloat_path, bfloat_file),
                        "{}/{}".format(bfloat_test_path, bfloat_file))
