import collections
import os

import pandas
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from src.config import definitions, models
from src.routes import data_path
from src.model_factory import ModelFactory

def compare_fnames(a, b):
  """
  Sorts the filenames as numbers, removing the .float extension
  """
  import os
  aName = int(os.path.splitext(a)[0])
  bName = int(os.path.splitext(b)[0])
  return 1 if aName > bName else 0 if aName == bName else -1

def load_data(model, factory):
  name = factory.gen_name(model)
  path = factory.get_dataset_path(model)
  feature_names = factory.get_param_names(model)

  floatFiles = [filename for filename in os.listdir(path) if filename.endswith(".float")]
  floatFiles.sort(compare_fnames)

  vectors = [None] * len(floatFiles)

  for fname in floatFiles:
    voxArray = np.genfromtxt("{}/{}".format(path, fname))
    voxNumber = int(filter(str.isdigit, fname))
    vectors[voxNumber] = voxArray.flatten()
  scaler = StandardScaler()
  ground_truth = np.genfromtxt("{}/{}.params".format(path, name))
  ground_truth = scaler.fit_transform(ground_truth)
  trainX, testX, trainY, testY = train_test_split(vectors, ground_truth, test_size=0.2)
  return ((scaler, (trainX, trainY), (testX, testY), feature_names), name)
