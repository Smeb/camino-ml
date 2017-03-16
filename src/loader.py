import collections
import os

import pandas
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from src.config import definitions, models, strip
from src.routes import data_path
from src.generation.model_factory import get_dataset_path, get_param_names, get_model_name

def compare_fnames(a, b):
  """
  Sorts the filenames as numbers, removing the .float extension
  """
  import os
  aName = int(os.path.splitext(a)[0])
  bName = int(os.path.splitext(b)[0])
  return 1 if aName > bName else 0 if aName == bName else -1

def load_float_data(model):
  name = get_model_name(model)
  path = get_dataset_path(model)
  float_path = path + "/float"
  feature_names = get_param_names(model)

  skip_list = []
  for compartment in model:
    skip_list += ["{}_{}".format(compartment, item) for item in strip]

  floatFiles = [filename for filename in os.listdir(float_path)]
  floatFiles.sort(compare_fnames)

  vectors = [None] * len(floatFiles)

  for fname in floatFiles:
    voxArray = np.genfromtxt("{}/{}".format(float_path, fname))
    voxNumber = int(filter(str.isdigit, fname))
    vectors[voxNumber] = voxArray.flatten().tolist()
  ground_truth = np.genfromtxt("{}/{}.params".format(path, name))

  # Filters out parameters in the skip list
  ground_truth = pandas.DataFrame(ground_truth, columns=feature_names)
  ground_truth = ground_truth[ground_truth.columns.difference(skip_list)]
  feature_names = ground_truth.columns.tolist()

  scaler = StandardScaler()
  ground_truth = scaler.fit_transform(ground_truth)

  trainX, testX, trainY, testY = train_test_split(vectors, ground_truth, test_size=0.2)

  return scaler, trainX, trainY, testX, testY, feature_names, name
