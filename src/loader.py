import collections
import json
from sklearn.preprocessing import StandardScaler

from src.routes import data_path
from src.biomodels.model import Model

def load_file(fname):
  with open(fname) as f:
    configs = collections.OrderedDict()
    obj = json.JSONDecoder(object_pairs_hook=collections.OrderedDict).decode(f.read())
    for item in obj:
      configs[item] = DatasetConfig(item, obj)
    return configs

def loadParams(path):
  import numpy as np
  return np.genfromtxt(path)

def compareFnames(a, b):
  """
  Sorts the filenames as numbers, removing the .float extension
  """
  import os
  aName = int(os.path.splitext(a)[0])
  bName = int(os.path.splitext(b)[0])
  return 1 if aName > bName else 0 if aName == bName else -1

def loadData(config):
  import os
  import pandas
  import numpy as np
  from sklearn.model_selection import train_test_split

  path = "{}/{}".format(data_path, config.name)

  floatFiles = [filename for filename in os.listdir(path) if filename.endswith(".float")]
  floatFiles.sort(compareFnames)

  vectors = [None] * len(floatFiles)

  feature_names = []
  for compartment in config.compartments:
    for name in compartment._paramNames:
      feature_names.append(compartment._name + '_' + name)

  for fname in floatFiles:
    voxArray = np.genfromtxt("{}/{}".format(path, fname))
    voxNumber = int(filter(str.isdigit, fname))
    vectors[voxNumber] = voxArray.flatten()
  scaler = StandardScaler()
  ground_truth = np.genfromtxt("{}/{}.params".format(path, config.name))
  ground_truth = scaler.fit_transform(ground_truth)
  trainX, testX, trainY, testY = train_test_split(vectors, ground_truth, test_size=0.2)
  return scaler, (trainX, trainY), (testX, testY), feature_names

class DatasetConfig:
  def build_compartments(self, models):
    self.compartments = []
    for model in models:
      self.compartments.append(Model(model, models[model]))

  def __init__(self, dataset_name, obj):
    self.name = dataset_name
    self.build_compartments(obj[dataset_name]['models'])
