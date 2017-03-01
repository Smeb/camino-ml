import collections
import json

from src.routes import data_path
from src.biomodels.model import Model

def load_file(fname):
  with open(fname) as f:
    configs = dict()
    obj = json.JSONDecoder(object_pairs_hook=collections.OrderedDict).decode(f.read())
    for item in obj:
      configs[item] = DatasetConfig(item, obj)
    return configs

def loadParams(path):
  import numpy as np
  return np.genfromtxt(path)

def loadData(config):
  import os
  import pandas
  import numpy as np
  from sklearn.model_selection import train_test_split

  path = "{}/{}".format(data_path, config.name)

  floatFiles = [filename for filename in os.listdir(path) if filename.endswith(".float")]
  vectors = [None] * len(floatFiles)

  for fname in floatFiles:
    voxArray = np.genfromtxt("{}/{}".format(path, fname))
    voxNumber = int(filter(str.isdigit, fname))
    vectors[voxNumber] = voxArray.flatten()

  return train_test_split(pandas.DataFrame(vectors), test_size=0.2)

class DatasetConfig:
  def build_compartments(self, models):
    self.compartments = []
    for model in models:
      self.compartments.append(Model(model, models[model]))

  def __init__(self, dataset_name, obj):
    self.name = dataset_name
    self.build_compartments(obj[dataset_name]['models'])
