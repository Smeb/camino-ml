import collections
import json
from src.biomodels.model import Model

def load_file(fname):
  with open(fname) as f:
    configs = []
    obj = json.JSONDecoder(object_pairs_hook=collections.OrderedDict).decode(f.read())
    for item in obj:
      configs.append(DatasetConfig(item, obj))
    return configs

class DatasetConfig:
  def build_compartments(self, models):
    self.compartments = []
    for model in models:
      self.compartments.append(Model(model, models[model]))

  def __init__(self, dataset_name, obj):
    self.name = dataset_name
    self.build_compartments(obj[dataset_name]['models'])

def loadParams(path):
  import numpy as np
  return np.genfromtxt(path)


def loadData(path):
  import os
  import numpy as np

  BFloatFiles = [filename for filename in os.listdir(path) if filename.endswith(".float")]
  vectors = [None] * len(BFloatFiles)

  # Prevents error in case of no float files
  for fname in BFloatFiles:
    voxArray = np.genfromtxt("{}{}".format(path, fname))
    voxNumber = int(filter(str.isdigit, fname))
    vectors[voxNumber] = voxArray.flatten()
  finalArray = np.array(vectors)
  return finalArray
