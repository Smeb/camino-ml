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

def transformData(path):
  import os
  import numpy as np
  dt = np.dtype('>f')

  isExec = False
  BFloatFiles = [filename for filename in os.listdir(path) if filename.endswith(".Bfloat")]
  voxelCount = len(BFloatFiles)

  # Prevents error in case of no BFloat files
  finalArray = []
  for fname in BFloatFiles:
    voxArray = np.fromfile("{}/{}".format(path, fname), dtype=dt, sep="")
    if isExec == False:
      finalArray = np.full([voxelCount, len(voxArray)], 0)
      isExec = True
    voxArray = voxArray.reshape((1, voxArray.size))

    voxNumber = int(filter(str.isdigit, fname))
    finalArray[voxNumber] = voxArray
  return finalArray
