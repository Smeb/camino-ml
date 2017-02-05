import collections
import json
from models.model import Model

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
    self.fname = obj[dataset_name]['fname']
