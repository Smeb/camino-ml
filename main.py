import sys

from src.generation.gen_voxels import generate
from src.config import models
from src.fit_models import fit_model_voxels
from src.dataset import Dataset
from src.routes import config_file_path
from src.machine_learning.entry import entry

# Silences the deprecation warning from scikit
def warn(*args, **kwargs):
  pass
import warnings
warnings.warn = warn

def usage():
  print "python main.py <cmd> where <cmd> is one of"
  print "  generate            -- to generate data"
  print "  train-all           -- to train all models using generated data"
  print
  print "Models can be configured in src/config.py; compartment names"
  print "must match those in camino_compartments, which is defined in the same file"

def all_models():
  for model in models:
    yield Dataset.from_model(model)

def train_models():
  for dataset in all_models():
    print("Training {}".format(dataset.name))
    entry(dataset)

def fit_all():
  for model in models:
    fit_model_voxels(model)

if __name__ == "__main__":
  if len(sys.argv) < 2:
    usage()
  arg = sys.argv[1]
  if arg == "generate":
      generate()
  elif arg == "fit-all":
      fit_all()
  elif arg == "train-all":
      train_models()
  else:
    usage()
