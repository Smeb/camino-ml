import src.loader as loader
import sys

from src.gen_voxels import generate
from src.config import models
from src.model_factory import ModelFactory
from src.routes import config_file_path
from src.mlmodels.tensorflow.entry import entry

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

def train_models():
  factory = ModelFactory()
  datum = [loader.load_float_data(model, factory) for model in models]
  for data, data_model in datum:
    print("Training {}".format(data_model))
    entry(data, data_model)

if __name__ == "__main__":
  if len(sys.argv) < 2:
    usage()
  arg = sys.argv[1]
  if arg == "generate":
      generate()
  elif arg == "train-all":
      train_models()
  elif arg == "evaluate-all":
      eval_all()
  else:
    usage()
