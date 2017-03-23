import sys

from matplotlib import rcParams

from src.generation.gen_voxels import generate
from src.config import models
from src.fitting.fit_models import fit_all_models
from src.dataset import Dataset
from src.routes import config_file_path
from src.machine_learning.entry import train_and_evaluate_all_datasets
from src.visualisation.entry import visualisation_entry

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

if __name__ == "__main__":

  rcParams.update({'figure.autolayout': True})
  if len(sys.argv) < 2:
    usage()
  arg = sys.argv[1]
  if arg == "generate":
      generate()
  elif arg == "fit-all":
      fit_all_models()
  elif arg == "train-all":
    train_and_evaluate_all_datasets()
  elif arg == "visualise":
    visualisation_entry()
  else:
    usage()
