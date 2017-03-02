import src.loader as loader
import sys

from src.gen_voxels import gen_data
from src.routes import config_file_path
from src.mlmodels.scikit_learn import entry, evaluate_all

def warn(*args, **kwargs):
  pass
import warnings
warnings.warn = warn

def usage():
  print("python main.py <cmd> <option> where <cmd> is one of")
  print("  gen_data            -- to generate data")
  print("  train <modelname>   -- to train a model (data must be generated first)")

def generate():
  configs = loader.load_file(config_file_path)
  print(configs)
  gen_data(configs)

def train_model(data_model):
  configs = loader.load_file(config_file_path)
  data = loader.loadData(configs[data_model])
  entry(data, data_model)

def eval_all():
  configs = loader.load_file(config_file_path)
  datum = [(loader.loadData(configs[data_model]), data_model) for data_model in configs.keys()]
  evaluate_all(datum)

if __name__ == "__main__":
  if len(sys.argv) < 2:
    usage()
  arg = sys.argv[1]
  if arg == "gen_data":
    generate()
  elif arg == "train":
    if len(sys.argv) < 3:
      usage()
    else:
      train_model(sys.argv[2])
  elif arg == "evaluate-all":
      eval_all()
  else:
    usage()
