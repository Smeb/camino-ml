import datetime
import errno
import logging
import os
import time
import src.loader as loader
import sys
from src.gen_voxels import gen_voxels
from src.settings import config_file_path, dataset_size, data_path
from src.mlmodels.tensorflow.neural_net import entry

def usage():
  print("python main.py <cmd> <option> where <cmd> is one of")
  print("  gen_data            -- to generate data")
  print("  train <modelname>   -- to train a model (data must be generated first)")

def gen_data():
  configs = loader.load_file(config_file_path)
  try:
    os.makedirs(data_path)
  except OSError as exc:
    if exc.errno == errno.EEXIST and os.path.isdir(data_path):
      pass
    else:
      raise
  logfile_path = "{}/{}.log".format(data_path, time.strftime("%Y_%m_%d-%H_%M_%S"))
  logging.basicConfig(level=logging.INFO,
    filename=logfile_path,
    format='%(message)s',
    filemode='w')

  logging.log(logging.INFO, 'Creating: {} datasets'.format(len(configs)))

  for config in configs:
    start = time.time()
    gen_voxels(config, dataset_size)
    end = time.time()
    logging.log(logging.INFO, '{0} {1} {2:.2}s'.format(config.name, dataset_size, end - start))

def train_model(model):
  configs = loader.load_file(config_file_path)
  entry(configs, model)

if __name__ == "__main__":
  if len(sys.argv) < 2:
    usage()
  arg = sys.argv[1]
  if arg == "gen_data":
    gen_data()
  elif arg == "train":
    if len(sys.argv) < 3:
      usage()
    else:
      train_model(sys.argv[2])
  else:
    usage()
