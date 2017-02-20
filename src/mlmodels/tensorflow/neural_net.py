import tensorflow as tf
from src.settings import data_path
import src.loader as loader

def entry(configs, modelname):
  config = None
  for cfg in configs:
    if modelname == cfg.name:
      config = cfg
  if config is None:
    raise Exception('Model not found in config')
  task(config)

def task(config):
  print(loader.transformData("{}/{}/".format(data_path, config.name)))
