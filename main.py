import datetime
import logging
import os
import time
import src.loader as loader
from src.gen_voxels import gen_voxels
from src.settings import config_file_path, dataset_size, data_path

if __name__ == "__main__":
  configs = loader.load_file(config_file_path)
  os.makedirs(data_path, exist_ok=True)
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
