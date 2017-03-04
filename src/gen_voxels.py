from __future__ import print_function
import errno
import os
import logging
import time
from subprocess import call, check_output

from tqdm import tqdm

from src.config import models, dataset_size
from src.model_factory import ModelFactory
from src.routes import data_path, datasynth_path, float2txt_path, scheme_path

def generate():
  try:
    os.makedirs(data_path)
  except OSError as exc:
    if exc.errno == errno.EEXIST and os.path.isdir(data_path):
      pass
    else:
      raise

  factory = ModelFactory()

  logfile_path = "{}/{}.log".format(data_path, time.strftime("%Y_%m_%d-%H_%M_%S"))
  logging.basicConfig(level=logging.INFO,
    filename=logfile_path,
    format='%(message)s',
    filemode='a+')

  logging.log(logging.INFO, 'Creating: {} datasets'.format(len(models)))

  for model in models:
    print('Generating {}'.format(model))
    start = time.time()
    name = factory.gen_model(model)
    end = time.time()
    logging.log(logging.INFO, '{0} {1} {2:.2}s'.format(name, dataset_size, end - start))
