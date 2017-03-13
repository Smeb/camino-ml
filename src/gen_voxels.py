from __future__ import print_function
import errno
import os
import logging
import time
from subprocess import call, check_output
import multiprocessing

from joblib import Parallel, delayed
from tqdm import tqdm

from src.config import models, dataset_size
from src.model_factory import gen_model
from src.routes import data_path, datasynth_path, float2txt_path, scheme_path

def generate():
  try:
    os.makedirs(data_path)
  except OSError as exc:
    if exc.errno == errno.EEXIST and os.path.isdir(data_path):
      pass
    else:
      raise

  Parallel(n_jobs=-1)(delayed(gen_model)(model, i) for model, i in zip(models, range(len(models))))
