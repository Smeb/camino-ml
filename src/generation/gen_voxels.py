from __future__ import print_function
import errno
import os
import logging
import time
from subprocess import call, check_output
import multiprocessing

from joblib import Parallel, delayed
from tqdm import tqdm

from .model_factory import gen_model
from src.config import models, dataset_size
from src.routes import (
  data_path,
  datasynth_path,
  float2txt_path,
  scheme_path,
  make_path_ignoring_existing)

def generate():
  make_path_ignoring_existing(data_path)
  Parallel(n_jobs=-1)(delayed(gen_model)(model, i) for model, i in zip(models, range(len(models))))
