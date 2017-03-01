from __future__ import print_function
import errno
import os
import logging
import time
from subprocess import call, check_output

from tqdm import tqdm

from src.settings import dataset_size
from src.routes import data_path, datasynth_path, float2txt_path, scheme_path

def gen_data(configs):
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


def gen_voxels(config, n_files):
  dataset_path = '{}/{}'.format(data_path, config.name)
  os.makedirs(dataset_path)
  logfile_path = '{}/{}.log'.format(data_path, config.name)
  params_path = '{}/{}.params'.format(dataset_path, config.name)
  print('Generating {} voxels at {}'.format(config.name, dataset_path))
  with open(logfile_path, 'w+') as log:
    for i in tqdm(range(n_files)):
      cmd, params = build_command(config.compartments, '{}/{}.float'.format(dataset_path, str(i)))
      store_params(params, params_path)
      call(' '.join(cmd), shell=True, stderr=log)

def build_command(compartments, output_path):
  cmd = [datasynth_path, '-synthmodel']
  params = []
  cmd.append('compartment {}'.format(str(len(compartments))))
  for compartment in compartments:
    params.append(str(compartment))
  cmd += params
  cmd.append('-schemefile {} -voxels 1'.format(scheme_path))
  cmd.append('| {} > {}'.format(float2txt_path, output_path))
  return cmd, params

def store_params(params, param_file):
  if os.path.exists(param_file):
    f_flag = 'a'
  else:
    f_flag = 'w'
  with open(param_file, f_flag) as f:
    for param in params:
      print(param.partition(' ')[2], file=f)
