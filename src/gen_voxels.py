from __future__ import print_function
import os
import sys
from progressbar import ProgressBar
from termcolor import colored

from subprocess import call, check_output
from src.settings import data_path, datasynth_path, float2txt_path, scheme_path

def gen_voxels(config, n_files):
  dataset_path = '{}/{}'.format(data_path, config.name)
  os.makedirs(dataset_path)
  logfile_path = '{}/{}.log'.format(data_path, config.name)
  params_path = '{}/{}.params'.format(dataset_path, config.name)
  print(colored('Generating {} at {}'.format(config.name, dataset_path), 'red'))
  with open(logfile_path, 'w+') as log:
    with ProgressBar(max_value=n_files) as bar:
      for i in range(n_files):
        cmd, params = build_command(config.compartments, '{}/{}.float'.format(dataset_path, str(i)))
        store_params(params, params_path)
        call(' '.join(cmd), shell=True, stderr=log)
        print(cmd)
        bar.update(i)

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
