import os
import sys
from progressbar import ProgressBar
from termcolor import colored

from models.model import Model
from subprocess import call, check_output
from settings import data_path, datasynth_path, scheme_path

def gen_voxels(config, n_files):
  dataset_path = '{}/{}'.format(data_path, config.name)
  os.makedirs(dataset_path)
  logfile_path = '{}/{}.log'.format(data_path, config.name)
  print(colored('Generating {} at {}'.format(config.name, dataset_path), 'red'))
  with open(logfile_path, 'w+') as log:
    with ProgressBar(max_value=n_files) as bar:
      for i in range(n_files):
        cmd = build_command(config.compartments, '{}/{}_{}.BFloat'.format(dataset_path, config.fname, str(i)))
        call(cmd, stderr=log)
        bar.update(i)

def build_command(compartments, output_path):
  cmd = [datasynth_path, '-synthmodel']
  cmd.append('compartment {}'.format(str(len(compartments))))
  for compartment in compartments:
    cmd.append(str(compartment))
  cmd.append('-schemefile {} -voxels 1'.format(scheme_path))
  cmd.append('-outputfile {}'.format(output_path))
  return cmd
