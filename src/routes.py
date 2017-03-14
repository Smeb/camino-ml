import os
import errno


root_path = os.path.abspath(os.getcwd())
config_file_path = root_path + '/config.json'
datasynth_path = root_path + '/camino/bin/datasynth'
float2txt_path = root_path + '/camino/bin/float2txt'
scheme_path = root_path + '/PGSE_90.scheme'
data_path = root_path + '/data'
media_path = root_path + '/media'

def make_path_ignoring_existing(path):
  try:
    os.makedirs(path)
  except OSError as exc:
    if exc.errno == errno.EEXIST and os.path.isdir(path):
      pass
    else:
      raise

