import os
import errno


root_path = os.path.abspath(os.getcwd())
config_file_path = root_path + '/config.json'
scheme_path = root_path + '/PGSE_90.scheme'
data_path = root_path + '/data'
media_path = root_path + '/media'

camino_bin_path = root_path + '/camino/bin'

datasynth_path = camino_bin_path + '/datasynth'
modelfit_path = camino_bin_path + '/modelfit'

float2txt_path = camino_bin_path + '/float2txt'
double2txt_path = camino_bin_path + '/double2txt'

def make_path_ignoring_existing(path):
  try:
    os.makedirs(path)
  except OSError as exc:
    if exc.errno == errno.EEXIST and os.path.isdir(path):
      pass
    else:
      raise

