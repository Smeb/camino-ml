from __future__ import print_function
import errno
import os
import random
from subprocess import call
from collections import OrderedDict
from time import sleep

from tqdm import tqdm

from src.config import definitions, dataset_size, signal_noise_ratio
from src.routes import data_path, datasynth_path, float2txt_path, scheme_path

# Datasynth uses different names than modelfit; the modelfit names are
# shorter so we prefer modelfit names
compartment_map = {
  'gdrcylinders': 'gammadistribradiicylinders',
  'cylinder': 'cylindergpd',
  'sphere': 'spheregpd',
}

camino_compartments = {
  "stick": ["d", "theta", "phi"],
  "cylinder": ["d", "theta", "phi", "R"],
  "gdrcylinders": ["k", "b", "d", "theta", "phi"],

  "ball": ["d"],
  "zeppelin": ["d", "theta", "phi", "d_perp1"],
  "tensor": ["d", "theta", "phi", "d_perp1", "d_perp2", "alpha"],

  "astrosticks": ["d"],
  "astrocylinders": ["d", "R"],
  "sphere": ["d", "Rs"],
  "dot": [],
}

def get_model_name(compartments):
  return "".join(compartments) + "_{}".format(dataset_size)  + "_{}".format(signal_noise_ratio)

def get_dataset_path(model):
  return "{}/{}".format(data_path, get_model_name(model))

def get_param_names(model):
  param_names = []
  for index, compartment in enumerate(model):
    params = camino_compartments[compartment]
    param_names.append("{}_ivf".format(compartment))
    for param in params:
      param_names.append("{}_{}".format(compartment, param))
  print(param_names)
  return param_names

def gen_model(compartments, position):
  compartments = [compartment.lower() for compartment in compartments]
  name = get_model_name(compartments)
  output_path = "{}/{}".format(data_path, name)
  try:
    os.makedirs(output_path)
    os.makedirs(output_path + '/raw')
    os.makedirs(output_path + '/float')
  except OSError as exc:
    if exc.errno == errno.EEXIST and os.path.isdir(data_path):
      print("Dataset {} exists, skipping generation".format(name))
      return
    else:
      raise

  write_spec(compartments, name)

  with open("{}/{}.params".format(output_path, name), 'w') as param_file:
    with open("{}/{}.log".format(output_path, name), 'w') as log_file:
      print("Generating {} voxels for model {}".format(dataset_size, name))
      for i in tqdm(range(dataset_size), position=position, desc=" ".join(compartments)):
        model = init_model(compartments)
        bfloat_path = "{}/raw/{}.Bfloat".format(output_path, str(i))
        gen_voxel(model, "{}/raw/{}.Bfloat".format(output_path, str(i)), log_file)
        convert_voxel(bfloat_path, "{}/float/{}.float".format(output_path, str(i)))
        write_params(model, param_file)

def convert_voxel(bfloat_path, output_path):
  call("cat {} | {} > {}".format(bfloat_path, float2txt_path, output_path), shell=True)

def gen_voxel(model, output_path, log):
  cmd = ["{} -synthmodel compartment {}".format(datasynth_path, len(model))]
  for index, compartment in enumerate(model):
    compartment_name = compartment
    if compartment in compartment_map:
      compartment_name = compartment_map[compartment]

    if len(model) - 1 == index:
      cmd.append("{} {}".format(compartment_name, stringify_params_no_ivf(model, compartment)))
    else:
      cmd.append("{} {}".format(compartment_name, stringify_params(model, compartment)))
  cmd.append("-schemefile {} -voxels 1 -snr {} > {}".format(scheme_path, signal_noise_ratio, output_path))
  call(" ".join(cmd), shell=True, stderr=log)

def write_params(model, param_file):
  for compartment in model:
    print(stringify_params(model, compartment), end=" ", file=param_file)
  print(file=param_file)

def stringify_params(model, compartment):
  return " ".join(str(model[compartment][param]) for param in model[compartment])

def stringify_params_no_ivf(model, compartment):
  return " ".join(str(model[compartment][param]) for param in model[compartment] if param is not "ivf")

def write_spec(compartments, name):
  with open("{}/{}/{}.spec".format(data_path, name, name), 'w') as definition_file:
    for compartment in compartments:
      print("{} : {}".format(compartment,
      camino_compartments[compartment]), file=definition_file)
    print(file=definition_file)
    print(definitions, file=definition_file)

def init_model(compartments):
  model = OrderedDict()
  run = gen_run()

  n_compartments = len(compartments)

  # for single compartment models, ivf = 1.0
  # for two compartment models, ivf1 = 0.35 - 0.65, ivf2 = 1 - ivf1
  # for three compartment models, ivf1 = 0.3 - 0.35, ivf2 = 0.3 - 0.35, ivf3 = 1 - (ivf1 + ivf2)
  max_ivf = 1.0
  if n_compartments == 2:
    ivf_range_min = 0.35
    ivf_range_max = 0.65
  else:
    ivf_range_min = 0.3
    ivf_range_max = 0.35

  for index, compartment in enumerate(compartments):
    if len(compartments) - 1 != index:
      ivf = random.uniform(ivf_range_min, ivf_range_max)
      max_ivf -= ivf
    else:
      ivf = max_ivf
    model[compartment] = gen_compartment(camino_compartments[compartment], run, ivf)
  return model

def gen_compartment(compartmentParams, run, ivf):
  compartment = OrderedDict()
  compartment['ivf'] = ivf
  for param in compartmentParams:
    compartment[param] = run[param]
  return compartment

def gen_run():
  run = dict()

  for name, param in definitions.iteritems():
    if name == 'd':
      gen_diffusivities(run)
    elif type(param) == list:
      run[name] = random.uniform(param[0], param[1])
    else:
      run[name] = param
  return run

def gen_diffusivities(run):
  from numpy.linalg import norm
  d = definitions['d']
  if type(d) is float:
    run['d'] = d
    run['d_perp1'] = d
    run['d_perp2'] = d
  else:
    d_sample = random.uniform(d[0], d[1])
    run['d'] = d_sample
    run['d_perp1'] = d_sample
    run['d_perp2'] = d_sample
