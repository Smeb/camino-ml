import os
import pandas
import re

from src.config import MODELS
from src.datasets.dataset_factory import CAMINO_COMPARTMENTS, get_model_name, get_param_names, get_dataset_data_path, get_value_from_spec
from src.routes import RESULTS_PATH

def tryint(s):
  try:
    return int(s)
  except:
    return s

def alphanum_keys(s):
  return [tryint(c) for c in re.split('([0-9]+)', s)]

def extract_all():
  for model in MODELS:
    extract_model(model)

def extract_model(model):
  model_name = get_model_name(model)
  print(model_name)
  dataset_path = get_dataset_data_path(model)
  fit_path = dataset_path + '/camino_fits'
  spec_path = '{}/{}.spec'.format(dataset_path, model_name)

  reordered_model = reorder_model(model)
  parameter_names = get_formatted_parameters(reordered_model)

  fit_files = os.listdir(fit_path)
  fit_files.sort(key=alphanum_keys)
  results_data_frame = pandas.DataFrame(columns=reduce(lambda x,y : x + y, parameter_names))
  index = 0
  for fit_file in fit_files:
    file_num = int(fit_file.split('.')[0])
    try:
      fit_frame = pandas.DataFrame(extract_fit(fit_path + '/' + fit_file, parameter_names, reordered_model), index=[index])
    except:
      return
    index += 1
    results_data_frame = results_data_frame.append(fit_frame)

  scheme = get_value_from_spec(spec_path, 'Scheme')
  noise = get_value_from_spec(spec_path, 'Noise')
  size = get_value_from_spec(spec_path, 'N_voxels')
  unique_dataset_id = '{}_{}_{}'.format(model_name, size, noise)

  results_data_frame.to_csv('{}/{}/{}-{}/camino_fit.csv'.format(RESULTS_PATH, scheme, unique_dataset_id, unique_dataset_id))

def extract_fit(fit_file_path, parameter_names, model):
  model_size = len(model)
  param_dict = {}

  with open(fit_file_path) as f:
    lines = [float(value) for value in f.readlines()] # ignore the first two values - exit code and signal

    if int(lines[0]) != 0:
      raise Exception('Failure in fitting dataset')

    index = 2
    if len(model) != 1:
      for compartment in model:
        param_dict['{}_ivf'.format(compartment)] = lines[index]
        index += 1
    else:
        param_dict['{}_ivf'.format(model[0])] = 1.0

    for compartment_index, compartment in enumerate(model):
      param_index = 1
      while param_index != len(parameter_names[compartment_index]):
        param_dict[parameter_names[compartment_index][param_index]] = lines[index]
        param_index += 1
        index += 1
  return param_dict

def get_formatted_parameters(model):
  parameter_names = get_param_names(model)
  return split_parameters(model, parameter_names)

def reorder_model(model):
  reordered_model = model
  if len(model) > 1:
    tmp = model[0]
    reordered_model[0] = model[1]
    reordered_model[1] = tmp
  return reordered_model

def split_parameters(model, parameters):
  split_parameter_list = []
  for compartment in model:
    parameter_list = [parameter for parameter in parameters if parameter.startswith(compartment)]
    split_parameter_list.append(parameter_list)
  return split_parameter_list
