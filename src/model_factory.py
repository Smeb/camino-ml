from __future__ import print_function
import errno
import os
import random
from subprocess import call
from collections import OrderedDict

from tqdm import tqdm
from config import camino_compartments, definitions, dataset_size
from routes import data_path, datasynth_path, float2txt_path, scheme_path

class ModelFactory:
  def __init__(self):
    self.camino_compartments = camino_compartments
    self.definitions = definitions

  def gen_name(self, compartments):
    return "".join(compartments) + "_{}".format(dataset_size)

  def get_dataset_path(self, model):
    return "{}/{}".format(data_path,self.gen_name(model))

  def get_param_names(self, model):
    param_names = []
    for index, compartment in enumerate(model):
      params = camino_compartments[compartment]
      param_names.append("{}_ivf".format(compartment))
      for param in params:
        param_names.append("{}_{}".format(compartment, param))
    print(param_names)
    return param_names

  def gen_model(self, compartments):
    name = self.gen_name(compartments)
    output_path = "{}/{}".format(data_path, name)
    try:
      os.makedirs(output_path)
    except OSError as exc:
      if exc.errno == errno.EEXIST and os.path.isdir(data_path):
        print("Dataset {} exists, skipping generation".format(name))
        return
      else:
        raise

    self.write_spec(compartments, name)

    with open("{}/{}.params".format(output_path, name), 'w') as param_file:
      with open("{}/{}.log".format(output_path, name), 'w') as log_file:
        print("Generating {} voxels for model {}".format(dataset_size, name))
        for i in tqdm(range(dataset_size)):
          model = self.init_model(compartments)
          self.gen_voxel(model, "{}/{}.float".format(output_path, str(i)), log_file)
          self.write_params(model, param_file)

  def gen_voxel(self, model, output_path, log):
    cmd = ["{} -synthmodel compartment {}".format(datasynth_path, len(model))]
    for compartment in model:
      cmd.append("{} {}".format(compartment, self.stringify_params(model, compartment)))
    cmd.append("-schemefile {} -voxels 1".format(scheme_path))
    cmd.append("| {} > {}".format(float2txt_path, output_path))
    call(" ".join(cmd), shell=True, stderr=log)

  def write_params(self, model, param_file):
        for compartment in model:
          print(self.stringify_params(model, compartment), end=" ", file=param_file)
        print(file=param_file)

  def stringify_params(self, model, compartment):
      return " ".join(str(model[compartment][param]) for param in model[compartment])

  def write_spec(self, compartments, name):
    with open("{}/{}/{}.spec".format(data_path, name, name), 'w') as definition_file:
      for compartment in compartments:
        print("{} : {}".format(compartment,
        camino_compartments[compartment]), file=definition_file)
      print(file=definition_file)
      print(self.definitions, file=definition_file)

  def init_model(self, compartments):
    model = OrderedDict()
    run = self.gen_run()

    n_compartments = len(compartments)
    max_ivf = 1.0
    for index, compartment in enumerate(compartments):
      if len(compartments) - 1 != index:
        ivf = random.uniform(0, max_ivf)
        max_ivf -= ivf
      else:
        ivf = max_ivf
      model[compartment] = self.gen_compartment(camino_compartments[compartment], run, ivf)
    return model

  def gen_compartment(self, compartmentParams, run, ivf):
    compartment = OrderedDict()
    compartment['ivf'] = ivf
    for param in compartmentParams:
      compartment[param] = run[param]
    return compartment

  def gen_run(self):
    run = dict()

    for name, param in self.definitions.iteritems():
      if name == 'd':
        self.gen_diffusivities(run)
      elif type(param) == list:
        run[name] = random.uniform(param[0], param[1])
      else:
        run[name] = param
    return run

  def gen_diffusivities(self, run):
    from numpy.linalg import norm
    d = self.definitions['d']
    if type(d) is float:
      run['d'] = d
      run['d_perp1'] = d
      run['d_perp2'] = d
    else:
      d_sample = random.uniform(d[0], d[1])
      run['d'] = d_sample
      run['d_perp1'] = d_sample
      run['d_perp2'] = d_sample
