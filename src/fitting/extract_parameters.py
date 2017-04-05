from src.config import MODELS
from src.datasets.dataset_factory import CAMINO_COMPARTMENTS, get_model_name, get_parameter_names

def extract_all():
  pass

def extract_model(model):
  model_name = get_model_name(model)
  parameter_names = get_parameter_names()
  with open(f_path, 'r') as open_file:
    lines = open_file.readlines()

def extract_fit():
  pass
