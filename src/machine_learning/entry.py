import os
import errno

from tqdm import tqdm

from src.config import models
from src.dataset import Dataset
from src.routes import make_path_ignoring_existing, media_path

from .experiment import Experiment
from .model import Model
from .algorithms.random_forest import train_RF
from .algorithms.knn import train_KNN
from .algorithms.svr import train_linear_SVR
from .algorithms.multilayer_perceptron import train_MLP
from .algorithms.ridge_regression import train_ridge
from .algorithms.convolutional_nn import initialise_nn_grid_search, search_convolutional_nn

algorithms = [
    ('RandomForest', train_RF, {}),
    ('MultiLayerPerceptron', train_MLP, {}),
    ('linearSVR', train_linear_SVR, {}),
    ('KNN', train_KNN, {}),
    ('RidgeRegression', train_ridge, {}),
]# + initialise_nn_grid_search(range(5, 26, 2), range(2, 5))

def list_to_lower(lst):
  return [item.lower() for item in lst]

def all_datasets():
  for model in models:
    yield Dataset.from_model(list_to_lower(model))

def gen_experiments(dataset):
  experiments = []
  for function_name, function, kwargs in tqdm(algorithms):
    experiments.append(Experiment(Model(function_name, function, dataset, kwargs), dataset))
  return experiments

def train_and_evaluate_all_datasets():
  make_path_ignoring_existing(media_path)

  for dataset in all_datasets():
    experiments = gen_experiments(dataset)
    for experiment in experiments:
      experiment.evaluate()
      experiment.visualise()
