import os
import errno

from tqdm import tqdm

from src.routes import media_path
from src.mlmodels.experiment import Experiment
from .algorithms.convolutional_nn import train_convolutional_nn

algorithms = [
  ('ConvolutionalNN', train_convolutional_nn),
]

def gen_experiments(data, data_model):
  experiments = []
  for name, function in tqdm(algorithms):
    experiments.append(Experiment(name, function, data, data_model))
  return experiments

def entry(data, data_model):
  try:
    os.makedirs(media_path)
  except OSError as exc:
    if exc.errno == errno.EEXIST and os.path.isdir(media_path):
      pass
    else:
      raise

  experiments = gen_experiments(data, data_model)
  evaluations = []

  for experiment in experiments:
    test_Ys, predict_Ys = experiment.predict()
    evaluations.append((experiment, experiment.evaluate(test_Ys, predict_Ys)))
    experiment.visualise(test_Ys, predict_Ys)

def evaluate_all(datum):
  evaluations = []
  for data, data_model in datum:
    experiments = gen_experiments(data, data_model)
    for experiment in experiments:
      evaluations.append((experiment, experiment.evaluate()))
  visualise_evaluations(evaluations)
