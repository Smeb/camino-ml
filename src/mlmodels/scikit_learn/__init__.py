from src.mlmodels.experiment import Experiment

from .visualisation import visualise_evaluations

from .random_forest import trainRF
from .knn import trainKNN
from .svm import trainSVM
from .neural_nets import trainMLP
from .ridge_regression import trainRidge

# TODO: split out model training from model evaluation and
# visualisation. Shouldn't have to make a new model each time (use pickle?)
def run_experiments(data, data_model):
  experiments = []
  experiments.append(Experiment('RandomForest', trainRF, data, data_model))
  experiments.append(Experiment('MultiLayerPerceptron', trainMLP, data, data_model))
  experiments.append(Experiment('SVM', trainSVM, data, data_model))
  experiments.append(Experiment('KNN', trainKNN, data, data_model))
  experiments.append(Experiment('RidgeRegression', trainRidge, data, data_model))
  return experiments

def entry(data, data_model):
  experiments = run_experiments(data, data_model)
  evaluations = []
  for experiment in experiments:
    evaluations.append((experiment, experiment.evaluate()))
    experiment.visualise()

def evaluate_all(datum):
  evaluations = []
  for data, data_model in datum:
    experiments = run_experiments(data, data_model)
    for experiment in experiments:
      evaluations.append((experiment, experiment.evaluate()))
  visualise_evaluations(evaluations)
