from src.mlmodels.experiment import Experiment

from .evaluation import evaluate
from .visualisation import visualise

from .random_forest import trainRF
from .knn import trainKNN
from .svm import trainSVM
from .mlp import trainMLP

def entry(data):
  rf_experiment = Experiment('RandomForest', trainRF, data)
  mlp_experiment = Experiment('MultiLayerPerceptron', trainMLP, data)
  svm_experiment = Experiment('SVM', trainSVM, data)
  knn_experiment = Experiment('KNN', trainKNN, data)

  rf_experiment.evaluate()
  mlp_experiment.evaluate()
  svm_experiment.evaluate()
  knn_experiment.evaluate()

  rf_experiment.visualise()
  mlp_experiment.visualise()
  svm_experiment.visualise()
  knn_experiment.visualise()
