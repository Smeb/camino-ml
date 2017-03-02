from src.mlmodels.scikit_learn.evaluation import evaluate
from src.mlmodels.scikit_learn.visualisation import visualise_difference
class Experiment:
  def __init__(self, name, method, data, data_model):
    self.name = name
    self.data = data
    self.model = method(data)
    self.data_model = data_model

  def predict(self, testX):
    return self.model.predict(testX)

  def evaluate(self):
    return evaluate(self.data, self.model)

  def visualise(self):
    visualise_difference(self.data, self.model, self.name, self.data_model)
