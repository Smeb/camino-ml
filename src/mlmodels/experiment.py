from src.mlmodels.scikit_learn.evaluation import evaluate
from src.mlmodels.scikit_learn.visualisation import visualise
class Experiment:
  def __init__(self, name, method, data):
    self.name = name
    self.data = data
    self.model = method(data)

  def predict(self, testX):
    return self.model.predict(testX)

  def evaluate(self):
    print("Evaluation for {}".format(self.name))
    evaluate(self.data, self.model)

  def visualise(self):
    visualise(self.data, self.model, self.name)
