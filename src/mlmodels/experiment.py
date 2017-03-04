from __future__ import print_function
from sklearn import metrics
from src.routes import media_path
from src.mlmodels.scikit_learn.visualisation import (
  visualise_param_v_param,
  visualise_difference)

class Experiment:
  def __init__(self, name, method, data, data_model):
    self.name = name
    self.data = data
    self.model = method(data)
    self.data_model = data_model

  def predict(self):
    scaler, (_, _), (testX, testY), feature_names = self.data
    prediction = self.model.predict(testX)
    return scaler.inverse_transform(testY), scaler.inverse_transform(prediction)

  def evaluate(self, test_Y, predict_Y):
    mean_absolute_error = metrics.mean_absolute_error(test_Y, predict_Y)
    mean_squared_error = metrics.mean_squared_error(test_Y, predict_Y)
    r2_score = metrics.r2_score(test_Y, predict_Y)

    with open('{}/{}_evaluation'.format(media_path, self.data_model), 'a+') as f:
      print(self.name, file=f)
      print('Mean absolute error: {}'.format(mean_absolute_error), file=f)
      print('Mean squared error: {}'.format(mean_squared_error), file=f)
      print('r2 score: {}'.format(r2_score), file=f)
      print(file=f)

    return mean_absolute_error, mean_squared_error, r2_score

  def visualise(self, test_Ys, predict_Ys):
    _, _, _, feature_names = self.data
    visualise_param_v_param(test_Ys, predict_Ys, feature_names, self.name, self.data_model)
    visualise_difference(test_Ys, predict_Ys, feature_names, self.name, self.data_model)
