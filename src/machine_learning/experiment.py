from __future__ import print_function
import os
import errno

from sklearn import metrics

from .visualisation import (
  visualise_param_v_param,
  visualise_bland_altman,
  visualise_difference,
  )
from src.routes import make_path_ignoring_existing, media_path

class Experiment:
  def __init__(self, method_name, method, data, dataset_name):
    self.method_name = method_name
    self.data = data
    self.model = method(data)
    self.dataset_name = dataset_name
    self.make_media_path()

  def predict(self):
    scaler, (_, _), (testX, testY), feature_names = self.data
    prediction = self.model.predict(testX)
    return scaler.inverse_transform(testY), scaler.inverse_transform(prediction)

  def make_media_path(self):
    self.dataset_path = "{}/{}".format(media_path, self.dataset_name)
    make_path_ignoring_existing(self.dataset_path)
    open('{}/evaluation'.format(self.dataset_path), 'w').close()

  def evaluate(self, test_Y, predict_Y):
    mean_absolute_error = metrics.mean_absolute_error(test_Y, predict_Y)
    mean_squared_error = metrics.mean_squared_error(test_Y, predict_Y)
    r2_score = metrics.r2_score(test_Y, predict_Y)

    with open('{}/evaluation'.format(self.dataset_path), 'a+') as f:
      print(self.method_name, file=f)
      print('Mean absolute error: {}'.format(mean_absolute_error), file=f)
      print('Mean squared error: {}'.format(mean_squared_error), file=f)
      print('r2 score: {}'.format(r2_score), file=f)
      print(file=f)

    return mean_absolute_error, mean_squared_error, r2_score

  def visualise(self, test_Ys, predict_Ys):
    _, _, _, feature_names = self.data
    visualise_param_v_param(test_Ys, predict_Ys, feature_names, self.method_name, self.dataset_path)
    visualise_bland_altman(test_Ys, predict_Ys, feature_names, self.method_name, self.dataset_path)
