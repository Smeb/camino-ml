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
  def __init__(self, algorithm_name, algorithm, dataset, kwargs):
    self.algorithm_name = algorithm_name
    self.dataset = dataset
    self.model = algorithm(dataset, **kwargs)
    self.make_media_path()

  def predict(self, dataset=None):
    if dataset is None:
      dataset = self.dataset

    test_X = dataset.test_X
    test_Y = dataset.test_Y

    prediction = self.model.predict(test_X)
    return dataset.inverse_transform(test_Y), dataset.inverse_transform(prediction)

  def make_media_path(self):
    self.dataset_path = "{}/{}".format(media_path, self.dataset.name)
    make_path_ignoring_existing(self.dataset_path)
    open('{}/evaluation'.format(self.dataset_path), 'w').close()

  def evaluate(self, test_Y, predict_Y):
    mean_absolute_error = metrics.mean_absolute_error(test_Y, predict_Y)
    mean_squared_error = metrics.mean_squared_error(test_Y, predict_Y)
    r2_score = metrics.r2_score(test_Y, predict_Y)

    with open('{}/evaluation'.format(self.dataset_path), 'a+') as f:
      print(self.algorithm_name, file=f)
      print('Mean absolute error: {}'.format(mean_absolute_error), file=f)
      print('Mean squared error: {}'.format(mean_squared_error), file=f)
      print('r2 score: {}'.format(r2_score), file=f)
      print(file=f)

    return mean_absolute_error, mean_squared_error, r2_score

  def visualise(self, test_Y, predict_Y):
    visualise_param_v_param(self.dataset, predict_Y, self.algorithm_name, self.dataset_path)
    visualise_bland_altman(test_Y, predict_Y, self.dataset.feature_names, self.algorithm_name, self.dataset_path)
