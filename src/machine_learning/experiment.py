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
  def __init__(self, model, test_dataset):
    self.model = model
    self.test_dataset = test_dataset

    # Predictions are scaled by the train dataset scaler
    self.unscaled_predict_Y = self.model.train_dataset.inverse_transform(self.model.predict(test_dataset.test_X))
    self.unscaled_test_Y = self.model.train_dataset.inverse_transform(test_dataset.test_Y)

    self.make_media_path()

  def make_media_path(self):
    self.dataset_path = "{}/{}".format(media_path, self.test_dataset.name)
    make_path_ignoring_existing(self.dataset_path)
    open('{}/evaluation'.format(self.dataset_path), 'w').close()

  def evaluate(self):
    mean_absolute_error = metrics.mean_absolute_error(self.unscaled_test_Y, self.unscaled_predict_Y)
    mean_squared_error = metrics.mean_squared_error(self.unscaled_test_Y, self.unscaled_predict_Y)
    r2_score = metrics.r2_score(self.unscaled_test_Y, self.unscaled_predict_Y)

    with open('{}/evaluation'.format(self.dataset_path), 'a+') as f:
      print(self.model.name, file=f)
      print('Mean absolute error: {}'.format(mean_absolute_error), file=f)
      print('Mean squared error: {}'.format(mean_squared_error), file=f)
      print('r2 score: {}'.format(r2_score), file=f)
      print(file=f)

    return mean_absolute_error, mean_squared_error, r2_score

  def visualise(self):
    visualise_param_v_param(self.unscaled_test_Y,
      self.unscaled_predict_Y,
      self.test_dataset.feature_names,
      self.model.name,
      self.dataset_path)

    visualise_bland_altman(self.unscaled_test_Y,
      self.unscaled_predict_Y,
      self.test_dataset.feature_names,
      self.model.name,
      self.dataset_path)
