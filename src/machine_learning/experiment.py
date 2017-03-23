from __future__ import print_function
import os
import errno

import pandas
from sklearn import metrics

from src.routes import make_path_ignoring_existing, media_path
from .visualisation import (
  visualise_param_v_param,
  visualise_bland_altman,
  visualise_difference,
  )

class Experiment:
  def __init__(self, model, test_dataset):
    self.model = model
    self.test_dataset = test_dataset

    # Predictions are scaled by the train dataset scaler
    self.unscaled_predict_Y = self.model.train_dataset.inverse_transform(self.model.predict(test_dataset.test_X))
    self.unscaled_test_Y = self.model.train_dataset.inverse_transform(test_dataset.test_Y)

    self.make_media_path()

  def make_media_path(self):
    self.media_path = '{}/{}-{}'.format(media_path, self.model.train_dataset.name, self.test_dataset.name)
    make_path_ignoring_existing(self.media_path)

  def results_dataframe(self):
    if self.df:
      return self.df
    self.df = pandas.DataFrame(
      columns=['algorithm',
      'mean_absolute_error',
      'mean_squared_error',
      'r2_score'] + self.test_dataset.feature_names)
    return self.df

  def evaluate(self):
    df = self.results_dataframe

    row = [self.model.name]
    row.append(metrics.mean_absolute_error(self.unscaled_test_Y, self.unscaled_predict_Y))
    row.append(metrics.mean_squared_error(self.unscaled_test_Y, self.unscaled_predict_Y))
    row.append(metrics.r2_score(self.unscaled_test_Y, self.unscaled_predict_Y))

    for param in self.test_dataset.feature_names:
      row.append(metrics.r2_score(self.unscaled_test_Y[param],
        self.unscaled_predict_Y[param]))


  def visualise(self):
    visualise_param_v_param(self.unscaled_test_Y,
      self.unscaled_predict_Y,
      self.test_dataset.feature_names,
      self.model.name,
      self.media_path)

    visualise_bland_altman(self.unscaled_test_Y,
      self.unscaled_predict_Y,
      self.test_dataset.feature_names,
      self.model.name, self.media_path)
