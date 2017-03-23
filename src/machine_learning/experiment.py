from __future__ import print_function
import os
import errno

import pandas
from sklearn import metrics

from src.config import uuid
from src.routes import make_path_ignoring_existing, media_path
from src.visualisation.visualisation import (
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

    self.media_path = '{}/{}-{}'.format(media_path, self.model.train_dataset.name, self.test_dataset.name)
    self.make_media_path()


  def make_media_path(self):
    make_path_ignoring_existing(self.media_path)

  def get_results_dataframe(self):
    self.results_path = '{}/results.csv'.format(self.media_path)
    if os.path.isfile(self.results_path):
      return pandas.DataFrame.from_csv(self.results_path)
    else:
      return pandas.DataFrame(
        columns=['uuid',
        'algorithm',
        'mean_absolute_error',
        'mean_squared_error',
        'r2_score'] + self.test_dataset.feature_names)
      self.df.to_csv(self.results_path)

  def evaluate(self):
    df = self.get_results_dataframe()


    row = [uuid, self.model.name]
    row.append(metrics.mean_absolute_error(self.unscaled_test_Y, self.unscaled_predict_Y))
    row.append(metrics.mean_squared_error(self.unscaled_test_Y, self.unscaled_predict_Y))
    row.append(metrics.r2_score(self.unscaled_test_Y, self.unscaled_predict_Y))

    for param in self.test_dataset.feature_names:
      row.append(metrics.r2_score(self.unscaled_test_Y[param],
        self.unscaled_predict_Y[param]))

    df.loc[len(df)] = row
    df.to_csv(self.results_path)

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
