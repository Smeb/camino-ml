import pandas

from src.loader import load_float_data

class Dataset:
  def __init__(self, model, scaler, train_X, train_Y, test_X, test_Y,
  feature_names, name):
    self.name = name
    self.model = model

    self.scaler = scaler
    self.train_X = train_X
    self.train_Y = train_Y
    self.test_X = test_X
    self.test_Y = test_Y

    self.feature_names = feature_names

  def inverse_transform(self, vector):
    return pandas.DataFrame(self.scaler.inverse_transform(vector), columns=self.feature_names)

  @classmethod
  def from_model(cls, model):
    scaler, train_X, train_Y, test_X, test_Y, feature_names, name = load_float_data(model)
    return cls(model, scaler, train_X, train_Y, test_X, test_Y, feature_names, name)
