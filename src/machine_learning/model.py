class Model:
  def __init__(self, algorithm_name, algorithm, train_dataset, kwargs):
    self.name = "{}_{}".format(algorithm_name, train_dataset.name)
    self.model = algorithm(train_dataset, **kwargs)
    self.train_dataset = train_dataset

  def predict(self, test_X):
    return self.model.predict(test_X)
