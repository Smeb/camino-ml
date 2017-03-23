from sklearn.neural_network import MLPRegressor

def train_MLP(dataset):
  mlp = MLPRegressor()
  mlp.fit(dataset.train_X, dataset.train_Y)
  return mlp
