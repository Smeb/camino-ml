from sklearn.linear_model import Ridge
def train_ridge(dataset):
  ridge_regressor = Ridge()
  ridge_regressor.fit(dataset.train_X, dataset.train_Y)
  return ridge_regressor

