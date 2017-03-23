from sklearn import neighbors
def train_KNN(dataset):
  knn = neighbors.KNeighborsRegressor(weights="distance")
  knn.fit(dataset.train_X, dataset.train_Y)
  return knn
