from sklearn import neighbors
def trainKNN(data):
  _, (trainX, trainY), _, _ = data
  knn = neighbors.KNeighborsRegressor(weights="distance")
  knn.fit(trainX, trainY)
  return knn
