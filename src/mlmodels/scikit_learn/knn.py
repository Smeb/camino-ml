from sklearn import neighbors
def trainKNN(data):
  scaler, (trainX, trainY), (testX, testY), _ = data
  knn = neighbors.KNeighborsRegressor(weights="distance")
  knn.fit(trainX, trainY)
  return knn
