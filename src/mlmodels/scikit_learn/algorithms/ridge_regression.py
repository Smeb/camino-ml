from sklearn.linear_model import Ridge
def trainRidge(data):
  _, (trainX, trainY), _, _ = data
  ridge_regressor = Ridge()
  ridge_regressor.fit(trainX, trainY)
  return ridge_regressor

