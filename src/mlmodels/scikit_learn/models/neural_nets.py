from sklearn.neural_network import MLPRegressor

def trainMLP(data):
  scaler, (trainX, trainY), (testX, testY), _ = data
  mlp = MLPRegressor()
  mlp.fit(trainX, trainY)
  return mlp
