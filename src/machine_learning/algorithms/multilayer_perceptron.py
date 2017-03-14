from sklearn.neural_network import MLPRegressor

def train_MLP(data):
  scaler, (trainX, trainY), (testX, testY), _ = data
  mlp = MLPRegressor()
  mlp.fit(trainX, trainY)
  return mlp
