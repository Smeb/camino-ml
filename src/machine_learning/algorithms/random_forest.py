from sklearn.ensemble import RandomForestRegressor
def train_RF(data):
  scaler, (trainX, trainY), (testX, testY), _ = data
  rfReg = RandomForestRegressor()
  rfReg.fit(trainX, trainY)
  return rfReg
