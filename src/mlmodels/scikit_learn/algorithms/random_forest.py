from sklearn.ensemble import RandomForestRegressor
def trainRF(data):
  scaler, (trainX, trainY), (testX, testY), _ = data
  rfReg = RandomForestRegressor()
  rfReg.fit(trainX, trainY)
  return rfReg
