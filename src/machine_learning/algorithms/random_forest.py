from sklearn.ensemble import RandomForestRegressor
def train_RF(dataset):
  rfReg = RandomForestRegressor()
  rfReg.fit(dataset.train_X, dataset.train_Y)
  return rfReg
