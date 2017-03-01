from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor

def trainSVM(data):
  scaler, (trainX, trainY), (testX, testY), _ = data
  svm_m = MultiOutputRegressor(SVR(kernel="linear"))
  svm_m.fit(trainX, trainY)
  return svm_m
