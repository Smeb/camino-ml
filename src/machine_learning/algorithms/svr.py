from src.settings import SVR_cache
from sklearn.svm import SVR, LinearSVR, NuSVR
from sklearn.multioutput import MultiOutputRegressor

def train_epsilon_SVR(data):
  scaler, (trainX, trainY), (testX, testY), _ = data
  svm_m = MultiOutputRegressor(SVR(cache_size=SVR_cache))
  svm_m.fit(trainX, trainY)
  return svm_m

def train_linear_SVR(data):
  scaler, (trainX, trainY), (testX, testY), _ = data
  svm_m = MultiOutputRegressor(LinearSVR())
  svm_m.fit(trainX, trainY)
  return svm_m

def train_nu_SVR(data):
  scaler, (trainX, trainY), (testX, testY), _ = data
  svm_m = MultiOutputRegressor(NuSVR(cache_size=SVR_cache))
  svm_m.fit(trainX, trainY)
  return svm_m
