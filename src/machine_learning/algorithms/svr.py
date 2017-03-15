from src.settings import SVR_cache
from sklearn.svm import SVR, LinearSVR, NuSVR
from sklearn.multioutput import MultiOutputRegressor

def train_epsilon_SVR(dataset):
  svm_m = MultiOutputRegressor(SVR(cache_size=SVR_cache))
  svm_m.fit(dataset.train_X, dataset.train_Y)
  return svm_m

def train_linear_SVR(dataset):
  svm_m = MultiOutputRegressor(LinearSVR())
  svm_m.fit(dataset.train_X, dataset.train_Y)
  return svm_m

def train_nu_SVR(dataset):
  svm_m = MultiOutputRegressor(NuSVR(cache_size=SVR_cache))
  svm_m.fit(dataset.train_X, dataset.train_Y)
  return svm_m
