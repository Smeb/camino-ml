"""svr.py
    Contains code interfacing with scikit-learn which returns a
    SupportVectorRegressor model fit to the input training data
"""
from sklearn.svm import SVR, LinearSVR, NuSVR
from sklearn.multioutput import MultiOutputRegressor

SVR_CACHE = 1000

def train_epsilon_svr(dataset):
    """Returns a fitted SupportVectorRegressor model using the epsilon
    kernel"""
    svm_m = MultiOutputRegressor(SVR(cache_size=SVR_CACHE))
    svm_m.fit(dataset.train_x, dataset.train_y)
    return svm_m

def train_linear_svr(dataset):
    """Returns a fitted SupportVectorRegressor model using the linear
    kernel"""
    svm_m = MultiOutputRegressor(LinearSVR())
    svm_m.fit(dataset.train_x, dataset.train_y)
    return svm_m

def train_nu_svr(dataset):
    """Returns a fitted SupportVectorRegressor model using the nu
    kernel"""
    svm_m = MultiOutputRegressor(NuSVR(cache_size=SVR_CACHE))
    svm_m.fit(dataset.train_x, dataset.train_y)
    return svm_m
