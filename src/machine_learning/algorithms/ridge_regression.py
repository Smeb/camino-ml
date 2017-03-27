"""ridge_regression.py
    Contains code interfacing with scikit-learn which returns a
    MultilayerPerceptron model fit to the input training data
"""
from sklearn.linear_model import Ridge
def train_ridge(dataset):
    """Returns a fitted RidgeRegression model"""
    ridge_regressor = Ridge()
    ridge_regressor.fit(dataset.train_x, dataset.train_y)
    return ridge_regressor
