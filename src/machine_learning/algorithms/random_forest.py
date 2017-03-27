"""random_forest.py
    Contains code interfacing with scikit-learn which returns a
    RandomForest model fit to the input training data
"""
from sklearn.ensemble import RandomForestRegressor
def train_rf(dataset):
    """Returns a fitted RandomForest model"""
    rf_reg = RandomForestRegressor()
    rf_reg.fit(dataset.train_x, dataset.train_y)
    return rf_reg
