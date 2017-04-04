"""random_forest.py
    Contains code interfacing with scikit-learn which returns a
    RandomForest model fit to the input training data
"""
import multiprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

def train_rf(dataset):
    """Returns a fitted RandomForest model"""
    param_grid = {
        "max_depth":[3, 5, None], # max depth of tree, None = deep as required
        "bootstrap": [True, False], # if bootstrap samples are used for trees
        "criterion": ["mse"], # mean square error, mean absolute error
        "n_estimators": [5, 10, 20, 40, 80] # number of trees to train
    }

    rf_reg = RandomForestRegressor()
    grid_search = GridSearchCV(rf_reg, param_grid, n_jobs=multiprocessing.cpu_count(), verbose=10)
    grid_search.fit(dataset.train_x, dataset.train_y)
    return grid_search
