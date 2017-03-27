"""knn.py
    Contains code interfacing with scikit-learn which returns a
    K-Nearest Neighbours model fit to the input training data
"""
from sklearn import neighbors
def train_knn(dataset):
    """Returns a fitted K-Nearest Neighbour model"""
    knn = neighbors.KNeighborsRegressor(weights="distance")
    knn.fit(dataset.train_x, dataset.train_y)
    return knn
