"""multilayer_perceptron.py
    Contains code interfacing with scikit-learn which returns a
    MultilayerPerceptron model fit to the input training data
"""
from sklearn.neural_network import MLPRegressor

def train_mlp(dataset):
    """Returns a fitted MultilayerPerceptron model"""
    mlp = MLPRegressor()
    mlp.fit(dataset.train_x, dataset.train_y)
    return mlp
