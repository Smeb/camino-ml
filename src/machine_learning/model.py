"""model.py
    Contains the definition for the Model class
"""
class Model(object):
    """The model class couples an algorithm to a training dataset, and
    trains the model on that dataset"""
    # pylint: disable=too-few-public-methods
    def __init__(self, algorithm_name, algorithm, train_dataset, kwargs):
        self.name = "{}".format(algorithm_name)
        self.model = algorithm(train_dataset, **kwargs)
        self.train_dataset = train_dataset

    def predict(self, test_x):
        """Produces a prediction based on an input feature vector"""
        return self.model.predict(test_x)
