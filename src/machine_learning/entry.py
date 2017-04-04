"""entry.py
    Entry point for the machine learning methods; can be altered to
    change which algorithms are applied to datasets
"""
from tqdm import tqdm

from src.config import MODELS
from src.datasets.dataset_factory import Dataset
from src.routes import make_path_ignoring_existing, MEDIA_PATH

from .experiment import Experiment
from .model import Model
from .algorithms.random_forest import train_rf
from .algorithms.knn import train_knn
# from .algorithms.svr import train_linear_svr
# from .algorithms.multilayer_perceptron import train_mlp
from .algorithms.ridge_regression import train_ridge
# from .algorithms.convolutional_nn import initialise_nn_grid_search, search_convolutional_nn

ALGORITHMS = [
    ('RandomForest', train_rf, {}),
    # ('MultiLayerPerceptron', train_mlp, {}),
    # ('linearSVR', train_linear_svr, {}),
    ('KNN', train_knn, {}),
    ('RidgeRegression', train_ridge, {}),
]# + initialise_nn_grid_search(range(5, 26, 2), range(2, 5))

def all_datasets():
    """Provides on demand access to dataset instances; since each
    dataset consists of potentially thousands of voxels, this is
    preferred to loading all datasets at once."""
    for model in MODELS:
        print("Loading: {}".format(model))
        yield Dataset.from_model(model)

def gen_experiments(dataset):
    """Generates the list of all experiments, which consist of a Model
    with a training dataset and algorithm, and a test dataset"""
    experiments = []
    print('Running experiments for {}'.format(dataset.name))
    for function_name, function, kwargs in tqdm(ALGORITHMS):
        experiments.append(Experiment(Model(function_name, function, dataset, kwargs), dataset))
    return experiments

def train_and_evaluate_all_datasets():
    """Parses all datasets that match the specification in
    src/config.py, trains the defined models in ALGORITHMS on all of the
    datasets, and outputs the results to the RESULTS_PATH"""
    make_path_ignoring_existing(MEDIA_PATH)

    for dataset in all_datasets():
        experiments = gen_experiments(dataset)
        for experiment in experiments:
            experiment.evaluate()
            experiment.visualise()
