"""test_train_equal.py
    Aggregates data from dataset where the test and train datasets are
    the same
"""
import os
import pandas

from src.datasets.dataset_factory import get_value_from_spec
from src.machine_learning.experiment import Experiment
from src.routes import RESULTS_PATH

def aggregate_data():
    """Aggregates data from all results.csv files in the results
    directory which match the selector (where test and train are the
    same)"""
    # pylint: disable=redefined-variable-type
    # pandas behaviour - variation of the same object
    data_frame = pandas.DataFrame(columns=['model_name',
                                           'algorithm',
                                           'noise',
                                           'r2_score',
                                           'scheme',
                                           'mean_squared_error',
                                           'mean_absolute_error'])

    for scheme_directory in os.listdir(RESULTS_PATH):
        scheme_folder_path = '{}/{}'.format(RESULTS_PATH, scheme_directory)
        for model_directory in os.listdir(scheme_folder_path):
            model_folder_path = '{}/{}'.format(scheme_folder_path, model_directory)
            dataset_name = Experiment.dir_test_name(model_directory)
            spec_path = '{}/{}.spec'.format(model_folder_path, dataset_name)

            subset_frame = pandas.DataFrame.from_csv('{}/results.csv'.format(model_folder_path))
            subset_frame['model_name'] = get_value_from_spec(spec_path, 'Name')
            subset_frame['noise'] = float(get_value_from_spec(spec_path, 'Noise'))
            subset_frame['scheme'] = get_value_from_spec(spec_path, 'Scheme')

            data_frame = data_frame.append(subset_frame[data_frame.columns])
    return data_frame

def select_metric(data, metric, noise, scheme):
    """Builds a dataframe where the index for each entry is the model,
    the column is the algorithm, and each row has only one value - the
    target metric"""

    algorithms = set(data['algorithm'])
    models = set(data['model_name'])
    metrics = pandas.DataFrame(columns=algorithms, index=models, dtype='float')
    for algorithm in algorithms:
        for model in models:
            metrics.loc[model, algorithm] = select_row_value(
                data, metric, noise, algorithm, model, scheme)
    return metrics

def select_row_value(data, metric, noise, algorithm, model, scheme):
    """Selects a single value from the dataframe based on the algorithm
    used, the model which generated the data, and the noise used when
    generating the model"""
    return float(data.ix[(data['algorithm'] == algorithm) &
                         (data['model_name'] == model) &
                         (data['noise'] == noise) &
                         (data['scheme'] == scheme)
                        ][metric])
