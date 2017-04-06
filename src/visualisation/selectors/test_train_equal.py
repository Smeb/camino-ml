"""test_train_equal.py
    Aggregates data from dataset where the test and train datasets are
    the same
"""
import os
import pandas

from src.config import MODELS, STRIP_LIST
from src.datasets.dataset_factory import CAMINO_COMPARTMENTS, get_model_name, get_value_from_spec
from src.machine_learning.experiment import Experiment
from src.routes import RESULTS_PATH

def load_all_parameters():
    columns = ['dataset', 'model_name', 'noise', 'scheme']
    filter_columns = columns[:]
    for model in CAMINO_COMPARTMENTS:
        for parameter in CAMINO_COMPARTMENTS[model]:
            columns.append('{}_{}'.format(model, parameter))
            if parameter not in STRIP_LIST:
                filter_columns.append('{}_{}'.format(model, parameter))


    data_frame = pandas.DataFrame(columns=columns)

    for scheme_directory in os.listdir(RESULTS_PATH):
        scheme_folder_path = '{}/{}'.format(RESULTS_PATH, scheme_directory)
        for model_directory in os.listdir(scheme_folder_path):
            model_folder_path = '{}/{}'.format(scheme_folder_path, model_directory)
            dataset_name = Experiment.dir_test_name(model_directory)
            spec_path = '{}/{}.spec'.format(model_folder_path, dataset_name)
            for parameter_csv_file in [name for name in os.listdir(model_folder_path) if not name.startswith('results') and name.endswith('.csv')]:
              csv_path = '{}/{}'.format(model_folder_path, parameter_csv_file)
              subset_frame = pandas.DataFrame.from_csv(csv_path)
              subset_frame['model_name'] = parameter_csv_file.split('.')[0]
              subset_frame['noise'] = float(get_value_from_spec(spec_path, 'Noise'))
              subset_frame['scheme'] = get_value_from_spec(spec_path, 'Scheme')
              subset_frame['dataset'] = dataset_name.split('_')[0]
              data_frame = pandas.concat([data_frame, subset_frame], axis=0, ignore_index=True)

    return data_frame[filter_columns]

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

    models = pandas.Categorical(PANAGIOTAKI_RANKING)
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
    row_value = data.ix[(data['algorithm'] == algorithm) &
                        (data['model_name'] == model) &
                        (data['noise'] == noise) &
                        (data['scheme'] == scheme)
                       ][metric]
    return float(row_value)

def select_dataset_results(model, algorithm, scheme):
    model_name = get_model_name(model)
    model_noise = 0
    model_n_voxels = 100
    model_unique_name = '{}_{}_{}'.format(model_name, model_n_voxels, model_noise)
    model_path = '{}/{}/{}-{}'.format(RESULTS_PATH, scheme, model_unique_name,
                                      model_unique_name)
    df = pandas.DataFrame.from_csv('{}/{}.csv'.format(model_path, algorithm))
    return df


PANAGIOTAKI_RANKING = [
    'TensorCylinderSphere',
    'TensorStickSphere',
    'ZeppelinCylinderSphere',
    'ZeppelinGDRCylindersSphere',
    'ZeppelinStickSphere',
    'TensorCylinderDot',
    'TensorGDRCylindersDot',
    'TensorGDRCylindersSphere',
    'ZeppelinCylinderDot',
    'ZeppelinGDRCylindersDot',
    'BallGDRCylindersDot',
    'BallCylinderSphere',
    'BallCylinderDot',
    'BallGDRCylindersSphere',
    'ZeppelinStickDot',
    'TensorCylinderAstrocylinders',
    'TensorCylinderAstrosticks',
    'TensorGDRCylindersAstrosticks',
    'TensorStickAstrocylinders',
    'TensorStickAstrosticks',
    'ZeppelinGDRCylindersAstrocylinders',
    'BallStickSphere',
    'BallCylinderAstrocylinders',
    'BallCylinderAstrosticks',
    'ZeppelinCylinderAstrocylinders',
    'BallStickAstrocylinders',
    'ZeppelinCylinderAstrosticks',
    'TensorStickDot',
    'BallStickAstrosticks',
    'BallGDRCylindersAstrosticks',
    'ZeppelinStickAstrocylinders',
    'ZeppelinStickAstrosticks',
    'BallGDRCylindersAstrocylinders',
    'BallStickDot',
    'TensorGDRCylindersAstrocylinders',
    'ZeppelinGDRCylindersAstrosticks',
    'TensorCylinder',
    'TensorGDRCylinders',
    'BallCylinder',
    'ZeppelinCylinder',
    'BallGDRCylinders',
    'TensorStick',
    'ZeppelinGDRCylinders',
    'ZeppelinZeppelin',
    'BallStick',
    'ZeppelinStick',
    'Tensor',
]

PANAGIOTAKI_RANKING = [item for item in PANAGIOTAKI_RANKING if item in list(map(get_model_name, MODELS))]
