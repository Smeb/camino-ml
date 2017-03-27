"""experiment.py
    Contains the definition for an experiment, which couples a Model to
    a test dataset
"""
import os
import shutil

import pandas
from sklearn import metrics

from src.config import UUID
from src.routes import make_path_ignoring_existing, MEDIA_PATH, RESULTS_PATH
from src.visualisation.graphs.parameter_scatterplot import parameter_scatterplot
from src.visualisation.graphs.bland_altman import bland_altman

class Experiment(object):
    """Couples a Model to a test_dataset, and implements methods to
    provide evaluations of model performance on that test dataset, as
    well as visualise the performance"""

    def __init__(self, model, test_dataset):
        self.model = model
        self.test_dataset = test_dataset

        # Predictions are scaled by the train dataset scaler
        self.unscaled_predict_y = self.model.train_dataset.inverse_transform(
            self.model.predict(test_dataset.test_x))

        self.unscaled_test_y = self.model.train_dataset.inverse_transform(test_dataset.test_y)

        if self.model.train_dataset.scheme != self.test_dataset.scheme:
            raise Exception('Scheme files of datasets should match')

        # The following elements should uniquely identify any two datasets
        identifying_path = '{}/{}-{}'.format(self.model.train_dataset.scheme,
                                             self.model.train_dataset.unique_string,
                                             self.test_dataset.unique_string)

        self.media_path = '{}/{}'.format(MEDIA_PATH, identifying_path)
        self.results_path = '{}/{}'.format(RESULTS_PATH, identifying_path)
        self.make_media_path()
        self.make_results_path()
        self.results_file_path = '{}/results.csv'.format(self.results_path)
        if os.path.exists(self.results_file_path):
            raise Exception('For {} on model {}, previous training data exists'.format(
                model.name, self.test_dataset.unique_string))

    def make_media_path(self):
        """Creates the folders for the media path, where visualisations
        are output"""
        make_path_ignoring_existing(self.media_path)

    def make_results_path(self):
        """Creates the folders for the results path, where visualisations
        are output, and also copies the spec files for both datasets to
        the directory"""
        make_path_ignoring_existing(self.results_path)
        shutil.copy(self.model.train_dataset.spec_path,
                    '{}/{}.spec'.format(self.results_path,
                                        self.model.train_dataset.unique_string))

        shutil.copy(self.test_dataset.spec_path,
                    '{}/{}.spec'.format(self.results_path, self.test_dataset.unique_string))

    def get_results_dataframe(self):
        """Gets the results dataframe if one exists, otherwise returns a
        new dataframe to store results in"""
        if os.path.isfile(self.results_file_path):
            return pandas.DataFrame.from_csv(self.results_file_path)
        else:
            return pandas.DataFrame(
                columns=['uuid',
                         'algorithm',
                         'mean_absolute_error',
                         'mean_squared_error',
                         'r2_score'] + self.test_dataset.feature_names)

    def evaluate(self):
        """Produces evaluations of the model's predictions on the
        ground_truth of the test dataset, and stores them in the results
        directory as a csv"""
        results_dataframe = self.get_results_dataframe()

        row = [UUID, self.model.name]
        row.append(metrics.mean_absolute_error(self.unscaled_test_y, self.unscaled_predict_y))
        row.append(metrics.mean_squared_error(self.unscaled_test_y, self.unscaled_predict_y))
        row.append(metrics.r2_score(self.unscaled_test_y, self.unscaled_predict_y))

        for param in self.test_dataset.feature_names:
            row.append(metrics.r2_score(self.unscaled_test_y[param],
                                        self.unscaled_predict_y[param]))

        # pylint: disable=no-member
        # Pylint doesn't recognize the function returns a dataframe
        results_dataframe.loc[len(results_dataframe)] = row
        results_dataframe.to_csv(self.results_file_path)

    def visualise(self):
        """Produces visualisations of the test ground truth for the test
        dataset against the values predicted by the model"""
        parameter_scatterplot(self.unscaled_test_y,
                              self.unscaled_predict_y,
                              self.test_dataset.feature_names,
                              self.model.name,
                              self.media_path)

        bland_altman(self.unscaled_test_y,
                     self.unscaled_predict_y,
                     self.test_dataset.feature_names,
                     self.model.name,
                     self.media_path)


    @staticmethod
    def dir_train_name(dirname):
        """Returns the training dataset name from the directory name of an experiment"""
        return dirname.split('-')[0]

    @staticmethod
    def dir_test_name(dirname):
        """Returns the test dataset name from the directory name of an experiment"""
        return dirname.split('-')[1]
