"""bland_altman.py
    Implements bland_altman plots of datasets
"""
import pandas
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.routes import make_path_ignoring_existing

def bland_altman(test_y, predict_y, feature_names, algorithm_name, experiment_media_path):
    """Generates a bland altman plot, showing the difference between two
    datasets, and colouring each point based on its error from the
    expected value
    """
    # pylint: disable=too-many-locals
    visualisation_path = "{}/{}/bland_altman".format(experiment_media_path, algorithm_name)
    make_path_ignoring_existing(visualisation_path)

    chart_x = pandas.DataFrame(test_y, columns=feature_names)
    chart_y = pandas.DataFrame(predict_y, columns=feature_names)

    print('Generating {} comparison Bland-Altman plots'.format(algorithm_name))
    for feature in tqdm(feature_names):
        x_features = np.array(chart_x[feature].tolist())
        y_features = np.array(chart_y[feature].tolist())

        mean = np.mean([x_features, y_features], axis=0)
        diff = y_features - x_features
        mean_diff = np.mean(diff)
        std = np.std(diff, axis=0)

        with np.errstate(divide='ignore', invalid='ignore'):
            percentage_error = (diff/x_features)*100

        with np.errstate(invalid='ignore'):
            percentage_error[percentage_error > 50] = 50
            percentage_error[percentage_error < -50] = -50

        plt.scatter(mean, diff, c=percentage_error, edgecolors='face', cmap='jet')
        plt.axhline(mean_diff, color='gray', linestyle='-')
        plt.axhline(mean_diff + 1.96 * std, color='gray', linestyle='--')
        plt.axhline(mean_diff - 1.96 * std, color='gray', linestyle='--')
        plt.ylabel('Difference')
        plt.xlabel('Mean value')
        plt.title(feature)
        x_range = np.max(mean) - np.min(mean)
        y_range = np.max(diff) - np.min(diff)
        axis_extra_range = 0.1
        plt.xlim([np.min(mean) - (x_range * axis_extra_range),
                  np.max(mean) + (x_range * axis_extra_range)])
        plt.ylim([np.min(diff) - (y_range * axis_extra_range),
                  np.max(diff) + (y_range * axis_extra_range)])

        plt.colorbar()

        plt.savefig('{}/{}-{}-Bland-Altman.png'.format(visualisation_path, algorithm_name, feature))
        plt.clf()
