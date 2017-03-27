"""parameter_scatterplot.py
    Implements scatterplots which compare X and Y values, with a line
    showing where x=y
"""
import pandas
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.routes import make_path_ignoring_existing

def calc_axis_limits(x_series, y_series):
    """Calculates the axes limits so that they can be scaled; axes
    minimum and maximums will be the corresponding minimum and maximum
    across both series, scaled by 0.1 * the largest absolute value"""
    min_xy = min(x_series + y_series)
    max_xy = max(x_series + y_series)
    if abs(max_xy) > abs(min_xy):
        coord_diff = 1 if max_xy == 0 else abs(max_xy) * 0.1
    else:
        coord_diff = 1 if min_xy == 0 else abs(min_xy) * 0.1
    return (min_xy - coord_diff, max_xy + coord_diff)

def parameter_scatterplot(test_y, predict_y, feature_names, algorithm_name, experiment_media_path):
    """Produces a scatterploy of actual values (as the x axis), versus
    the predicted values (as the y axis), with a line showing where x = y"""
    visualisation_path = "{}/{}/predicted_vs_actual".format(experiment_media_path, algorithm_name)
    make_path_ignoring_existing(visualisation_path)

    chart_x = pandas.DataFrame(test_y, columns=feature_names)
    chart_y = pandas.DataFrame(predict_y, columns=feature_names)

    print('Generating {} comparison graphs X_actual vs X_predicted'.format(algorithm_name))
    for feature in tqdm(feature_names):
        x_features = chart_x[feature].tolist()
        y_features = chart_y[feature].tolist()

        plt.xlabel("ground truth")
        plt.ylabel("prediction")
        plt.title(feature)
        axis_limits = calc_axis_limits(x_features, y_features)
        plt.ylim(axis_limits)
        plt.xlim(axis_limits)

        plt.scatter(x_features, y_features)
        plt.plot(axis_limits, axis_limits, label="actual = predicted", color='r')

        plt.savefig('{}/{}-{}.png'.format(visualisation_path, algorithm_name, feature))
        plt.clf()
