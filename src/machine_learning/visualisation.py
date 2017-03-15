import pandas
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from tqdm import tqdm

from src.routes import make_path_ignoring_existing, media_path

def best_fit(X, Y):
  xbar = sum(X) / len(X)
  ybar = sum(Y) / len(Y)
  n = len(X)

  numer = sum([xi*yi for xi, yi in zip(X, Y)]) - n * xbar * ybar
  denum = sum([xi**2 for xi in X]) - n * xbar**2

  b = numer / denum
  a = ybar - b * xbar

  return a, b

def calc_axis_limits(X, Y):
  min_xy = min(X + Y)
  max_xy = max(X + Y)
  coord_diff = 1 if max_xy == 0 else max_xy * 0.1
  return (min_xy - coord_diff, max_xy + coord_diff)

def visualise_param_v_param(test_Y, predict_Y, feature_names, algorithm_name, dataset_path):
  visualisation_path = "{}/{}/predicted_vs_actual".format(dataset_path, algorithm_name)
  make_path_ignoring_existing(visualisation_path)

  chart_x = pandas.DataFrame(test_Y, columns=feature_names)
  chart_y = pandas.DataFrame(predict_Y, columns=feature_names)

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

    # plot line of best fit
    try:
      if axis_limits[0] != -1 and axis_limits[1] != 1:
        a, b = best_fit(x_features, y_features)
        yfit = [a + b * xi for xi in x_features]
        plt.plot(x_features, yfit)
    except:
      pass

    plt.savefig('{}/{}-{}.png'.format(visualisation_path, algorithm_name, feature))
    plt.clf()

def visualise_bland_altman(test_Y, predict_Y, feature_names, algorithm_name, dataset_path):
  visualisation_path = "{}/{}/bland_altman".format(dataset_path, algorithm_name)
  make_path_ignoring_existing(visualisation_path)

  chart_x = pandas.DataFrame(test_Y, columns=feature_names)
  chart_y = pandas.DataFrame(predict_Y, columns=feature_names)

  print('Generating {} comparison Bland-Altman plots'.format(algorithm_name))
  for feature in tqdm(feature_names):
    x_features = np.array(chart_x[feature].tolist())
    y_features = np.array(chart_y[feature].tolist())

    mean      = np.mean([x_features, y_features], axis=0)
    diff      = y_features - x_features                  # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference


    with np.errstate(divide='ignore', invalid='ignore'):
      percentage_error = (diff/x_features)*100

    with np.errstate(invalid='ignore'):
      percentage_error[percentage_error > 50] = 50
      percentage_error[percentage_error < -50] = -50

    plt.scatter(mean, diff, c=percentage_error, edgecolors='face', cmap='jet')
    plt.axhline(md,           color='gray', linestyle='-')
    plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
    plt.ylabel('Difference')
    plt.xlabel('Mean value')
    plt.title(feature)
    x_range = np.max(mean) - np.min(mean)
    y_range = np.max(diff) - np.min(diff)
    axis_extra_range = 0.1
    plt.xlim([np.min(mean) - (x_range * axis_extra_range), np.max(mean) + (x_range * axis_extra_range)])
    plt.ylim([np.min(diff) - (y_range * axis_extra_range), np.max(diff) + (y_range * axis_extra_range)])
    plt.colorbar()

    plt.savefig('{}/{}-{}-Bland-Altman.png'.format(visualisation_path, algorithm_name, feature))
    plt.clf()


def visualise_evaluations(evaluations):
  return

def visualise_difference(test_Ys, predict_Ys, feature_names, model_name):
  return

def heat_map(data, row, column, color):
  return
