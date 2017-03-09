import os
import errno

import pandas
import matplotlib.pyplot as plt
from altair import Axis, Chart, Text, X, Y, Color
from sklearn import metrics
from tqdm import tqdm

from src.routes import media_path

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

def visualise_param_v_param(test_Ys, predict_Ys, feature_names, method_name, dataset_path):
  visualisation_path = "{}/predicted_vs_actual".format(dataset_path)
  try:
    os.makedirs(visualisation_path)
  except OSError as exc:
    if exc.errno == errno.EEXIST and os.path.isdir(visualisation_path):
      pass
    else:
      raise

  chart_x = pandas.DataFrame(test_Ys, columns=feature_names)
  chart_y = pandas.DataFrame(predict_Ys, columns=feature_names)

  print('Generating {} comparison graphs X_actual vs X_predicted'.format(method_name))
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

    plt.savefig('{}/{}-{}.png'.format(visualisation_path, method_name, feature))
    plt.clf()

def visualise_evaluations(evaluations):
  return

def visualise_difference(test_Ys, predict_Ys, feature_names, model_name):
  return

def heat_map(data, row, column, color):
  return
