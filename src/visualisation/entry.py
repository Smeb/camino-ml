"""entry.py
    Entry point for visualisation functions; can be used to select which
    aggregate visualisations to generate
"""
import pandas

from sklearn.metrics import r2_score

from src.config import GRADIENT_STRENGTH, MODELS
from src.routes import make_path_ignoring_existing, MEDIA_PATH

from src.datasets.dataset_factory import get_model_name

from .selectors.test_train_equal import aggregate_data, load_all_parameters, select_metric, select_dataset_results
from .graphs.heatmap import heatmap
from .graphs.box_plots import box_plots

def visualisation_entry():
    """Entry point function for visualisations"""
    make_path_ignoring_existing('{}/aggregate'.format(MEDIA_PATH))
    aggregate_scores = aggregate_data()
    parameter_results = load_all_parameters()
    scheme = 'PGSE_90_{}t'.format(GRADIENT_STRENGTH)
    visualise_metric(0, scheme, 'r2_score', aggregate_scores)

    # compare_camino_with_rf(aggregate_scores, parameter_results)
    ground_truth = select_dataset_results(['Ball', 'Cylinder'], 'ground_truth', scheme)
    random_forest_predictions = select_dataset_results(['Ball', 'Cylinder'], 'RandomForest', scheme)
    box_plots(random_forest_predictions, ground_truth)
    exit()



    # Generate a comparison for noisy and non-noisy datasets
    # dataset_noise_0 = select_metric(aggregate_scores, 'r2_score', 0, scheme)
    # dataset_noise_20 = select_metric(aggregate_scores, 'r2_score', 20, scheme)
    # visualise_metric_difference(dataset_noise_0, dataset_noise_20, 'r2_score')

def compare_camino_with_rf(aggregate_scores, parameter_results):
    ground_truth = select_by_field(parameter_results, 'model_name', 'ground_truth')
    ground_truth = select_by_field(ground_truth, 'noise', 0)

    camino_estimates = select_by_field(parameter_results, 'model_name', 'camino_fit')
    camino_estimates = select_by_field(camino_estimates, 'noise', 0)

    camino_aggregate = pandas.DataFrame(columns=['model_name', 'r2_score'])


    print(select_by_field(camino_estimates, 'dataset', 'BallStick'))
    print(select_by_field(ground_truth, 'dataset', 'BallStick'))
    exit()
    for model in MODELS:
        model_name = get_model_name(model)
        try:
          model_ground_truth = as_numpy(select_by_field(ground_truth, 'dataset', model_name))
          model_camino_estimates = as_numpy(select_by_field(camino_estimates, 'dataset', model_name))
        except:
          continue
        print(model_name)
        print(model_ground_truth)
        print(model_camino_estimates)
        r2 = r2_score(model_ground_truth, model_camino_estimates)
        camino_aggregate.loc[-1] = [model_name, r2]
    print(camino_aggregate)


def as_numpy(df):
    return df.select_dtypes(include=['float64']).drop('noise', axis=1).as_matrix()


def visualise_metric_difference(dataset_a, dataset_b, metric):
    percentage_difference = (dataset_b - dataset_a) / dataset_a * 100
    title = 'percentage_difference_from_snr=0_when_snr=20'
    heatmap(percentage_difference, '{} percentage difference'.format(metric), title)

def visualise_metric(noise, scheme, metric, aggregate_scores):
    """Creates an score heatmap for the given dataset (defined by noise and scheme), and metric"""
    score_grid = select_metric(aggregate_scores, metric, noise, scheme)
    title = 'r2_score_when_noise=0'
    heatmap(score_grid, metric, title)

def select_by_field(df, field, value):
    return df.loc[df[field] == value].dropna(axis='columns', how='all')
