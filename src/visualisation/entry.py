"""entry.py
    Entry point for visualisation functions; can be used to select which
    aggregate visualisations to generate
"""

from src.config import GRADIENT_STRENGTH
from src.routes import make_path_ignoring_existing, MEDIA_PATH

from .selectors.test_train_equal import aggregate_data, select_metric
from .graphs.heatmap import heatmap

def visualisation_entry():
    """Entry point function for visualisations"""
    make_path_ignoring_existing('{}/aggregate'.format(MEDIA_PATH))
    aggregate_scores = aggregate_data()
    scheme = 'PGSE_90_{}t'.format(GRADIENT_STRENGTH)
    visualise_metric(0, scheme, 'r2_score', aggregate_scores)
    visualise_metric_difference(0, 20, scheme, 'r2_score', aggregate_scores)

def visualise_metric_difference(dataset_a_noise, dataset_b_noise, scheme, metric, aggregate_scores):
    dataset_a = select_metric(aggregate_scores, metric, dataset_a_noise, scheme)
    dataset_b = select_metric(aggregate_scores, metric, dataset_b_noise, scheme)
    percentage_difference = (dataset_b - dataset_a) / dataset_a * 100
    title = 'percentage_difference_from_snr=0_when_snr=20'
    heatmap(percentage_difference, '{} percentage difference'.format(metric), title, )

def visualise_metric(noise, scheme, metric, aggregate_scores):
    """Creates an score heatmap for the given dataset (defined by noise and scheme), and metric"""
    score_grid = select_metric(aggregate_scores, metric, noise, scheme)
    title = 'r2_score_when_noise=0'
    heatmap(score_grid, metric, title)
