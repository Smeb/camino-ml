"""entry.py
    Entry point for visualisation functions; can be used to select which
    aggregate visualisations to generate
"""

from src.routes import make_path_ignoring_existing, MEDIA_PATH

from .selectors.test_train_equal import aggregate_data, select_metric
from .graphs.heatmap import heatmap

def visualisation_entry():
    """Entry point function for visualisations"""
    make_path_ignoring_existing('{}/aggregate'.format(MEDIA_PATH))
    aggregate_scores = aggregate_data()
    r2_score_grid = select_metric(aggregate_scores, 'r2_score', 0, 'PGSE_90_60t')
    heatmap(r2_score_grid, 'r2_score', 0)
