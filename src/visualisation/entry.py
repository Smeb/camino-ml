from .selectors.test_train_equal import aggregate_data, select_metric
from .graphs.heatmap import heatmap

def visualisation_entry():
  aggregate_scores = aggregate_data()
  r2_score_grid = select_metric(aggregate_scores, 'r2_score')
  heatmap(r2_score_grid, 'r2_score')
