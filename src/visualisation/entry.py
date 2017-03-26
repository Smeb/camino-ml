from .selectors.test_train_equal import aggregate_data, select_metric
from .graphs.heatmap import heatmap
from src.routes import make_path_ignoring_existing, media_path

def visualisation_entry():
  make_path_ignoring_existing('{}/aggregate'.format(media_path))
  aggregate_scores = aggregate_data()
  r2_score_grid = select_metric(aggregate_scores, 'r2_score', 0)
  heatmap(r2_score_grid, 'r2_score', 0)
