import os
import pandas

from src.dataset import Dataset
from src.machine_learning.experiment import Experiment

from src.routes import media_path
def aggregate_data():
  sub_directories = os.listdir(media_path)
  data_frame = pandas.DataFrame(columns=['model_name', 'algorithm', 'r2_score',
    'mean_squared_error', 'mean_absolute_error'])
  for sub_directory in sub_directories:
    model_name = Dataset.str_get_name(Experiment.dir_test_name(sub_directory))
    df = pandas.DataFrame.from_csv('{}/{}/results.csv'.format(media_path, sub_directory))
    df['model_name'] = model_name
    data_frame = data_frame.append(df[data_frame.columns])
  return data_frame

def select_metric(data, metric):
  algorithms = set(data['algorithm'])
  models = set(data['model_name'])
  metrics = pandas.DataFrame(columns=algorithms, index=models, dtype='float')
  for algorithm in algorithms:
    for model in models:
      value = float(data.ix[(data['algorithm'] == algorithm) & (data['model_name'] ==
        model)][metric])
      metrics.loc[model, algorithm] = value
  return metrics
