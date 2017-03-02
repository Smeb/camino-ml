from altair import Axis, Chart, Text, X, Color
import pandas
from sklearn import metrics

from src.routes import media_path

def visualise_evaluations(evaluations):
  experiment_data = pandas.DataFrame(columns=['method', 'mean_absolute_error', 'mean_squared_error', 'r2_score'])
  for method, evaluation_set in evaluations:
    mean_absolute_error, mean_squared_error, r2_score = evaluation_set
    experiment_data = experiment_data.append({
      'method': method.name,
      'model': method.data_model,
      'mean_absolute_error': mean_absolute_error,
      'mean_squared_error': mean_squared_error,
      'r2_score': r2_score
    }, ignore_index=True)
  chartTypes = ['mean_absolute_error', 'mean_squared_error', 'r2_score']
  charts = []
  for type in chartTypes:
    chart = heat_map(experiment_data, row='method', column='model', color=type).to_html()
    with open('{}/{}_heatmap.html'.format(media_path, type), 'w') as f:
      f.write(chart)

def visualise_difference(data, model, model_name, dataset_name):
  _, (_, _), (testX, testY), feature_names = data
  prediction = model.predict(testX)

  difference_matrix = pandas.DataFrame(testY - prediction)
  difference_matrix.columns = feature_names

  chart_data = pandas.DataFrame(columns=['name', 'difference'])

  for _, row in difference_matrix.iterrows():
    for index, value in enumerate(row):
      chart_data = chart_data.append({'name': feature_names[index], 'difference': value}, ignore_index=True)

  chart = Chart(chart_data,
    width=1000,
  ).mark_point().encode(
    X('difference', axis=Axis(format='g')),
    y='name',
  ).to_html()

  with open('{}/{}_{}_difference.html'.format(media_path, dataset_name, model_name), 'w') as f:
    f.write(chart)

def heat_map(data, row, column, color):
  # TODO: Width setting is very hacky, don't use magic number
  return Chart(data,
    width=200,
    ).mark_text(
    applyColorToBackground=True,
    ).encode(
      row=row,
      column=column,
      text=Text(value=' '),
      color=color
    )
