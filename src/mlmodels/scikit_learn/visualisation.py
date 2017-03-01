from altair import Axis, Chart, X, load_dataset
import pandas
from sklearn import metrics

from src.routes import media_path

def visualise(data, model, model_name):
  _, (_, _), (testX, testY), feature_names = data
  prediction = model.predict(testX)

  difference_matrix = pandas.DataFrame(testY - prediction)
  difference_matrix.columns = feature_names

  chartData = pandas.DataFrame(columns=['name', 'difference'])

  for index, row in difference_matrix.iterrows():
    for index, value in enumerate(row):
      chartData = chartData.append({'name': feature_names[index], 'difference': value}, ignore_index=True)

  chart = Chart(chartData,
    width=1000,
  ).mark_point().encode(
    X('difference', axis=Axis(format='g')),
    y='name',
  ).to_html()

  with open('{}/{}_difference.html'.format(media_path, model_name), 'w') as f:
    f.write(chart)
