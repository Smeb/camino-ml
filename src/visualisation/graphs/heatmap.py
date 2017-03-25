import re
import numpy
import matplotlib.pyplot as plt

from src.routes import media_path

def heatmap(data, metric):
  name = 'aggregate_{}_heatmap'.format(metric)
  print(data)

  fig, ax = plt.subplots()
  heatmap = ax.pcolor(data, cmap=plt.cm.Blues, alpha=0.8)

  fig = plt.gcf()
  fig.set_size_inches(8, 11)

  ax.set_frame_on(False)

  ax.set_yticks(numpy.arange(data.shape[0]) + 0.5, minor=False)
  ax.set_xticks(numpy.arange(data.shape[1]) + 0.5, minor=False)

  ax.invert_yaxis()
  ax.xaxis.tick_top()

  # Trim id values from models
  regex = re.compile(r"-[0-9]")
  labels = [re.sub(regex, "", label) for label in data.columns]

  ax.set_xticklabels(data.columns, minor=False)
  ax.set_yticklabels(data.index, minor=False)

  plt.xticks(rotation=90)
  ax.grid(False)
  ax = plt.gca()

  for t in ax.xaxis.get_major_ticks():
    t.tick1On = False
    t.tick2On = False

  for t in ax.yaxis.get_major_ticks():
    t.tick1On = False
    t.tick2On = False

  fig.savefig('{}/heatmap.png'.format(media_path))
