"""heatmap.py
    Contains implementations of heatmaps (graphs which show points as
    shaded squares)
"""
import numpy
import matplotlib.pyplot as plt

from src.routes import MEDIA_PATH

def heatmap(data, metric, noise):
    """Produces a blue shaded HEATMAP; based on a visualisation by Nathan Yau
    of flowingdata.com"""
    name = '{}_heatmap_{}'.format(metric, noise)

    fig, axes = plt.subplots()
    heatmap_plt = axes.pcolor(data, cmap=plt.cm.Blues, alpha=0.8, edgecolors='face')

    fig = plt.gcf()
    fig.set_size_inches(8, 11)

    axes.set_frame_on(False)

    axes.set_yticks(numpy.arange(data.shape[0]) + 0.5, minor=False)
    axes.set_xticks(numpy.arange(data.shape[1]) + 0.5, minor=False)

    axes.invert_yaxis()
    axes.xaxis.tick_top()

    axes.set_xticklabels(data.columns, minor=False)
    axes.set_yticklabels(data.index, minor=False)

    plt.xticks(rotation=90)
    axes.grid(False)
    axes = plt.gca()

    for tick in axes.xaxis.get_major_ticks():
        tick.tick1On = False
        tick.tick2On = False
    for tick in axes.yaxis.get_major_ticks():
        tick.tick1On = False
        tick.tick2On = False

    print(data)
    plt.colorbar(heatmap_plt)

    fig.savefig('{}/aggregate/{}.png'.format(MEDIA_PATH, name))
