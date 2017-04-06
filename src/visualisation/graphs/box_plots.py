import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from src.routes import MEDIA_PATH

from time import sleep

def box_plots(predictions, ground_truth):
    title = 'Random_forests_box_plots'
    fig = plt.figure()


    percentages = ((predictions - ground_truth) / ground_truth) * 100
    columns = [label.replace('_', ' ') for label in percentages.columns.values]
    percentages.columns = columns
    axis = percentages.boxplot(patch_artist=True, return_type='dict')

    print(axis.keys())
    for box in axis['boxes']:
        box.set_facecolor('white')
    axis = axis['boxes'][0].axes

    fmt = '%.0f%%'
    yticks = axis.get_yticks()
    axis.set_yticklabels(['{:3.2f}%'.format(y) for y in yticks])
    axis.set_ylabel('Percentage difference')
    axis.set_xlabel('Parameter')
    fig.add_axes(axis)

    fig.savefig('{}/aggregate/{}.png'.format(MEDIA_PATH, title))

