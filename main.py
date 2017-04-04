"""
Entry point for program
"""
import sys
import warnings

from matplotlib import rcParams

from src.datasets.gen_voxels import generate
from src.fitting.fit_models import fit_all_models
from src.machine_learning.entry import train_and_evaluate_all_datasets
from src.visualisation.entry import visualisation_entry
from src.scripts.divide_fits import divide_fits
from src.scripts.divide_dataset import divide_dataset

def warn(*args, **kwargs):
    #pylint: disable=unused-argument
    """Overrides deprecation warning from scikit-learn"""
    pass

warnings.warn = warn

def usage():
    """Prints usage information for the program"""
    print "python main.py <CMD> where <CMD> is one of"
    print "  generate            -- to generate data"
    print "  train-all           -- to train all models using generated data"
    print "  visualise           -- to generate aggregate visualisations"
    print "Models can be configured in src/config.py; compartment names"
    print "must match those in camino_compartments, which is defined in the same file"

if __name__ == "__main__":
    rcParams.update({'figure.autolayout': True})
    if len(sys.argv) < 2:
        usage()
    CMD = sys.argv[1]
    if CMD == "generate":
        generate()
    elif CMD == "fit-all":
        fit_all_models()
    elif CMD == "train-all":
        train_and_evaluate_all_datasets()
    elif CMD == "visualise":
        visualisation_entry()
    elif CMD == "divide-fits":
        divide_fits(2, 1)
    elif CMD == "divide-dataset":
        divide_dataset()
    else:
        usage()
