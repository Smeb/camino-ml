"""gen_voxels.py
    Starts a dataset generation task based on definitions in src/config.py
"""
from joblib import Parallel, delayed

from src.config import MODELS
from src.routes import DATA_PATH, make_path_ignoring_existing

from .dataset_factory import gen_dataset

def generate():
    """Generates datasets at DATA_PATH based on definitions in
    src/config.py. Uses joblib to do generation in parallel"""
    make_path_ignoring_existing(DATA_PATH)
    bar_positions = range(len(MODELS))
    Parallel(n_jobs=-1)(delayed(gen_dataset)(model, i) for model, i in zip(MODELS, bar_positions))
