from joblib import Parallel, delayed
from tqdm import tqdm

from .model_factory import gen_model
from src.config import models
from src.routes import data_path, make_path_ignoring_existing

def generate():
  make_path_ignoring_existing(data_path)
  Parallel(n_jobs=-1)(delayed(gen_model)(model, i) for model, i in zip(models, range(len(models))))
