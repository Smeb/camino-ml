from __future__ import print_function
from src.fitting.fit_models import MODELS

def divide_fits(n_fits, machine_number):
  with open('./fits.py', 'w') as f:
    fit_list = []
    for index, fit in enumerate(MODELS):
      if index  % n_fits == machine_number - 1:
        fit_list.append(fit)
    print(fit_list, file=f)
