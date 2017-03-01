import random

class Model:
  def __init__(self, name, params):
    self._name = name
    self._params = []
    self._paramNames = params.keys()
    for item in params:
      self._params.append(params[item])

  def __str__(self):
    string = self._name
    for param in self._params:
      if isinstance(param, list):
        sample = random.uniform(param[0], param[1])
      else:
        sample = param
      string = ' '.join([string, str(sample)])
    return string
