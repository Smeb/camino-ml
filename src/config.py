camino_compartments = {
  "stick": ["d", "theta", "phi"],
  "cylindergpd": ["d", "theta", "phi", "R"],
  "gammadistribradiicylinders": ["k", "b", "d", "theta", "phi"],

  "ball": ["d"],
  "zeppelin": ["d", "theta", "phi", "d_perp1"],
  "tensor": ["d", "theta", "phi", "d_perp1", "d_perp2", "alpha"],

  "astrosticks": ["d"],
  "astrocylinders": ["d", "R"],
  "spheregpd": ["d", "Rs"],
  "dot": [],
}

definitions = {
  "d": [0.8E-9, 2.2E-9],
  "theta": 0.0,
  "phi": 0.0,
  "alpha": 0.0,
  "R": [1.5E-6, 2.5E-6],
  "Rs": [1.5E-6, 2.5E-6],
  "k": [1.7, 2.7],
  "b": [0.8E-6, 1.4E-6],
}

models = [
  ['tensor'],
  ['ball', 'stick'],
  ['ball', 'cylindergpd', 'dot'],
  ['ball', 'gammadistribradiicylinders', 'dot'],
  ['tensor', 'cylindergpd', 'spheregpd'],
  ['zeppelin', 'gammadistribradiicylinders', 'spheregpd'],
  ['ball', 'stick', 'spheregpd'],
]

dataset_size = 10
