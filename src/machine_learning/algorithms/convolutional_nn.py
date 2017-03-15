import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation

def search_convolutional_nn(dataset, neurons=[13, 13], n_hidden_layers=1):
  model = Sequential()
  model.add(Dense(neurons[0], input_dim=len(dataset.train_X[0]), activation='relu'))
  for i in range(1, n_hidden_layers + 1):
    model.add(Dense(neurons[i], activation='relu'))
  model.add(Dense(output_dim=len(dataset.train_Y[0])))

  model.compile(loss="mse", optimizer="rmsprop")

  model.fit(dataset.train_X, dataset.train_Y, verbose=1, nb_epoch=150)
  return model
