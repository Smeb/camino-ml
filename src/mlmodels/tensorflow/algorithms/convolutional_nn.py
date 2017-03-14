import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation

def train_convolutional_nn(data):
  _, (trainX, trainY), _, _ = data

  model = Sequential()
  model.add(Dense(output_dim=13, input_dim=len(trainX[0])))
  model.add(Dense(13))
  model.add(Dense(output_dim=len(trainY[0])))

  model.compile(loss="mse", optimizer="rmsprop", metrics=["accuracy"])

  model.fit(trainX, trainY, verbose=1)
  return model
