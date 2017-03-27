"""convolutional_nn.py
    Uses Keras to implement several neural network topologies
"""
from keras.models import Sequential
from keras.layers import Dense

def search_convolutional_nn(dataset, neurons=[13, 13], n_hidden_layers=1):
    """Generates a neural network model defined by the input topology
    description."""
    # pylint: disable=dangerous-default-value
    # Input list is never modified by the function
    model = Sequential()
    model.add(Dense(neurons[0], input_dim=len(dataset.train_y[0]), activation='relu'))
    for i in range(1, n_hidden_layers + 1):
        model.add(Dense(neurons[i], activation='relu'))
    model.add(Dense(output_dim=len(dataset.train_y[0])))

    model.compile(loss="mse", optimizer="rmsprop")

    model.fit(dataset.train_x, dataset.train_y, verbose=1, nb_epoch=150)
    return model

def nn_search_item(n_neurons, n_layers):
    """Generates keyword arguments which can be passed to
    search_convolutional_nn to generate a neural network topology"""
    kwargs = {
        'neurons': [n_neurons] * (n_layers + 1),
        'n_hidden_layers': n_layers
    }
    return ('ConvolutionalNN_{}_{}'.format(n_layers, n_neurons), search_convolutional_nn, kwargs)

def initialise_nn_grid_search(neurons, layers):
    """Initialises a list of keyword arguments which can be used as
    elements in a grid search of neural network topologies"""
    grid_search = []
    for layer in layers:
        for neuron in neurons:
            grid_search.append(nn_search_item(neuron, layer))
    return grid_search
