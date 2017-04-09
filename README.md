# rm_ml_experiments

## Installation

### General Dependencies:
  - javac 1.7+
  - git
  - python2.7

### Camino installation
Run `make install`; you may need to call this with sudo depending on
your operating system

### Python installation
- Initialise a virtual environment (you need pip for this)
  - `virtualenv -p /usr/bin/python2.7 ml_research`
  - the directory may be different depending on your OS
- Activate the environment using source ml\_research/bin/activate (your prompt should change)
- Install the requirements using pip install -r requirements.txt

## Makefile commands

- `generate`: Generates voxel files; output path will be `data/{dataset_name}`
- `train-all`: Train on all models matching current src/config.py and
  produce visualisations
- `install` : Installs dependencies (including camino); see [Installation](#Installation)
- `clean_data` : Deletes all model data generated by the program
- `clean` : Deletes all files installed by the program

## Configuration
Configurations for voxel generation and training are defined in the
_config.py_ file. An example is given in [config.py.example](./src/config.py.example)
### Models
Each configuration is a list of strings in the models list, i.e:

```
models = [
  ["compartment1", "compartment2", "compartment3"],
]
```

### Meta-parameters
The number of voxels to generate can also be set

```
dataset_size = 12500
```

As well as the signal to noise ratio

```
signal_to_noise = 20
```

The values supplied for a parameter can either be a single number value, or a list
with two values. If two values are given, a sample will be taken
in the range provided using uniform sampling.

### Filtering
To filter out parameters you can define a list of parameters to strip
from the training data

```
strip = ["theta", "phi", "alpha"]
```
## Notes on running
If running on a mac, it is suggested to use `caffeinate -i`, so that the
mac won't sleep until the tasks are completed:

`caffeinate -i python main.py generate && caffeinate -i python main.py train-all`


```
.
├── camino            # The Camino repository used to generate data
├── schemes
|   ├── PGSE_90_100t.scheme      # A scheme using a gradient strength of 100
|   └── ...
├── scripts
|   ├── run_generate      # Generates data for an Azuru machine
|   └── setup_ubuntu      # Sets up an Azure Ubuntu machine
├── src
|   ├── datasets          # Dataset Generation Module
|      ├── dataset_factory.py   # factory to generate and import data
|      └── gen_voxels.py        # entry point for data generation using threads
|   ├── fitting           # Fitting Module
|      └── fit_models.py        # Uses Camino to fit compartment models to generated data
|   ├── machine_learning  # Machine Learning module
|      ├── algortihms           # Contains the machine learning algorithms used to train models
|         ├── convolutional_nn.py         # Convolutional Neural Network using Keras and Tensorflow
|         ├── knn.py                      # K-Nearest Neighbours using Scikit-Learn
|         ├── multilayer_perceptron.py    # Multilayer Perceptron using Scikit-Learn
|         ├── random_forest.py            # Random Forest using Scikit-Learn
|         ├── ridge_regression.py         # Ridge Regression using Scikit-Learn
|         └── svm.py                      # Support Vector Machine using Scikit-Learn
|      ├── experiment.py        # Class encapsulating a single experiment (algorithm, train_dataset, test_dataset)
|      └── ...
|   └── visualisation   # Visualisation Module
|      ├── graphs           # Various graphs
|         └── ...
|      ├── selectors        # Contains selector methods to select subsets of all the trained data
|         └── ...
|      └── ...
├── main.py           # The entry point for Camino-ml
├── Makefile          # A makefile which simplifies using main.py
├── Media          # A .gitignored folder containing visualisations
├── Results          # A .gitignored folder containing results
├── Data          # A .gitignored folder containing generated data
└── ...
```
