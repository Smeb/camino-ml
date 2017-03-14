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
