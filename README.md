# rm_ml_experiments

## Installation

Dependencies:
  - javac 1.7+
  - git
  - python3 (as python3 on path)

Run `make install`; you may need to call this with sudo.

## Makefile commands

- `gen_data`: Generates voxel files; output path will be `data/{dataset_name}/{fname}.BFloat`
- `install` : Installs dependencies (including camino); see [Installation](#Installation)
- `clean_data` : Deletes all data generated by the program
- `clean` : Deletes all files installed by the program, as well as running `make clean_data`

## Configuration

__Note: Parameters for a model must be in the order camino expects them. The Parameter
  names are not used by the script, and are for reference only__

Configurations for voxel generation are defined in the [config.json](./config.json) file,
and loaded by logic in the [loader.py](./loader.py) file. Each configuration takes
the following form:

```json
"{dataset_name}": {
    "fname": "{output file name}",
    "models": "{compartment models used by camino}",
  }
```

The models take the form:

```json
"{model_name}": {
    "{parameter_name}": {value or [value, value]}
  }
```

The values supplied for a parameter can either be a numerical value, or a list
of two values. If two values are given, a sample will be taken
in the range provided.

Parameter names have no internal use in the program (at the moment), but are
useful for understanding the config.json file. This means that parameters are
given to camino in the order of declaration, not based on the name of the parameter.
