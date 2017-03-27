"""dataset_factory.py
    Responsible for generating datasets by repeatedly calling UCL
    Camino. Definitions for the generated datasets come from
    src/config.py.
"""
from __future__ import print_function
from collections import OrderedDict
import errno
import os
import random
from subprocess import call

import pandas
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from src.config import DEFINITIONS, DATASET_SIZE, SIGNAL_NOISE_RATIO, STRIP_LIST
from src.routes import (DATA_PATH,
                        DATASYNTH_PATH,
                        FLOAT2TXT_PATH,
                        MEDIA_PATH,
                        RESULTS_PATH,
                        SCHEME_PATH)

# Datasynth uses different names than modelfit; the modelfit names are
# shorter so we prefer modelfit names
COMPARTMENT_MAP = {
    'gdrcylinders': 'gammadistribradiicylinders',
    'cylinder': 'cylindergpd',
    'sphere': 'spheregpd',
}

CAMINO_COMPARTMENTS = {
    "stick": ["d", "theta", "phi"],
    "cylinder": ["d", "theta", "phi", "R"],
    "gdrcylinders": ["k", "b", "d", "theta", "phi"],

    "ball": ["d"],
    "zeppelin": ["d", "theta", "phi", "d_perp1"],
    "tensor": ["d", "theta", "phi", "d_perp1", "d_perp2", "alpha"],

    "astrosticks": ["d"],
    "astrocylinders": ["d", "R"],
    "sphere": ["d", "Rs"],
    "dot": [],
}
def get_value_from_spec(dataset_spec_path, value):
    """Returns a value from a dataset spec"""
    with open(dataset_spec_path) as spec_file:
        spec_contents = spec_file.readlines()
        line = [line for line in spec_contents if '{}:'.format(value) in line]
        if len(line) == 1:
            return extract_line_value(line[0])
        else:
            raise 'Error in spec file: {}'.format(dataset_spec_path)

def get_file_name(path):
    """Gets the first part of a filename from a complete path"""
    return os.path.splitext(os.path.basename(path))[0]

def get_dataset_data_path(model):
    """Returns the dataset data path for a given model
    The data path contains the raw signal voxels
    """
    return "{}/{}/{}/{}".format(DATA_PATH,
                                get_file_name(SCHEME_PATH),
                                SIGNAL_NOISE_RATIO,
                                get_model_name(model))

def get_dataset_results_path(model):
    """Returns the dataset results path for a given model
    Results contains evaluations of model performance
    """
    return "{}/{}/{}/{}".format(RESULTS_PATH,
                                get_file_name(SCHEME_PATH),
                                SIGNAL_NOISE_RATIO,
                                get_model_name(model))

def get_dataset_media_path(model):
    """Returns the dataset media path for a given model
    Media contains graphs and visualisations
    """
    return "{}/{}/{}/{}".format(MEDIA_PATH,
                                get_file_name(SCHEME_PATH),
                                SIGNAL_NOISE_RATIO,
                                get_model_name(model))
def extract_line_value(line):
    """Returns the value after the colon of the line, stripping newlines
    and whitespace"""
    return line.split(':')[1].strip()

def get_param_names(model):
    """Gets the parameter names"""
    param_names = []
    for compartment in model:
        params = CAMINO_COMPARTMENTS[get_compartment_type(compartment)]
        param_names.append("{}_ivf".format(compartment))
        for param in params:
            param_names.append("{}_{}".format(compartment, param))
    return param_names

def get_compartment_type(compartment):
    """Returns the first part of a string {compartment}-{#} where #
    represents a number. Allows for duplicate compartments in models"""
    return compartment.split('-')[0]

def get_model_name(model):
    """Returns the model name given a list of compartments"""
    return "".join(list(map(get_compartment_type, model)))

def gen_dataset(model, position):
    """Generates a dataset given a specific model; the position
    parameter informs the placement of tqdm progress bars"""
    # TODO: Progress bars are currently buggy in how they display
    compartments = [compartment.lower() for compartment in model]
    name = get_model_name(model)
    output_path = get_dataset_data_path(model)
    try:
        os.makedirs(output_path)
        os.makedirs(output_path + '/raw')
        os.makedirs(output_path + '/float')
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(DATA_PATH):
            print("Dataset {} exists, skipping generation".format(name))
            return
        else:
            raise

    write_spec(compartments, output_path, name)

    with open("{}/{}.params".format(output_path, name), 'w') as param_file:
        with open("{}/{}.log".format(output_path, name), 'w') as log_file:
            print("Generating {} voxels for model {}".format(DATASET_SIZE, name))
            for i in tqdm(range(DATASET_SIZE), position=position, desc=" ".join(compartments)):
                camino_model = init_model(compartments)
                bfloat_path = "{}/raw/{}.Bfloat".format(output_path, str(i))
                gen_voxel(camino_model, "{}/raw/{}.Bfloat".format(output_path, str(i)), log_file)
                convert_voxel(bfloat_path, "{}/float/{}.float".format(output_path, str(i)))
                write_parameters(camino_model, param_file)

def convert_voxel(bfloat_path, output_path):
    """Converts a voxel from a Bfloat to a float using Camino's float2txt"""
    call("cat {} | {} > {}".format(bfloat_path, FLOAT2TXT_PATH, output_path), shell=True)

def gen_voxel(model, output_path, log):
    """Generates a voxel for a model, and stores the results at
    output_path"""
    cmd = ["{} -synthmodel compartment {}".format(DATASYNTH_PATH, len(model))]
    for index, compartment in enumerate(model):
        compartment_name = get_compartment_type(compartment)
        if compartment in COMPARTMENT_MAP:
            compartment_name = COMPARTMENT_MAP[compartment]

        if len(model) - 1 == index:
            # if it's the final compartment don't pass the ivf to camino
            parameters = stringify_parameters_no_ivf(model, compartment)
        else:
            parameters = stringify_parameters(model, compartment)

        cmd.append("{} {}".format(compartment_name, parameters))
    cmd.append("-schemefile {} -voxels 1 -snr {} > {}".format(SCHEME_PATH,
                                                              SIGNAL_NOISE_RATIO,
                                                              output_path))
    call(" ".join(cmd), shell=True, stderr=log)

def write_parameters(model, parameter_file):
    """Writes model parameters as a line in the parameter_file"""
    for compartment in model:
        print(stringify_parameters(model, compartment), end=" ", file=parameter_file)
    print(file=parameter_file)

def stringify_parameters(model, compartment):
    """Stringifies model parameters as a space separated string"""
    return " ".join(str(model[compartment][parameter]) for parameter in model[compartment])

def stringify_parameters_no_ivf(model, compartment):
    """Stringifies model parameters as a space separated string,
    excluding intra-volume fraction (ivf)"""
    return " ".join(str(model[compartment][parameter]) for
                    parameter in model[compartment] if parameter is not "ivf")

def write_spec(compartments, output_path, name):
    """Writes the model specification (the compartments and their
    parameters) to a file ending .params"""
    with open("{}/{}.spec".format(output_path, name), 'w') as definition_file:
        print("Name: {}".format(name), file=definition_file)
        for compartment in compartments:
            compartment_definition = CAMINO_COMPARTMENTS[get_compartment_type(compartment)]
            print("{} : {}".format(compartment, compartment_definition), file=definition_file)
        print(DEFINITIONS, file=definition_file)
        print("Noise: {}".format(SIGNAL_NOISE_RATIO), file=definition_file)
        print("Scheme: {}".format(get_file_name(SCHEME_PATH)), file=definition_file)
        print("N_voxels: {}".format(DATASET_SIZE), file=definition_file)

def init_model(compartments):
    """Initialises the ground truth parameters of a voxel for a single run on Camino.
    Behaviour depends on the definitions in src/config.py; if a single value is given
    for a parameter, then that value is used. Otherwise, a value is chosen in the range
    given using uniform sampling"""
    model = OrderedDict()
    run = gen_run()

    n_compartments = len(compartments)

    # for single compartment models, ivf = 1.0
    # for two compartment models, ivf1 = 0.35 - 0.65, ivf2 = 1 - ivf1
    # for three compartment models, ivf1 = 0.3 - 0.35, ivf2 = 0.3 - 0.35, ivf3 = 1 - (ivf1 + ivf2)
    max_ivf = 1.0
    if n_compartments == 2:
        ivf_range_min = 0.35
        ivf_range_max = 0.65
    else:
        ivf_range_min = 0.3
        ivf_range_max = 0.35

    for index, compartment in enumerate(compartments):
        if len(compartments) - 1 != index:
            ivf = random.uniform(ivf_range_min, ivf_range_max)
            max_ivf -= ivf
        else:
            ivf = max_ivf
        model[compartment] = gen_compartment(
            CAMINO_COMPARTMENTS[get_compartment_type(compartment)], run, ivf)
    return model

def gen_compartment(compartment_params, run, ivf):
    """Generates a single compartment of a model by mapping compartment
    parameters to the values defined in the run"""
    compartment = OrderedDict()
    compartment['ivf'] = ivf
    for param in compartment_params:
        compartment[param] = run[param]
    return compartment

def gen_run():
    """Picks parameter values for all possible model parameters based on
    the definitions in src/config.py. From the returned dictionary the
    defined model will then choose the required values"""
    run = dict()

    for name, param in DEFINITIONS.iteritems():
        if name == 'd':
            gen_diffusivities(run)
        elif isinstance(param, list):
            run[name] = random.uniform(param[0], param[1])
        else:
            run[name] = param
    return run

def gen_diffusivities(run):
    """Initialises the diffusivities for a run"""
    diffusivity = DEFINITIONS['d']
    if isinstance(diffusivity, float):
        run['d'] = diffusivity
        run['d_perp1'] = diffusivity
        run['d_perp2'] = diffusivity
    else:
        d_sample = random.uniform(diffusivity[0], diffusivity[1])
        run['d'] = d_sample
        run['d_perp1'] = d_sample
        run['d_perp2'] = d_sample

class Dataset(object):
    """Class containing information relating to a single dataset loaded
    from disc"""
    def __init__(self, model, scaler, train_x, train_y, test_x, test_y,
                 feature_names, name):
        self.name = name
        self.model = model
        self.path = get_dataset_data_path(model)
        self.spec_path = '{}/{}.spec'.format(self.path, self.name)

        self.scaler = scaler
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y

        self.feature_names = feature_names
        self._size = get_value_from_spec(self.spec_path, 'N_voxels')
        self._noise = int(get_value_from_spec(self.spec_path, 'Noise'))
        self._scheme = get_value_from_spec(self.spec_path, 'Scheme')

    def inverse_transform(self, vector):
        """Unscales a vector; could be the original scaled vector, or a
        prediction from a model trained with the original vector"""
        return pandas.DataFrame(self.scaler.inverse_transform(vector), columns=self.feature_names)

    @classmethod
    def from_model(cls, model):
        """Builds a Dataset from information on the disc"""
        name = get_model_name(model)
        data_path = get_dataset_data_path(model)
        float_path = data_path + "/float"
        feature_names = get_param_names(model)

        features = load_features(float_path)
        ground_truth = load_ground_truth(data_path, name, model, feature_names)

        feature_names = ground_truth.columns.tolist()

        scaler = StandardScaler()
        ground_truth = scaler.fit_transform(ground_truth)

        train_x, test_x, train_y, test_y = train_test_split(features, ground_truth, test_size=0.2)

        return cls(model, scaler, train_x, train_y, test_x, test_y, feature_names, name)

    @property
    def unique_string(self):
        return '{}_{}_{}'.format(self.name, self.size, self.noise)
    @property
    def size(self):
        """Returns the size of the dataset"""
        return self._size

    @property
    def noise(self):
        """Returns the noise of the dataset"""
        return self._noise

    @property
    def scheme(self):
        """Returns the scheme of the dataset"""
        return self._scheme

def compare_fnames(fname_a, fname_b):
    """Sorts filenames as numbers, removing the .float extension"""
    a_name = int(os.path.splitext(fname_a)[0])
    b_name = int(os.path.splitext(fname_b)[0])
    return 1 if a_name > b_name else 0 if a_name == b_name else -1

def load_features(float_path):
    """Loads features from files saved on disc"""
    float_files = os.listdir(float_path)
    float_files.sort(compare_fnames)
    features = [None] * len(float_files)

    for fname in float_files:
        vox_array = np.genfromtxt("{}/{}".format(float_path, fname))
        vox_number = int(filter(str.isdigit, fname))
        features[vox_number] = vox_array.flatten().tolist()

    return features

def load_ground_truth(data_path, name, model, feature_names):
    """Loads ground truth parameters from {modelname}.params file"""
    ground_truth = np.genfromtxt("{}/{}.params".format(data_path, name))

    skip_list = []
    for compartment in model:
        skip_list += ["{}_{}".format(compartment, item) for item in STRIP_LIST]

    ground_truth = pandas.DataFrame(ground_truth, columns=feature_names)
    filtered_ground_truth = ground_truth[ground_truth.columns.difference(skip_list)]

    return filtered_ground_truth
