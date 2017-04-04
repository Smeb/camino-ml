"""fit_models.py
    Fits models to all voxels in a dataset using the Camino fitting process
"""
import errno
import os
from subprocess import call
import re
from multiprocessing import cpu_count

from joblib import Parallel, delayed
from tqdm import tqdm

from src.config import MODELS
from src.routes import DOUBLE2TXT_PATH, MODELFIT_PATH, SCHEME_PATH
from src.datasets.dataset_factory import get_dataset_data_path

CAMINO_FITS = [
    "DT",
    "BiZeppelin",
    "BallStick",
    "BallCylinder",
    "BallGDRCylinders",
    "ZeppelinStick",
    "ZeppelinCylinder",
    "ZeppelinGDRCylinders",
    "TensorStick",
    "TensorCylinder",
    "TensorGDRCylinders",
    "BallStickDot",
    "BallCylinderDot",
    "BallGDRCylindersDot",
    "ZeppelinStickDot",
    "ZeppelinCylinderDot",
    "ZeppelinGDRCylindersDot",
    "TensorStickDot",
    "TensorCylinderDot",
    "TensorGDRCylindersDot",
    "BallStickAstrosticks",
    "BallCylinderAstrosticks",
    "BallGDRCylindersAstrosticks",
    "ZeppelinStickAstrosticks",
    "ZeppelinCylinderAstrosticks",
    "ZeppelinGDRCylindersAstrosticks",
    "TensorStickAstrosticks",
    "TensorCylinderAstrosticks",
    "TensorGDRCylindersAstrosticks",
    "BallStickAstrocylinders",
    "BallCylinderAstrocylinders",
    "BallGDRCylindersAstrocylinders",
    "ZeppelinStickAstrocylinders",
    "ZeppelinCylinderAstrocylinders",
    "ZeppelinGDRCylindersAstrocylinders",
    "TensorStickAstrocylinders",
    "TensorCylinderAstrocylinders",
    "TensorGDRCylindersAstrocylinders",
    "BallStickSphere",
    "BallCylinderSphere",
    "BallGDRCylindersSphere",
    "ZeppelinStickSphere",
    "ZeppelinCylinderSphere",
    "ZeppelinGDRCylindersSphere",
    "TensorStickSphere",
    "TensorCylinderSphere",
    "TensorGDRCylindersSphere",
]

FIT_MAP = {
    # Note: Cannot fit BiZeppelin models when they're facing the same
    # direction
    'DT': 'Tensor',
    # 'zeppelinzeppelin': 'BiZeppelin',
}

LC_CAMINO_FITS = [fit_name.lower() for fit_name in CAMINO_FITS]

def fit_all_models():
    """Fits all models in MODELS"""
    for model in MODELS:
        fit_model_voxels(model)

def fit_model_voxels(model):
    """Fits all voxels in a given model; the position parameter controls
    the placement of tqdm bars"""
    model_name = re.sub(r"-[0-9]", "", "".join(model)).lower()
    if model_name in FIT_MAP:
        model_name = FIT_MAP[model_name]
    print(model_name)
    if model_name not in LC_CAMINO_FITS:
        print('Model {} cannot be fit by Camino; skipping'.format(model_name))
        return

    dataset_path = get_dataset_data_path(model)
    raw_path = dataset_path + '/raw/test'
    fit_path = dataset_path + '/camino_fits'
    try:
        os.makedirs(fit_path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(fit_path):
            print("Model {} has already been fit, skipping".format(model_name))
            return
        else:
            raise

    bfloat_files = [filename for filename in os.listdir(raw_path)]

    n_cpus = cpu_count()
    bfloat_groups = [[] for _ in range(n_cpus)]
    for index, bfloat_file in enumerate(bfloat_files):
        bfloat_groups[index % n_cpus].append(bfloat_file)

    print('Fitting using LM for {}'.format(model_name))
    Parallel(n_jobs=-1)(delayed(fit_bfloats)(
        bfloat_group, model_name, raw_path, fit_path, position) for bfloat_group, position
                        in zip(bfloat_groups, range(len(bfloat_groups))))

def fit_bfloats(bfloat_files, model_name, raw_path, fit_path, position):
    """fits a compartment models to a list of voxels given in bfloat
    format and records the result at fit_path"""
    for bfloat_file in tqdm(bfloat_files, position=position):
        index = int(os.path.splitext(bfloat_file)[0])

        outfile = "{}/{}.txt".format(fit_path, index)
        if model_name == 'tensor':
            fit_command = "-model dt"
        else:
            fit_command = "-fitmodel {}".format(model_name)
        cmd = "{} -inputfile {}/{} {} -fitalgorithm LM -schemefile {} | {} > {}".format(
            MODELFIT_PATH,
            raw_path, bfloat_file,
            fit_command,
            SCHEME_PATH,
            DOUBLE2TXT_PATH,
            outfile)
        call(cmd, shell=True)
