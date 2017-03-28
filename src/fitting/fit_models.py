"""fit_models.py
    Fits models to all voxels in a dataset using the Camino fitting process
"""
import errno
import os
from subprocess import call
import re

from joblib import Parallel, delayed
from tqdm import tqdm

from src.config import MODELS
from src.routes import DOUBLE2TXT_PATH, MODELFIT_PATH, SCHEME_PATH
from src.datasets.dataset_factory import get_dataset_data_path

CAMINO_FITS = [
    "DT",
    "Tensor",
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
    # 'zeppelinzeppelin': 'BiZeppelin',
}

LC_CAMINO_FITS = [fit_name.lower() for fit_name in CAMINO_FITS]

def fit_all_models():
    """Fits all models using parallel threads to speed up the fitting proceess"""
    # TODO: Parallelize on a per voxel basis
    n_models = range(len(MODELS))
    Parallel(n_jobs=-1)(delayed(fit_model_voxels)(model, i) for model, i in zip(MODELS, n_models))

def fit_model_voxels(model, position):
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

    print('Fitting using LM for {}'.format(model_name))
    for bfloat_file in tqdm(bfloat_files, position=position, desc=" ".join(model)):
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
