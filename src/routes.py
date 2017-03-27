"""
Routing constants definitions
    - Routing is based on the relative paths from the root directory
    - Routing shouldn't be changed after data has been generated, since
      the data won't be findable
"""
import os
import errno

from src.config import EXPLICIT_SCHEME, GRADIENT_STRENGTH

ROOT_PATH = os.path.abspath(os.getcwd())
CONFIG_FILE_PATH = ROOT_PATH + '/config.json'

if EXPLICIT_SCHEME is not None:
    SCHEME_PATH = ROOT_PATH + '/schemes/EXPLICIT_SCHEME'
else:
    SCHEME_PATH = ROOT_PATH + '/schemes/PGSE_90_{}t.scheme'.format(GRADIENT_STRENGTH)

DATA_PATH = ROOT_PATH + '/data'
MEDIA_PATH = ROOT_PATH + '/media'
RESULTS_PATH = ROOT_PATH + '/results'

CAMINO_BIN_PATH = ROOT_PATH + '/camino/bin'

DATASYNTH_PATH = CAMINO_BIN_PATH + '/datasynth'
MODELFIT_PATH = CAMINO_BIN_PATH + '/modelfit'

FLOAT2TXT_PATH = CAMINO_BIN_PATH + '/float2txt'
DOUBLE2TXT_PATH = CAMINO_BIN_PATH + '/double2txt'

def make_path_ignoring_existing(path):
    """Makes a directory, including all intermediate directories given,
    and doesn't throw an error if the directory already exists"""
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
