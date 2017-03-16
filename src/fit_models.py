import os
from subprocess import call
from tqdm import tqdm

from src.routes import double2txt_path, make_path_ignoring_existing, modelfit_path, scheme_path
from src.generation.model_factory import get_dataset_path, get_model_name

camino_fits = [
  "Zeppelin",
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

def fit_model_voxels(model):
  model_name =  "".join(model)

  if model_name not in (name.lower() for name in camino_fits):
    print("{} is not in list of camino_fits; fits will not be generated".format(model_name))
    return


  dataset_path = get_dataset_path(model)
  raw_path = dataset_path + '/raw'
  fit_path = dataset_path + '/camino_fits'
  make_path_ignoring_existing(fit_path)

  bfloatFiles = [filename for filename in os.listdir(raw_path)]

  print('Fitting using LM for {}'.format(model_name))
  for bfloatFile in tqdm(bfloatFiles):
    index = int(os.path.splitext(bfloatFile)[0])

    cmd = "{} -inputfile {}/{} -fitmodel {} -fitalgorithm LM -schemefile {} | {} > {}/{}.txt".format(
      modelfit_path,
      raw_path, bfloatFile,
      model_name,
      scheme_path,
      double2txt_path,
      fit_path,
      index)
    call(cmd, shell=True)
