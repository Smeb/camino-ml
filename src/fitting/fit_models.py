import os
from subprocess import call

from joblib import Parallel, delayed
from tqdm import tqdm

from src.config import models
from src.routes import double2txt_path, make_path_ignoring_existing, modelfit_path, scheme_path
from src.generation.model_factory import get_dataset_path, get_model_name

camino_fits = [
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

fit_map = {
  'ZeppelinZeppelin': 'BiZeppelin',
  'Tensor': 'DT'
}

camino_fits = [model.lower() for model in camino_fits]

def fit_all_models():
  Parallel(n_jobs=-1)(delayed(fit_model_voxels)(model, i) for model, i in zip(models, range(len(models))))

def fit_model_voxels(model, position):
  model_name =  re.sub(r"-[0-9]", "","".join(model))
  print(model_name)
  if model_name in fit_map:
    model_name = fit_map[model_name]
  if model_name not in camino_fits:
    print("{} is not in list of camino_fits; fits will not be generated".format(model_name))
    return

  dataset_path = get_dataset_path(model)
  raw_path = dataset_path + '/raw'
  fit_path = dataset_path + '/camino_fits'
  make_path_ignoring_existing(fit_path)

  bfloatFiles = [filename for filename in os.listdir(raw_path)]

  print('Fitting using LM for {}'.format(model_name))
  for bfloatFile in tqdm(bfloatFiles, position=position, desc=" ".join(model)):
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
