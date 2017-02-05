from subprocess import call, check_output
from settings import root_path, datasynth_path, scheme_path

def gen_voxels():
  cmd = build_command([['CYLINDERGPD', 0.6, '1.7E-9', 0.0, 0.0, '4E-6'],
                       ['zeppelin', 0.1, '1.7E-9', 0.0, 0.0, '2E-10'],
                       ['Dot']], '{}'.format(root_path + '/tmp'))
  call(cmd)

def build_command(compartments, output_path):
  cmd = [datasynth_path, '-synthmodel']
  cmd.append('compartment {}'.format(str(len(compartments))))
  for compartment in compartments:
    for item in compartment:
      cmd.append(str(item))
  cmd.append('-schemefile {} -voxels 1'.format(scheme_path))
  cmd.append('-outputfile {}'.format(output_path))
  return cmd
gen_voxels()
