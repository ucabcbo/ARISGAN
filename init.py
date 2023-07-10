import os
import json
import sys
import sis_toolbox as tbx
from sis_toolbox import RGBProfile as rgb
try:
    from preprocessing import snap_toolbox as stbx
except Exception as e:
    print(f'STBX not loaded: {str(e)}')

# Read the configuration file
with open('../env.json') as f:
    config = json.load(f)

# Access parameter values
PROJECT_ROOT = config['project_root']
DATA_ROOT = config['data_root']
OUTPUT_ROOT = config['output_root']

TILESIZE = config['tilesize']
TRAIN_DIR = os.path.join(DATA_ROOT, f'tfrecords{TILESIZE}_train/')
VAL_DIR = os.path.join(DATA_ROOT, f'tfrecords{TILESIZE}_val/')

TIF_DIR = os.path.join(DATA_ROOT, f'tif{TILESIZE}/')

sys.path.append(PROJECT_ROOT)

print('init loaded')
