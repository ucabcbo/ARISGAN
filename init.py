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
try:
    with open('../env.json') as f:
        config = json.load(f)
except:
    with open('env.json') as f:
        config = json.load(f)

# Access parameter values
ENVIRONMENT = config['environment']

PROJECT_ROOT = config['project_root']
DATA_ROOT = config['data_root']
OUTPUT_ROOT = config['output_root']

TILESIZE = config['tilesize']
IMG_WIDTH = config['img_width']
IMG_HEIGHT = config['img_height']

TRAIN_DIR = os.path.join(DATA_ROOT, f'tfrecords{TILESIZE}/')
VAL_DIR = os.path.join(DATA_ROOT, f'tfrecords{TILESIZE}_val/')

TIF_DIR = os.path.join(DATA_ROOT, f'tif{TILESIZE}/')

from datetime import datetime
TIMESTAMP = datetime.now().strftime("%m%d-%H%M")

sys.path.append(PROJECT_ROOT)

print('init loaded')
