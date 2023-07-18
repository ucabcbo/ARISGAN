import os
import json
import sys

# Read the configuration file
possiblepaths = ['environment.json',
                 '../environment.json',
                 '/home/ucabcbo/sis2/environment.json']
environment = None
for possiblepath in possiblepaths:
    if os.path.exists(possiblepath):
        with open(possiblepath) as f:
            environment = json.load(f)
        break

assert environment is not None

#TODO: as those parameters don't change often in my experiments, I put them
# to the environment.json file to keep the experiment.json more clean.
# They should be in the experiment section though.
TILESIZE = environment['tilesize']
IMG_WIDTH = environment['img_width']
IMG_HEIGHT = environment['img_height']
INPUT_CHANNELS = environment['input_channels']
OUTPUT_CHANNELS = environment['output_channels']
SAMPLE_FREQ = environment['sample_freq']
CKPT_FREQ = environment['ckpt_freq']
PARSE_CODE = environment['parse_code']

MAX_SHUFFLE_BUFFER = environment['max_shuffle_buffer']

ENVIRONMENT = environment['environment']

PROJECT_ROOT = environment['project_root']
DATA_ROOT = environment['data_root']
OUTPUT_ROOT = environment['output_root']

TRAIN_DIR = os.path.join(DATA_ROOT, f'tfrecords{TILESIZE}_train/')
VAL_DIR = os.path.join(DATA_ROOT, f'tfrecords{TILESIZE}_val/')
TIF_DIR = os.path.join(DATA_ROOT, f'tif{TILESIZE}/')

from datetime import datetime
TIMESTAMP = datetime.now().strftime("%m%d-%H%M")

sys.path.append(PROJECT_ROOT)

print('init loaded')
