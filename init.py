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
    with open('../environment.json') as f:
        environment = json.load(f)
except:
    with open('environment.json') as f:
        environment = json.load(f)

try:
    with open('../experiment.json') as f:
        experiment = json.load(f)
except:
    with open('experiment.json') as f:
        experiment = json.load(f)


EXP = experiment

MODEL_NAME = experiment['model_name']
BATCH_SIZE = experiment['batch_size']
SHUFFLE = experiment['shuffle']
STEPS = experiment['steps']

DATA_SAMPLE = None
if experiment['sample_train'] is not None and experiment['sample_val'] is not None:
    DATA_SAMPLE = (experiment['sample_train'],experiment['sample_val'])
else:
    if experiment['sample_train'] is not None or experiment['sample_val'] is not None:
        print('W: Both sample_train and sample_val must be set for any to take effect.')

SAMPLE_FREQ = environment['sample_freq']
CKPT_FREQ = environment['ckpt_freq']

GEN_LOSS = experiment['gen_loss']
DISC_LOSS = experiment['disc_loss']

#TODO: as those parameters don't change often in my experiments, I put them
# to the environment.json file to keep the experiment.json more clean.
# They should be in the experiment section though.
TILESIZE = environment['tilesize']
IMG_WIDTH = environment['img_width']
IMG_HEIGHT = environment['img_height']
INPUT_CHANNELS = environment['input_channels']
OUTPUT_CHANNELS = environment['output_channels']

# Access parameter values
ENVIRONMENT = environment['environment']

PROJECT_ROOT = environment['project_root']
DATA_ROOT = environment['data_root']
OUTPUT_ROOT = environment['output_root']

TRAIN_DIR = os.path.join(DATA_ROOT, f'tfrecords{TILESIZE}/')
VAL_DIR = os.path.join(DATA_ROOT, f'tfrecords{TILESIZE}_val/')
TIF_DIR = os.path.join(DATA_ROOT, f'tif{TILESIZE}/')

from datetime import datetime
TIMESTAMP = datetime.now().strftime("%m%d-%H%M")

SUBFOLDER = f'{TIMESTAMP}_{MODEL_NAME}_{BATCH_SIZE}x{TILESIZE}'

OUTPUT_LOGS = os.path.join(OUTPUT_ROOT, f'{SUBFOLDER}/logs/')
OUTPUT_CKPT = os.path.join(OUTPUT_ROOT, f'{SUBFOLDER}/ckpt/')
OUTPUT_SAMPLES = os.path.join(OUTPUT_ROOT, f'{SUBFOLDER}/samples/')
OUTPUT_MODEL = os.path.join(OUTPUT_ROOT, f'{SUBFOLDER}/model/')

sys.path.append(PROJECT_ROOT)

def setup_output():
    os.makedirs(OUTPUT_LOGS, exist_ok=True)
    os.makedirs(OUTPUT_CKPT, exist_ok=True)
    os.makedirs(OUTPUT_SAMPLES, exist_ok=True)
    os.makedirs(OUTPUT_MODEL, exist_ok=True)

    with open(os.path.join(OUTPUT_MODEL, 'experiment.json'), 'w') as f:
        json.dump(experiment, f, indent=4)

print('init loaded')
