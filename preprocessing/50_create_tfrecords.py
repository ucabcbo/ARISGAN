### Arguments
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--inventory", required=True, help='Inventory code (folder name within _tif directory)')
parser.add_argument("--dataset", required=True, help='Dataset Suffix in Output directory')
parser.add_argument("--train_ratio", required=False, default=0.9, type=float, help='Ratio of training vs. test files')
parser.add_argument("--tilesize", required=False, default=256, type=int, help='Tilesize')

a = parser.parse_args()
print(f'Arguments read: {a}')
### End arguments


INVENTORY = a.inventory
DATASET = a.dataset
TRAIN_RATIO = a.train_ratio
TILESIZE = a.tilesize

import os
import rasterio
import random

import sys
sys.path.append('../')
from environment import Environment
import toolbox as tbx


env = Environment()

tifdir = os.path.join(env.DATA_ROOT, '_tif', INVENTORY, str(TILESIZE))
files = os.listdir(tifdir)

outputdir_cur = os.path.join(env.DATA_ROOT, f'cur_{DATASET}', str(TILESIZE))
outputdir_alt3 = os.path.join(env.DATA_ROOT, f'alt3_{DATASET}', str(TILESIZE))
outputdir_alt6 = os.path.join(env.DATA_ROOT, f'alt6_{DATASET}', str(TILESIZE))
outputdir_alt12 = os.path.join(env.DATA_ROOT, f'alt12_{DATASET}', str(TILESIZE))
outputdir_alt30 = os.path.join(env.DATA_ROOT, f'alt30_{DATASET}', str(TILESIZE))

outputdirs = [outputdir_cur, outputdir_alt3, outputdir_alt6, outputdir_alt12, outputdir_alt30]

for outputdir in outputdirs:
    if not os.path.exists(os.path.join(outputdir, 'train')):
        os.makedirs(os.path.join(outputdir, 'train'))
    if not os.path.exists(os.path.join(outputdir, 'val')):
        os.makedirs(os.path.join(outputdir, 'val'))


for file in files:

    filename = os.path.splitext(file)[0]
    trainval = random.choices(['train', 'val'], [TRAIN_RATIO, (1-TRAIN_RATIO)])[0]

    raw_tiff = rasterio.open(os.path.join(tifdir, file))

    if os.path.exists(os.path.join(outputdir_cur, 'train', f'{filename}.tfrecord')) or os.path.exists(os.path.join(outputdir_cur, 'val', f'{filename}.tfrecord')):
        print(f'Already exists: {filename}.tfrecord')
    else:
        tbx.save_tfrecord(raw_tiff, os.path.join(outputdir_cur, trainval, f'{filename}.tfrecord'))
        tbx.save_tfrecord_alt(raw_tiff, os.path.join(outputdir_alt3, trainval, f'{filename}.tfrecord'), downsample=3)
        tbx.save_tfrecord_alt(raw_tiff, os.path.join(outputdir_alt6, trainval, f'{filename}.tfrecord'), downsample=6)
        tbx.save_tfrecord_alt(raw_tiff, os.path.join(outputdir_alt12, trainval, f'{filename}.tfrecord'), downsample=12)
        tbx.save_tfrecord_alt(raw_tiff, os.path.join(outputdir_alt30, trainval, f'{filename}.tfrecord'), downsample=30)
        print(f'Saved tfrecords for {filename}')
