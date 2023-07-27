# import snappy
# from snappy import GPF
# from snappy import ProductIO

import os
import rasterio

import sys
sys.path.append('../')
from environment import Environment
import toolbox as tbx

import random

env = Environment()

TILESIZE = 256

# files = os.listdir(os.path.join(env.DATA_ROOT, f'tif{TILESIZE}/'))
files = os.listdir('/home/cb/sis2/data/curated/256/tif')
selected_files = files
# selected_files = random.sample(files, k=5490)

for file in selected_files:

    filename = os.path.splitext(file)[0]
    print('Filename:', filename)

    suffix = random.choices(['train', 'val'], [0.63747, 0.36253])[0]
    print('Suffix:', suffix)

    raw_tiff = rasterio.open(os.path.join('/home/cb/sis2/data/curated/256/tif', file))

    subfolder = f'newalt6/{TILESIZE}/{suffix}/'
    if not os.path.exists(os.path.join(env.DATA_ROOT, subfolder)):
        os.makedirs(os.path.join(env.DATA_ROOT, subfolder))
    tensorpath = os.path.join(env.DATA_ROOT, subfolder, f'{filename}.tfrecord')
    if os.path.exists(tensorpath):
        print(f'Already exists: {tensorpath}')
    else:
        tbx.save_tfrecord_alt(raw_tiff, tensorpath, downsample=6)

    subfolder = f'newalt12/{TILESIZE}/{suffix}/'
    if not os.path.exists(os.path.join(env.DATA_ROOT, subfolder)):
        os.makedirs(os.path.join(env.DATA_ROOT, subfolder))
    tensorpath = os.path.join(env.DATA_ROOT, subfolder, f'{filename}.tfrecord')
    if os.path.exists(tensorpath):
        print(f'Already exists: {tensorpath}')
    else:
        tbx.save_tfrecord_alt(raw_tiff, tensorpath, downsample=12)

    subfolder = f'newalt30/{TILESIZE}/{suffix}/'
    if not os.path.exists(os.path.join(env.DATA_ROOT, subfolder)):
        os.makedirs(os.path.join(env.DATA_ROOT, subfolder))
    tensorpath = os.path.join(env.DATA_ROOT, subfolder, f'{filename}.tfrecord')
    if os.path.exists(tensorpath):
        print(f'Already exists: {tensorpath}')
    else:
        tbx.save_tfrecord_alt(raw_tiff, tensorpath, downsample=30)

    subfolder = f'newcurated/{TILESIZE}/{suffix}/'
    if not os.path.exists(os.path.join(env.DATA_ROOT, subfolder)):
        os.makedirs(os.path.join(env.DATA_ROOT, subfolder))
    tensorpath = os.path.join(env.DATA_ROOT, subfolder, f'{filename}.tfrecord')
    if os.path.exists(tensorpath):
        print(f'Already exists: {tensorpath}')
    else:
        tbx.save_tfrecord(raw_tiff, tensorpath)
