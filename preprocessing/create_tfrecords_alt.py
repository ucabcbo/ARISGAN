# import snappy
# from snappy import GPF
# from snappy import ProductIO

import os
import rasterio

import sys
sys.path.append('../')
import init
import sis_toolbox as tbx

import random

files = os.listdir(os.path.join(init.DATA_ROOT, f'tif{init.TILESIZE}/'))
selected_files = files
# selected_files = random.sample(files, k=5490)

for file in selected_files:

    filename = os.path.splitext(file)[0]
    print('Filename:', filename)

    suffix = random.choices(['train', 'val'], [0.9, 0.1])[0]
    print('Suffix:', suffix)

    raw_tiff = rasterio.open(os.path.join(init.DATA_ROOT, f'tif{init.TILESIZE}/', file))

    subfolder = f'alt6/tfrecords{init.TILESIZE}_{suffix}/'
    tensorpath = os.path.join(init.DATA_ROOT, subfolder, f'{filename}.tfrecord')
    if os.path.exists(tensorpath):
        print(f'Already exists: {tensorpath}')
    else:
        tbx.save_tfrecord_alt(raw_tiff, tensorpath, downsample=6)

    subfolder = f'alt12/tfrecords{init.TILESIZE}_{suffix}/'
    tensorpath = os.path.join(init.DATA_ROOT, subfolder, f'{filename}.tfrecord')
    if os.path.exists(tensorpath):
        print(f'Already exists: {tensorpath}')
    else:
        tbx.save_tfrecord_alt(raw_tiff, tensorpath, downsample=12)

    subfolder = f'alt30/tfrecords{init.TILESIZE}_{suffix}/'
    tensorpath = os.path.join(init.DATA_ROOT, subfolder, f'{filename}.tfrecord')
    if os.path.exists(tensorpath):
        print(f'Already exists: {tensorpath}')
    else:
        tbx.save_tfrecord_alt(raw_tiff, tensorpath, downsample=30)

    subfolder = f'curated/tfrecords{init.TILESIZE}_{suffix}/'
    tensorpath = os.path.join(init.DATA_ROOT, subfolder, f'{filename}.tfrecord')
    if os.path.exists(tensorpath):
        print(f'Already exists: {tensorpath}')
    else:
        tbx.save_tfrecord(raw_tiff, tensorpath)
