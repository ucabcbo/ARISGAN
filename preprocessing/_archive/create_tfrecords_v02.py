# import snappy
# from snappy import GPF
# from snappy import ProductIO

import os
import rasterio

import sys
sys.path.append('../')
import environment
import toolbox as tbx

import random

files = os.listdir(os.path.join(environment.DATA_ROOT, f'tif{environment.TILESIZE}/'))
selected_files = files
# selected_files = random.sample(files, k=5490)

for file in selected_files:

    filename = os.path.splitext(file)[0]
    print('Filename:', filename)

    suffix = random.choices(['train', 'val'], [0.637, 0.363])[0]
    print('Suffix:', suffix)

    raw_tiff = rasterio.open(os.path.join(environment.DATA_ROOT, f'tif{environment.TILESIZE}/', file))

    subfolder = f'alt6/tfrecords{environment.TILESIZE}_{suffix}/'
    tensorpath = os.path.join(environment.DATA_ROOT, subfolder, f'{filename}.tfrecord')
    if os.path.exists(tensorpath):
        print(f'Already exists: {tensorpath}')
    else:
        tbx.save_tfrecord_alt(raw_tiff, tensorpath, downsample=6)

    subfolder = f'alt12/tfrecords{environment.TILESIZE}_{suffix}/'
    tensorpath = os.path.join(environment.DATA_ROOT, subfolder, f'{filename}.tfrecord')
    if os.path.exists(tensorpath):
        print(f'Already exists: {tensorpath}')
    else:
        tbx.save_tfrecord_alt(raw_tiff, tensorpath, downsample=12)

    subfolder = f'alt30/tfrecords{environment.TILESIZE}_{suffix}/'
    tensorpath = os.path.join(environment.DATA_ROOT, subfolder, f'{filename}.tfrecord')
    if os.path.exists(tensorpath):
        print(f'Already exists: {tensorpath}')
    else:
        tbx.save_tfrecord_alt(raw_tiff, tensorpath, downsample=30)

    subfolder = f'curated/tfrecords{environment.TILESIZE}_{suffix}/'
    tensorpath = os.path.join(environment.DATA_ROOT, subfolder, f'{filename}.tfrecord')
    if os.path.exists(tensorpath):
        print(f'Already exists: {tensorpath}')
    else:
        tbx.save_tfrecord(raw_tiff, tensorpath)
