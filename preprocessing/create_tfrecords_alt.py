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

files = os.listdir(os.path.join(init.DATA_ROOT, 'tif256/'))
selected_files = random.sample(files, k=5490)

for file in selected_files:
    filename = os.path.splitext(file)[0]
    print('Filename:', filename)

    raw_tiff = rasterio.open(os.path.join(init.DATA_ROOT, 'tif256', file))
    tensorpath = os.path.join(init.DATA_ROOT, 'tfrecords256_alt', f'{filename}.tfrecord')
    tbx.save_tfrecord_alt(raw_tiff, tensorpath, downsample=6)
