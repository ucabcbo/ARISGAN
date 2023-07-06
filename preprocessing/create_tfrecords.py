import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import tensorflow as tf

import sys
sys.path.append('./')
import sis_toolbox as toolbox
from sis_toolbox import RGBProfile as rgb

# PATH = '/Users/christianboehm/projects/sis2/data'
PATH = '/home/cb/sis2/data/'
TILESIZE = 256

# tif_files = [os.path.join(PATH, f'tif{TILESIZE}/00005_12x3405.tif')]
tif_files = glob.glob(os.path.join(PATH, f'tif{TILESIZE}/*.tif'))
for tif_file in tif_files:
    tif_filename = os.path.splitext(os.path.basename(tif_file))[0]
    tensor_filename = os.path.join(PATH, f'tfrecords{TILESIZE}/{tif_filename}.tfrecord')

    if os.path.exists(tensor_filename):
        print(f'File {tensor_filename} already exists')
        os.rename(tif_file, f'{tif_file}_tfexisting')
        continue
    print(f'File {tensor_filename} is new')

    raw_tiff = rasterio.open(tif_file)
    toolbox.save_tfrecord(raw_tiff, tensor_filename)
    os.rename(tif_file, f'{tif_file}_tfcreated')