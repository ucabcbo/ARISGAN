import sys
sys.path.append('/home/cb/sis2/')

with open('env.txt') as f:
    ENVIRONMENT = f.readlines()[0][:-1]
print(f'running on environment: "{ENVIRONMENT}"')
assert ENVIRONMENT in ['blaze',
                       'colab',
                       'local',
                       'cpom']

import os
import sys
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

sys.path.append(os.path.expanduser('~/.snap/snap-python'))
import snappy
from snappy import ProductIO
from snappy import jpy
HashMap = snappy.jpy.get_type('java.util.HashMap')

sys.path.append('../')
import sis_toolbox as toolbox
import preprocessing.snap_toolbox as snap_toolbox

from datetime import datetime
datetime_string = datetime.now().strftime("%m%d-%H%M")

if ENVIRONMENT == 'cpom':
    PATH_DATA = '/home/cb/sis2/data/'
elif ENVIRONMENT == 'local':
    PATH_DATA = '/Users/christianboehm/projects/sis2/data/'

TILESIZE = 256
# TILESIZE = 960

# img_pairs_inventory = pd.read_csv(os.path.join(PATH_DATA, 'inventory/img_pairs.csv'), index_col='index').loc[[80]]
img_pairs_inventory = pd.read_csv(os.path.join(PATH_DATA, 'inventory/img_pairs.csv'), index_col='index')

for index, row in img_pairs_inventory.iterrows():
    # if not (pd.isna(img_pairs_inventory['status'].iloc[index]) or img_pairs_inventory['status'].iloc[index] == 'new'):
    #     status = row['status']
    #     print(f'index {index} skipped due to status \'{status}\'')
    #     continue

    S2_FILE = row['s2']
    S3_FILE = row['s3']
    print(row['s2'])
    print(row['s3'])

    s2_raw = ProductIO.readProduct(S2_FILE)
    s3_raw = ProductIO.readProduct(S3_FILE)

    s2_bands = snap_toolbox.band_subset(s2_raw, 'B2,B3,B4,B_opaque_clouds')
    s3_bands = snap_toolbox.band_subset(s3_raw, 'Oa01_radiance,Oa02_radiance,Oa03_radiance,Oa04_radiance,Oa05_radiance,Oa06_radiance,Oa07_radiance,Oa08_radiance,Oa09_radiance,Oa10_radiance,Oa11_radiance,Oa12_radiance,Oa13_radiance,Oa14_radiance,Oa15_radiance,Oa16_radiance,Oa17_radiance,Oa18_radiance,Oa19_radiance,Oa20_radiance,Oa21_radiance')
    s2_bands = snap_toolbox.resample(s2_bands, 'B2')
    collocated = snap_toolbox.collocate(s2_bands, s3_bands)
    collocated = snap_toolbox.band_subset(collocated,'B2,B3,B4,Oa01_radiance,Oa02_radiance,Oa03_radiance,Oa04_radiance,Oa05_radiance,Oa06_radiance,Oa07_radiance,Oa08_radiance,Oa09_radiance,Oa10_radiance,Oa11_radiance,Oa12_radiance,Oa13_radiance,Oa14_radiance,Oa15_radiance,Oa16_radiance,Oa17_radiance,Oa18_radiance,Oa19_radiance,Oa20_radiance,Oa21_radiance,B_opaque_clouds,quality_flags,collocationFlags')

    s2_polygon = snap_toolbox.get_metadata_polygon(s2_raw, 's2')
    s3_polygon = snap_toolbox.get_metadata_polygon(s3_raw, 's3')

    tile_list, quality_list = snap_toolbox.cut_tiles(collocated,
                                                     tilesize=TILESIZE,
                                                     file_index=index,
                                                     output_path=PATH_DATA,
                                                     ensure_intersect_with=[s2_polygon,s3_polygon],
                                                     cloud_threshold=1.0)

    img_pairs_inventory.loc[index, 'status'] = 'tifs created'
    img_pairs_inventory.to_csv(os.path.join(PATH_DATA, f'inventory/img_pairs_{datetime_string}.csv'))

    s2_raw.dispose()
    s3_raw.dispose()
    s2_bands.dispose()
    s3_bands.dispose()
    collocated.dispose()

img_pairs_inventory.to_csv(os.path.join(PATH_DATA, f'inventory/img_pairs_{datetime_string}.csv'))
