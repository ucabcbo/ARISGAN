import sys
sys.path.append('..')
from environment import Environment

import os
import sys
import pandas as pd

sys.path.append(os.path.expanduser('~/.snap/snap-python'))
import snappy
from snappy import ProductIO
from snappy import jpy
HashMap = snappy.jpy.get_type('java.util.HashMap')

sys.path.append('../')
import toolbox as tbx
import preprocessing.snap_toolbox as stbx

env = Environment()

TILESIZE = 256

img_pairs_inventory = pd.read_csv(os.path.join(env.DATA_ROOT, 'inventory/img_pairs.csv'), index_col='index')

curated_pairs = [34,
                 41,
                 47,
                 77,
                 81,
                 89,
                 103,
                 126,
                 141,
                 149,
                 170,
                 178,
                 264]


# In reality, only executed for parts of image 77, as 15,000 images was reached after 10hrs.
perfect_pairs = [77,
                 149]


for i in range(len(curated_pairs)):
    img_index = curated_pairs[i]
    img_pair = img_pairs_inventory.iloc[img_index]

    s2_raw = ProductIO.readProduct(img_pair['s2'])
    s3_raw = ProductIO.readProduct(img_pair['s3'])

    s2_bands = stbx.band_subset(s2_raw, 'B2,B3,B4')
    s3_bands = stbx.band_subset(s3_raw, 'Oa01_radiance,Oa02_radiance,Oa03_radiance,Oa04_radiance,Oa05_radiance,Oa06_radiance,Oa07_radiance,Oa08_radiance,Oa09_radiance,Oa10_radiance,Oa11_radiance,Oa12_radiance,Oa13_radiance,Oa14_radiance,Oa15_radiance,Oa16_radiance,Oa17_radiance,Oa18_radiance,Oa19_radiance,Oa20_radiance,Oa21_radiance')
    collocated = stbx.collocate(s2_bands, s3_bands)
    collocated = stbx.band_subset(collocated,'B2,B3,B4,Oa01_radiance,Oa02_radiance,Oa03_radiance,Oa04_radiance,Oa05_radiance,Oa06_radiance,Oa07_radiance,Oa08_radiance,Oa09_radiance,Oa10_radiance,Oa11_radiance,Oa12_radiance,Oa13_radiance,Oa14_radiance,Oa15_radiance,Oa16_radiance,Oa17_radiance,Oa18_radiance,Oa19_radiance,Oa20_radiance,Oa21_radiance,quality_flags,collocationFlags')

    s2_polygon = stbx.get_metadata_polygon(s2_raw, 's2')
    s3_polygon = stbx.get_metadata_polygon(s3_raw, 's3')

    tile_list, quality_list = stbx.cut_random_tiles(collocated,
                                            tilesize=TILESIZE,
                                            file_index=img_index,
                                            output_path=env.DATA_ROOT,
                                            save_if_errors=False,
                                            ensure_intersect_with=[s2_polygon,s3_polygon],
                                            cloud_threshold=1.0)
