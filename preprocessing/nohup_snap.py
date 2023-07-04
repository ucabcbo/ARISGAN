import sys
sys.path.append('/home/cb/sis2/')

with open('../env.txt') as f:
    ENVIRONMENT = f.readlines()[0][:-1]
print(f'running on environment: "{ENVIRONMENT}"')
assert ENVIRONMENT in ['blaze',
                       'colab',
                       'local',
                       'cpom']

if ENVIRONMENT == 'cpom':
    sys.path.append('/home/cb/.snap/snap-python')
elif ENVIRONMENT == 'local':
    sys.path.append('/Users/christianboehm/.snap/snap-python')
else:
    print("snappy is only available on CPOM and possibly local machines")

import os
import datetime
import numpy as np
from matplotlib import pyplot as plt
import snappy
from snappy import ProductIO
from snappy import jpy
HashMap = snappy.jpy.get_type('java.util.HashMap')

import sis_helper as helper
import pandas as pd

if ENVIRONMENT == 'cpom':
    PATH_DATA = '/home/cb/sis2/data/'
elif ENVIRONMENT == 'local':
    PATH_DATA = '/Users/christianboehm/projects/sis2/data/'

TILESIZE = 256

img_pairs_inventory = pd.read_csv(os.path.join(PATH_DATA, 'inventory/img_pairs.csv'))

if not f'tif_{TILESIZE}' in img_pairs_inventory.columns:
    img_pairs_inventory[f'tif_{TILESIZE}'] = 'new'

worklist = img_pairs_inventory[img_pairs_inventory[f'tif_{TILESIZE}'] == 'new']

def get_collocated_image(S2_FILE, S3_FILE):
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    print(f'{timestamp} Collocating:')
    print(f'\tS2: {S2_FILE}')
    print(f'\tS3: {S3_FILE}')

    # Reading S2
    s2_raw = ProductIO.readProduct(S2_FILE)
    # len(list(s2_raw.getBandNames()))

    # Selecting subset bands from S2
    parameters = HashMap()
    parameters.put('sourceBands','B2,B3,B4')
    s2_bands = snappy.GPF.createProduct('BandSelect', parameters, s2_raw)
    # len(list(s2_bands.getBandNames()))

    # Reading S3
    s3_raw = ProductIO.readProduct(S3_FILE)
    # len(list(s3_raw.getBandNames()))

    # Selecting subset bands from S3
    parameters = HashMap()
    parameters.put('sourceBands','Oa01_radiance,Oa02_radiance,Oa03_radiance,Oa04_radiance,Oa05_radiance,Oa06_radiance,Oa07_radiance,Oa08_radiance,Oa09_radiance,Oa10_radiance,Oa11_radiance,Oa12_radiance,Oa13_radiance,Oa14_radiance,Oa15_radiance,Oa16_radiance,Oa17_radiance,Oa18_radiance,Oa19_radiance,Oa20_radiance,Oa21_radiance')
    s3_bands = snappy.GPF.createProduct('BandSelect', parameters, s3_raw)
    # len(list(s3_bands.getBandNames()))

    # Collocating the images
    parameters = HashMap()
    parameters.put('masterProductName',s2_bands.getName())
    # parameters.put('targetProductName','_collocated')
    parameters.put('targetProductType','COLLOCATED')
    parameters.put('renameMasterComponents','false')
    parameters.put('renameSlaveComponents','false')
    parameters.put('resamplingType','NEAREST_NEIGHBOUR')
    # parameters.put('masterComponentPattern','${ORIGINAL_NAME}_M')
    # parameters.put('slaveComponentPattern','${ORIGINAL_NAME}_S${SLAVE_NUMBER_ID}')
    collocated = snappy.GPF.createProduct('Collocate', parameters, [s2_bands, s3_bands])
    # len(collocated.getBandNames())

    # Subsetting again to relevant bands only (quality bands and flags being removed, except S3 quality flag and collocation flag)
    parameters = HashMap()
    parameters.put('sourceBands','B2,B3,B4,Oa01_radiance,Oa02_radiance,Oa03_radiance,Oa04_radiance,Oa05_radiance,Oa06_radiance,Oa07_radiance,Oa08_radiance,Oa09_radiance,Oa10_radiance,Oa11_radiance,Oa12_radiance,Oa13_radiance,Oa14_radiance,Oa15_radiance,Oa16_radiance,Oa17_radiance,Oa18_radiance,Oa19_radiance,Oa20_radiance,Oa21_radiance,quality_flags,collocationFlags')
    collocated_bands = snappy.GPF.createProduct('BandSelect', parameters, collocated)
    # len(list(collocated_bands.getBandNames()))

    print('Finished')

    return collocated_bands


# Open existing or create new dataframe to store tile-level results
if os.path.exists(os.path.join(PATH_DATA, f'inventory/{TILESIZE}.csv')):
    tif_inventory = pd.read_csv(os.path.join(PATH_DATA, f'inventory/{TILESIZE}.csv'))
else:
    tif_inventory = pd.DataFrame(columns=['img_pair_id', 'tile', 'tif_status', 'tfrecord_status'])
    tif_inventory = tif_inventory.astype({'img_pair_id': str,
                        'tile': str,
                        'tif_status': str,
                        'tfrecord_status': str})


for index, row in worklist.iterrows():

    collocated = get_collocated_image(row['s2'], row['s3'])

    # Number of tiles in y direction
    y_tiles = int(collocated.getSceneRasterHeight() / TILESIZE)
    # Starting pixel of each tile (i.e.: width from one tile to the next)
    y_step = int(collocated.getSceneRasterHeight() / y_tiles)
    # Starting offset to center the tiles on the image, i.e. half the distance between the individual tile gaps
    y_offset = int((collocated.getSceneRasterHeight() - ((y_tiles-1)*y_step+(TILESIZE-1))) / 2)

    # Number of tiles in x direction
    x_tiles = int(collocated.getSceneRasterHeight() / TILESIZE)
    # Starting pixel of each tile (i.e.: width from one tile to the next)
    x_step = int(collocated.getSceneRasterHeight() / x_tiles)
    # Starting offset to center the tiles on the image, i.e. half the distance between the individual tile gaps
    x_offset = int((collocated.getSceneRasterHeight() - ((x_tiles-1)*x_step+(TILESIZE-1))) / 2)

    num = row['img_pair_id']
    TILE_PREFIX = f'{num:05d}'

    for x in range(x_tiles):
        for y in range(y_tiles):
            try:
                output_filename = f'tif{TILESIZE}/{TILE_PREFIX}_{x_offset+x*x_step}x{y_offset+y*y_step}.tif'
                if os.path.exists(os.path.join(PATH_DATA, output_filename)):
                    tif_inventory = tif_inventory.append({'img_pair_id': TILE_PREFIX,
                                            'tile': output_filename,
                                            'tif_status': 'exists'}, ignore_index=True)
                    print(f'File {output_filename} already exists')
                    continue

                region = f'{x_offset+x*x_step},{y_offset+y*y_step},{TILESIZE},{TILESIZE}'

                parameters = HashMap()
                parameters.put('referenceBand','B2')
                parameters.put('region',region)
                parameters.put('subSamplingX','1')
                parameters.put('subSamplingY','1')
                parameters.put('fullSwath','false')
                parameters.put('copyMetadata','false')

                cropped = snappy.GPF.createProduct('Subset', parameters, collocated)

                ProductIO.writeProduct(cropped, os.path.join(PATH_DATA, output_filename), 'GeoTIFF')
                tif_inventory = tif_inventory.append({'img_pair_id': TILE_PREFIX,
                                        'tile': output_filename,
                                        'tif_status': 'created'}, ignore_index=True)
                # print(f'product written: {output_filename}')

            except Exception as e:
                tif_inventory = tif_inventory.append({'img_pair_id': TILE_PREFIX,
                                        'tile': output_filename,
                                        'tif_status': str(e)}, ignore_index=True)
                print(f'ERROR IN TILE {x_offset+x*x_step}x{y_offset+y*y_step}')
                print(str(e))
        
        tif_inventory.to_csv(os.path.join(PATH_DATA, f'inventory/{TILESIZE}.csv'), index=False)

    img_pairs_inventory.at[index, f'tif_{TILESIZE}'] = 'created'
    img_pairs_inventory.to_csv(os.path.join(PATH_DATA, 'inventory/img_pairs.csv'), index=False)

