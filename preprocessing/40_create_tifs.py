# This script creates tif tiles from the S2/S3 image pairs selected in the previous (manual) step
# It performs the task end-to-end by selecting the correct bands (B2-B4 from Sentinel-2, all 21 OLCI bands for Sentinel-3),
# collocating the images, thereby upsampling Sentinel-3 by nearest neighbour upsampling, cropping
# random tiles from the result, and saving them as tif.

# The resulting tif files have 26 channels: 0 and 1 are quality bands, 2-4 are the Sentinel-2 bands, the rest Sentinel-3 bands.

# Masks can be specified. Masks need to be stored as kml files (one polygon per file) in the data/_masks directory. Tif files are
# created regardless, but the ones outside the mask get the suffix "notinmask".

# The script also filters out those tiles that are out of the visible Sentinel-2/Sentinel-3 bounds, i.e. do not have an overlap
# with the geographic extent. The Sentinel metadata are used for this check.

# A plot displaying the overlay, including the mask(s), is saved as a png file.


### Arguments
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--selection", required=True, help='Selection code (without .json extension)')
parser.add_argument("--overlap", required=False, default=0.5, type=float, help='Minimum overlap to consider S2/S3 match')
parser.add_argument("--masks", required=True, help='Masks to apply (comma-separated), leave blank if none needed')
parser.add_argument("--quantity", required=False, default=5, type=int, help='Number of tiles to cut per S2 image')
parser.add_argument("--tilesize", required=False, default=256, type=int, help='Tilesize in pixel')

a = parser.parse_args()
print(f'Arguments read: {a}')
### End arguments


SELECTION = a.selection
OVERLAP = a.overlap
MASKS = a.masks
TILE_QUANTITY = a.quantity
TILESIZE = a.tilesize

import os
import sys
import json
import pandas as pd
import random

sys.path.append('.')
sys.path.append('..')
from environment import Environment
env = Environment()

import snap_toolbox as stbx
import toolbox as tbx

with open(os.path.join(env.DATA_ROOT, '_inventory', f'{SELECTION}.json'), "r") as json_file:
    selection = json.load(json_file)

invname = selection.get('inventory', None)
inventory = pd.read_csv(os.path.join(env.DATA_ROOT, '_inventory', f'{invname}.csv'), index_col='index')

s2_indices = selection.get('s2_indices', [])
masks = MASKS.split(',')

for s2_index in s2_indices:
    targetpairs = inventory[(inventory['s2'] == inventory['s2'][s2_index]) &
                        (inventory['overlap'] >= OVERLAP)].sort_values('timediff')
    targetpair = targetpairs.iloc[0]
    inv_index = targetpairs.index[0]


    s2_raw = stbx.read_product(targetpair['s2'])
    s3_raw = stbx.read_product(targetpair['s3'])

    s2_filename = os.path.basename(targetpair['s2'])
    s3_filename = os.path.basename(targetpair['s3'])

    s2_polygon = stbx.get_metadata_polygon(s2_raw, 's2')
    s3_polygon = stbx.get_metadata_polygon(s3_raw, 's3')

    maskpaths = [os.path.join(env.DATA_ROOT, '_masks', f'{maskname}.kml') for maskname in masks]
    polygons = [stbx.load_kml(maskpath) for maskpath in maskpaths]

    targetdir_png = os.path.join(env.DATA_ROOT, '_inventory', SELECTION)
    if not os.path.exists(targetdir_png):
        os.makedirs(targetdir_png)

    stbx.plot_polygons(s2_raw, s3_raw,
                    polygons=polygons,
                    polygon_labels=masks,
                    title=f'{invname}: {inv_index}\n{s2_filename}\n{s3_filename}',
                    savefig=os.path.join(targetdir_png, f'{inv_index}.png'),
                    show=False)

    s2_subset = stbx.band_subset(s2_raw, 'B2,B3,B4')
    s3_subset = stbx.band_subset(s3_raw, 'Oa01_radiance,Oa02_radiance,Oa03_radiance,Oa04_radiance,Oa05_radiance,Oa06_radiance,Oa07_radiance,Oa08_radiance,Oa09_radiance,Oa10_radiance,Oa11_radiance,Oa12_radiance,Oa13_radiance,Oa14_radiance,Oa15_radiance,Oa16_radiance,Oa17_radiance,Oa18_radiance,Oa19_radiance,Oa20_radiance,Oa21_radiance')

    collocated = stbx.collocate(s2_subset, s3_subset)
    collocated = stbx.band_subset(collocated,'B2,B3,B4,Oa01_radiance,Oa02_radiance,Oa03_radiance,Oa04_radiance,Oa05_radiance,Oa06_radiance,Oa07_radiance,Oa08_radiance,Oa09_radiance,Oa10_radiance,Oa11_radiance,Oa12_radiance,Oa13_radiance,Oa14_radiance,Oa15_radiance,Oa16_radiance,Oa17_radiance,Oa18_radiance,Oa19_radiance,Oa20_radiance,Oa21_radiance')

    targetdir_tif = os.path.join(env.DATA_ROOT, '_tif', invname, str(TILESIZE))
    if not os.path.exists(targetdir_tif):
        os.makedirs(targetdir_tif)

    maxstart_x = collocated.getSceneRasterWidth() - TILESIZE - 1
    maxstart_y = collocated.getSceneRasterHeight() - TILESIZE - 1

    for _ in range(TILE_QUANTITY):
        tile_x = random.randint(0, maxstart_x)
        tile_y = random.randint(0, maxstart_y)

        tilecode = f'{inv_index:05d}_{tile_x}x{tile_y}'

        try:

            region = f'{tile_x},{tile_y},{TILESIZE},{TILESIZE}'
            tile = stbx.region_subset(collocated, region)

            # stbx.plot_tile(tile, 's2')

            tile_polygon = stbx.get_bbox_polygon(tile)

            s2_intersect = tile_polygon.intersection(s2_polygon).area / tile_polygon.area
            if s2_intersect < 0.95:
                continue

            s3_intersect = tile_polygon.intersection(s3_polygon).area / tile_polygon.area
            if s2_intersect < 0.95:
                continue

            is_in_polygon = False
            for polygon in polygons:
                mask_intersect = tile_polygon.intersection(polygon).area / tile_polygon.area
                if mask_intersect >= 0.8:
                    is_in_polygon = True

            # stbx.plot_polygons(s2_raw, s3_raw, [tile], polygons, masks)

            targetfile = os.path.join(targetdir_tif, f'{tilecode}{"_notinmask" if not is_in_polygon else ""}.tif')

            stbx.save_geotiff(tile, targetfile)
            print(f'Saved {targetfile}')

        except Exception as e:
            print(f'Error creating tile {tilecode}: {e}')
    
    s2_raw.dispose()
    s3_raw.dispose()
    s2_subset.dispose()
    s3_subset.dispose()
    collocated.dispose()


tbx.send_email('40_create_tifs completed', f'Selection: {SELECTION}')
