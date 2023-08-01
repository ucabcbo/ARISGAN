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

    #TODO: Dispose S2/S3

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
    