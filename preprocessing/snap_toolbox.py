
import os
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

import snappy
from snappy import ProductIO, PixelPos
from snappy import jpy
HashMap = snappy.jpy.get_type('java.util.HashMap')

import sys
sys.path.append('../')
import sis_toolbox as toolbox


def band_subset(raw, bands):
    parameters = HashMap()
    parameters.put('sourceBands',bands)
    result = snappy.GPF.createProduct('BandSelect', parameters, raw)
    # print(f'Subset created with {len(list(result.getBandNames()))} bands')
    return result


def region_subset(raw, region):
    parameters = HashMap()
    parameters.put('referenceBand','B2')
    parameters.put('region',region)
    parameters.put('subSamplingX','1')
    parameters.put('subSamplingY','1')
    parameters.put('fullSwath','false')
    parameters.put('copyMetadata','true')
    result = snappy.GPF.createProduct('Subset', parameters, raw)
    # print(f'Subset created, result has {len(list(result.getBandNames()))} bands')
    return result


def resample(raw, reference_band, upsamling_method='Nearest'):
    parameters = HashMap()
    parameters.put('referenceBand',reference_band)
    parameters.put('upsampling',upsamling_method)
    parameters.put('downsampling','First')
    parameters.put('flagDownsampling','First')
    parameters.put('resampleOnPyramidLevels','true')
    result = snappy.GPF.createProduct('Resample', parameters, raw)
    # print(f'Resampled, result has {len(list(result.getBandNames()))} bands')
    return result


def collocate(master, slave, resampling_method='NEAREST_NEIGHBOUR', rename=False):
    parameters = HashMap()
    parameters.put('masterProductName',master.getName())
    # parameters.put('targetProductName','_collocated')
    parameters.put('targetProductType','COLLOCATED')
    parameters.put('resamplingType',resampling_method)
    if rename:
        parameters.put('renameMasterComponents','true')
        parameters.put('renameSlaveComponents','true')
        parameters.put('masterComponentPattern','${ORIGINAL_NAME}_M')
        parameters.put('slaveComponentPattern','${ORIGINAL_NAME}_S${SLAVE_NUMBER_ID}')
    else:
        parameters.put('renameMasterComponents','false')
        parameters.put('renameSlaveComponents','false')
    result = snappy.GPF.createProduct('Collocate', parameters, [master, slave])
    # print(f'Collocated, result has {len(list(result.getBandNames()))} bands')
    return result


def nparray(product, bandname):
    band = product.getBand(bandname)
    width = band.getRasterWidth()
    height = band.getRasterHeight()
    nparray = np.zeros(width*height, dtype=np.float32)
    band.readPixels(0,0,width,height,nparray)
    nparray.shape = (height,width)
    return nparray


def plot_tile(product, cloudmask=False, figsize=(10,10)):
    # Extract the bands for red, green, and blue

    red_data = toolbox.normalize_numpy(nparray(product, 'B4'))
    green_data = toolbox.normalize_numpy(nparray(product, 'B3'))
    blue_data = toolbox.normalize_numpy(nparray(product, 'B2'))

    stacked_array = np.stack([red_data, green_data, blue_data], axis=2)

    plt.figure(figsize=figsize)
    plt.imshow(stacked_array)

    if cloudmask:
        cloud_data = ~(nparray(product, 'B_opaque_clouds').astype(int))
        plt.imshow(cloud_data, alpha=0.3, cmap=cm.gray)

    # fig.axes.get_xaxis().set_visible(False)
    # fig.axes.get_yaxis().set_visible(False)
    plt.show()

def plot_s3_tile(product, figsize=(10,10)):
    # Extract the bands for red, green, and blue

    red_data = toolbox.normalize_numpy(nparray(product, 'Oa17_radiance'))
    green_data = toolbox.normalize_numpy(nparray(product, 'Oa06_radiance'))
    blue_data = toolbox.normalize_numpy(nparray(product, 'Oa03_radiance'))

    stacked_array = np.stack([red_data, green_data, blue_data], axis=2)

    plt.figure(figsize=figsize)
    plt.imshow(stacked_array)

    # fig.axes.get_xaxis().set_visible(False)
    # fig.axes.get_yaxis().set_visible(False)
    plt.show()


def cut_tiles(product, TILESIZE, PAIR_INDEX, output_path):
    tile_inventory = pd.DataFrame(columns=['pair_index', 'tile', 'size', 'status', 'comment', 'filename'])
    tile_inventory = tile_inventory.astype({'pair_index': str,
                        'tile': str,
                        'size': int,
                        'status': str,
                        'comment': str,
                        'filename': str})

    tile_list = dict()
    quality_list = dict()

    # Number of tiles in y direction
    y_tiles = int(product.getSceneRasterHeight() / TILESIZE)
    # Starting pixel of each tile (i.e.: width from one tile to the next)
    y_step = int(product.getSceneRasterHeight() / y_tiles)
    # Starting offset to center the tiles on the image, i.e. half the distance between the individual tile gaps
    y_offset = int((product.getSceneRasterHeight() - ((y_tiles-1)*y_step+(TILESIZE-1))) / 2)

    # Number of tiles in x direction
    x_tiles = int(product.getSceneRasterHeight() / TILESIZE)
    # Starting pixel of each tile (i.e.: width from one tile to the next)
    x_step = int(product.getSceneRasterHeight() / x_tiles)
    # Starting offset to center the tiles on the image, i.e. half the distance between the individual tile gaps
    x_offset = int((product.getSceneRasterHeight() - ((x_tiles-1)*x_step+(TILESIZE-1))) / 2)

    TILE_PREFIX = f'{PAIR_INDEX:05d}'

    for x in range(x_tiles):
        for y in range(y_tiles):
            TILE_XPOS = x_offset+x*x_step
            TILE_YPOS = y_offset+y*y_step
            TILECODE = f'{TILE_XPOS}x{TILE_YPOS}'
            output_filename = f'tif{TILESIZE}/{TILE_PREFIX}_{TILECODE}.tif'
            status = 'ok'
            comment = ''

            # if os.path.exists(os.path.join(PATH_DATA, output_filename)):
            #     tif_inventory = tif_inventory.append({'img_pair_id': TILE_PREFIX,
            #                             'tile': TILECODE,
            #                             'size': TILESIZE,
            #                             'tif_status': 'exists'}, ignore_index=True)
            #     print(f'File {output_filename} already exists')
            #     continue


            try:
                region = f'{TILE_XPOS},{TILE_YPOS},{TILESIZE},{TILESIZE}'
                tile = region_subset(product, region)

                cloud_array = nparray(tile, 'B_opaque_clouds')
                cloudpct = np.sum(cloud_array) / np.size(cloud_array)
                if cloudpct > 0.1:
                    status = 'quality'
                    comment = f'cloud coverage: {int(cloudpct * 100)}%'

                black_array_s2 = nparray(tile, 'B2')
                blackpct_s2 = np.count_nonzero(black_array_s2 == -0.1) / np.size(black_array_s2)
                if blackpct_s2 > 0.05:
                    status = 'quality'
                    comment = f'S2 black: {int(blackpct_s2 * 100)}%'

                black_array_s3 = nparray(tile, 'Oa17_radiance')
                max_value = np.max(black_array_s3)
                blackpct_s3 = np.count_nonzero(black_array_s3 == max_value) / np.size(black_array_s3)
                if blackpct_s3 > 0.05:
                    status = 'quality'
                    comment = f'S3 (likely) black: {int(blackpct_s3 * 100)}%'

                if status == 'ok':
                    ProductIO.writeProduct(tile, os.path.join(output_path, output_filename), 'GeoTIFF')
                    tile_list[TILECODE] = tile
                elif status == 'quality':
                    quality_list[TILECODE] = tile
                
                tile_inventory = tile_inventory.append({'pair_index': TILE_PREFIX,
                                        'tile': TILECODE,
                                        'size': TILESIZE,
                                        'status': status,
                                        'comment': comment,
                                        'filename': output_filename}, ignore_index=True)
                
                tile.dispose()
                
            except Exception as e:
                tile_inventory = tile_inventory.append({'pair_index': TILE_PREFIX,
                                        'tile': TILECODE,
                                        'size': TILESIZE,
                                        'status': 'error',
                                        'comment': str(e)}, ignore_index=True)
                continue
    
    tile_inventory.to_csv(os.path.join(output_path, f'inventory/{TILE_PREFIX}_{TILESIZE}.csv'), index=False)

    return tile_list, quality_list


def s2_metadata_cloud_percentage(s2_product):
    metadata = s2_product.getMetadataRoot()
    cloud_pct = float(metadata.getElement('Granules')
                      .getElementAt(0)
                      .getElement('Quality_Indicators_Info')
                      .getElement('Image_Content_QI')
                      .getAttribute('CLOUDY_PIXEL_PERCENTAGE')
                      .getData().getElemString())
    return cloud_pct


def check_overlap(product_a, product_b):
    
    polygon_a = get_polygon(product_a)
    polygon_b = get_polygon(product_b)
    
    # Define overlap ratio as: S2/S3 intersection area in relation to S2 area
    intersect = polygon_a.intersection(
        polygon_b).area / polygon_a.area

    return intersect


def get_polygon(product):
    
    # Initialize geocoding
    gc_a = product.getSceneGeoCoding()

    # Get geolocation of pixel (0,0) and pixel (max,max)
    nw = gc_a.getGeoPos(PixelPos(0, 0), None)
    ne = gc_a.getGeoPos(PixelPos(product.getSceneRasterWidth(), 0), None)
    se = gc_a.getGeoPos(PixelPos(product.getSceneRasterWidth(), product.getSceneRasterHeight()), None)
    sw = gc_a.getGeoPos(PixelPos(0, product.getSceneRasterHeight()), None)

    # Create bbox
    bbox = [[nw.getLat(), nw.getLon()], [ne.getLat(), ne.getLon()],
                [se.getLat(), se.getLon()], [sw.getLat(), sw.getLon()]]
        
    # Define shapely Polygon based on the bboxes
    polygon = Polygon(bbox)

    return polygon
