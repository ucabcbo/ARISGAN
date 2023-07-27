
import os
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import random
from shapely.geometry import Polygon

import snappy
from snappy import ProductIO, PixelPos
from snappy import jpy
HashMap = snappy.jpy.get_type('java.util.HashMap')

import sys
sys.path.append('../')
import toolbox as tbx


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


def plot_tile(product, satellite, cloudmask=False, figsize=(10,10)):
    # Extract the bands for red, green, and blue

    if satellite == 's2':
        red_data = tbx.normalize_numpy(nparray(product, 'B4'))
        green_data = tbx.normalize_numpy(nparray(product, 'B3'))
        blue_data = tbx.normalize_numpy(nparray(product, 'B2'))
    elif satellite == 's3':
        red_data = tbx.normalize_numpy(nparray(product, 'Oa17_radiance'))
        green_data = tbx.normalize_numpy(nparray(product, 'Oa06_radiance'))
        blue_data = tbx.normalize_numpy(nparray(product, 'Oa03_radiance'))

    stacked_array = np.stack([red_data, green_data, blue_data], axis=2)

    plt.figure(figsize=figsize)
    plt.imshow(stacked_array)

    if cloudmask:
        try:
            cloud_data = ~(nparray(product, 'B_opaque_clouds').astype(int))
            plt.imshow(cloud_data, alpha=0.3, cmap=cm.gray)
        except:
            print('cloud band B_opaque_clouds not found')

    # fig.axes.get_xaxis().set_visible(False)
    # fig.axes.get_yaxis().set_visible(False)
    plt.show()


def cut_tiles(product, tilesize, file_index, output_path, save_if_errors:bool, ensure_intersect_with=[], intersect_threshold=0.95, cloud_threshold=1.0):
    tile_inventory = pd.DataFrame(columns=['pair_index', 'tile', 'size', 'status', 'clouds', 'intersect', 's3black', 'error', 'filename'])

    file_index = f'{file_index:05d}'

    tile_list = dict()
    quality_list = dict()

    # Number of tiles in y direction
    y_tiles = int(product.getSceneRasterHeight() / tilesize)
    # Starting pixel of each tile (i.e.: width from one tile to the next)
    y_step = int(product.getSceneRasterHeight() / y_tiles)
    # Starting offset to center the tiles on the image, i.e. half the distance between the individual tile gaps
    y_offset = int((product.getSceneRasterHeight() - ((y_tiles-1)*y_step+(tilesize-1))) / 2)

    # Number of tiles in x direction
    x_tiles = int(product.getSceneRasterHeight() / tilesize)
    # Starting pixel of each tile (i.e.: width from one tile to the next)
    x_step = int(product.getSceneRasterHeight() / x_tiles)
    # Starting offset to center the tiles on the image, i.e. half the distance between the individual tile gaps
    x_offset = int((product.getSceneRasterHeight() - ((x_tiles-1)*x_step+(tilesize-1))) / 2)


    for x in range(x_tiles):
        for y in range(y_tiles):
            TILE_XPOS = x_offset+x*x_step
            TILE_YPOS = y_offset+y*y_step
            TILECODE = f'{TILE_XPOS}x{TILE_YPOS}'
            output_tif = f'tif{tilesize}/{file_index}_{TILECODE}.tif'

            tile_inventory = tile_inventory.append({'pair_index': file_index,
                                                    'tile': TILECODE,
                                                    'size': tilesize,
                                                    'status': 'ok'}, ignore_index=True)
            tile_index = tile_inventory.index[-1]

            # if os.path.exists(os.path.join(PATH_DATA, output_filename)):
            #     tif_inventory = tif_inventory.append({'img_pair_id': TILE_PREFIX,
            #                             'tile': TILECODE,
            #                             'size': TILESIZE,
            #                             'tif_status': 'exists'}, ignore_index=True)
            #     print(f'File {output_filename} already exists')
            #     continue

            try:
                region = f'{TILE_XPOS},{TILE_YPOS},{tilesize},{tilesize}'
                tile = region_subset(product, region)

                if cloud_threshold < 1.0:
                    cloud_array = nparray(tile, 'B_opaque_clouds')
                    cloudpct = np.sum(cloud_array) / np.size(cloud_array)
                    tile_inventory.at[tile_index, 'clouds'] = f'{int(cloudpct * 100)}%'
                    if cloudpct > cloud_threshold:
                        tile_inventory.at[tile_index, 'status'] = 'nok'

                if len(ensure_intersect_with) > 0:
                    tile_polygon = get_bbox_polygon(tile)
                    intersections = []
                    for i in range(len(ensure_intersect_with)):
                        intersect = tile_polygon.intersection(ensure_intersect_with[i]).area / tile_polygon.area
                        if intersect < intersect_threshold:
                            tile_inventory.at[tile_index, 'status'] = 'nok'
                        intersections.append(f'[{i}]:{int(intersect * 100)}%')
                    tile_inventory.at[tile_index, 'intersect'] = '|'.join(intersections)
                        
                # black_array_s2 = nparray(tile, 'B2')
                # blackpct_s2 = np.count_nonzero(black_array_s2 == -0.1) / np.size(black_array_s2)
                # if blackpct_s2 > 0.05:
                #     status = 'quality'
                #     comment = f'S2 black: {int(blackpct_s2 * 100)}%'

                black_array_s3 = nparray(tile, 'Oa17_radiance')
                max_value = np.max(black_array_s3)
                blackpct_s3 = np.count_nonzero(black_array_s3 == max_value) / np.size(black_array_s3)
                if blackpct_s3 > 0.05:
                    tile_inventory.at[tile_index, 'status'] = 'nok'
                tile_inventory.at[tile_index, 's3black'] = f'{int(blackpct_s3 * 100)}%'

                if tile_inventory.at[tile_index, 'status'] == 'ok':
                    ProductIO.writeProduct(tile, os.path.join(output_path, output_tif), 'GeoTIFF')
                    tile_inventory.at[tile_index, 'filename'] = os.path.join(output_path, output_tif)
                elif save_if_errors:
                    ProductIO.writeProduct(tile, os.path.join(output_path, output_tif + 'x'), 'GeoTIFF')
                    tile_inventory.at[tile_index, 'filename'] = os.path.join(output_path, output_tif + 'x.tif')
                                
                tile.dispose()
                tile_inventory.to_csv(os.path.join(output_path, f'inventory/tiles/{file_index}_{tilesize}.csv'), index=False)

            except Exception as e:
                tile_inventory.at[tile_index, 'status'] = 'error'
                tile_inventory.at[tile_index, 'error'] = str(e)
                tile_inventory.to_csv(os.path.join(output_path, f'inventory/tiles/{file_index}_{tilesize}.csv'), index=False)
                continue
    
    tile_inventory.to_csv(os.path.join(output_path, f'inventory/tiles/{file_index}_{tilesize}.csv'), index=False)

    return tile_list, quality_list


def cut_random_tiles(product, tilesize, file_index, output_path, save_if_errors:bool, ensure_intersect_with=[], intersect_threshold=0.95, cloud_threshold=1.0, tile_ratio=0.75):
    tile_inventory = pd.DataFrame(columns=['pair_index', 'tile', 'size', 'status', 'clouds', 'intersect', 's3black', 'error', 'filename'])

    file_index = f'{file_index:05d}'

    tile_list = dict()
    quality_list = dict()

    # Number of tiles in y direction
    y_tiles = int(product.getSceneRasterHeight() / tilesize)
    y_maxstart = product.getSceneRasterHeight() - tilesize - 1

    # Number of tiles in x direction
    x_tiles = int(product.getSceneRasterHeight() / tilesize)
    x_maxstart = product.getSceneRasterHeight() - tilesize - 1

    tile_quantity = int(x_tiles * y_tiles * tile_ratio)

    for _ in range(tile_quantity):
        TILE_XPOS = random.randint(0, x_maxstart)
        TILE_YPOS = random.randint(0, y_maxstart)

        TILECODE = f'{TILE_XPOS}x{TILE_YPOS}'
        output_tif = f'tif{tilesize}/{file_index}_{TILECODE}.tif'

        tile_inventory = tile_inventory.append({'pair_index': file_index,
                                                'tile': TILECODE,
                                                'size': tilesize,
                                                'status': 'ok'}, ignore_index=True)
        tile_index = tile_inventory.index[-1]

        # if os.path.exists(os.path.join(PATH_DATA, output_filename)):
        #     tif_inventory = tif_inventory.append({'img_pair_id': TILE_PREFIX,
        #                             'tile': TILECODE,
        #                             'size': TILESIZE,
        #                             'tif_status': 'exists'}, ignore_index=True)
        #     print(f'File {output_filename} already exists')
        #     continue

        try:
            region = f'{TILE_XPOS},{TILE_YPOS},{tilesize},{tilesize}'
            tile = region_subset(product, region)

            if cloud_threshold < 1.0:
                cloud_array = nparray(tile, 'B_opaque_clouds')
                cloudpct = np.sum(cloud_array) / np.size(cloud_array)
                tile_inventory.at[tile_index, 'clouds'] = f'{int(cloudpct * 100)}%'
                if cloudpct > cloud_threshold:
                    tile_inventory.at[tile_index, 'status'] = 'nok'

            if len(ensure_intersect_with) > 0:
                tile_polygon = get_bbox_polygon(tile)
                intersections = []
                for i in range(len(ensure_intersect_with)):
                    intersect = tile_polygon.intersection(ensure_intersect_with[i]).area / tile_polygon.area
                    if intersect < intersect_threshold:
                        tile_inventory.at[tile_index, 'status'] = 'nok'
                    intersections.append(f'[{i}]:{int(intersect * 100)}%')
                tile_inventory.at[tile_index, 'intersect'] = '|'.join(intersections)
                    
            # black_array_s2 = nparray(tile, 'B2')
            # blackpct_s2 = np.count_nonzero(black_array_s2 == -0.1) / np.size(black_array_s2)
            # if blackpct_s2 > 0.05:
            #     status = 'quality'
            #     comment = f'S2 black: {int(blackpct_s2 * 100)}%'

            black_array_s3 = nparray(tile, 'Oa17_radiance')
            max_value = np.max(black_array_s3)
            blackpct_s3 = np.count_nonzero(black_array_s3 == max_value) / np.size(black_array_s3)
            if blackpct_s3 > 0.05:
                tile_inventory.at[tile_index, 'status'] = 'nok'
            tile_inventory.at[tile_index, 's3black'] = f'{int(blackpct_s3 * 100)}%'

            if tile_inventory.at[tile_index, 'status'] == 'ok':
                ProductIO.writeProduct(tile, os.path.join(output_path, output_tif), 'GeoTIFF')
                tile_inventory.at[tile_index, 'filename'] = os.path.join(output_path, output_tif)
            elif save_if_errors:
                ProductIO.writeProduct(tile, os.path.join(output_path, output_tif + 'x'), 'GeoTIFF')
                tile_inventory.at[tile_index, 'filename'] = os.path.join(output_path, output_tif + 'x.tif')
                            
            tile.dispose()
            tile_inventory.to_csv(os.path.join(output_path, f'inventory/tiles/{file_index}_{tilesize}.csv'), index=False)

        except Exception as e:
            tile_inventory.at[tile_index, 'status'] = 'error'
            tile_inventory.at[tile_index, 'error'] = str(e)
            tile_inventory.to_csv(os.path.join(output_path, f'inventory/tiles/{file_index}_{tilesize}.csv'), index=False)
            print(str(e))
            continue
    
    tile_inventory.to_csv(os.path.join(output_path, f'inventory/tiles/{file_index}_{tilesize}.csv'), index=False)

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


def check_overlap(master_product, slave_product, type_master='bbox', type_slave='bbox'):
    
    if type_master[:8] == 'metadata':
        master_polygon = get_metadata_polygon(master_product, type_master[-2:])
    else:
        master_polygon = get_bbox_polygon(master_product)

    if type_slave[:8] == 'metadata':
        slave_polygon = get_metadata_polygon(slave_product, type_slave[-2:])
    else:
        slave_polygon = get_bbox_polygon(slave_product)
    
    # Define overlap ratio as: master+slave intersection area in relation to master area
    intersect = master_polygon.intersection(slave_polygon).area / master_polygon.area

    return intersect


def get_bbox_polygon(product):
    
    # Initialize geocoding
    gc_a = product.getSceneGeoCoding()

    # Get geolocation of pixels at corner points
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


def get_metadata_polygon(product, satellite:str):
    metadata = product.getMetadataRoot()

    coordinates_list = None

    if satellite == 's2':
        coordinates_list = str(metadata.getElement('Level-1C_User_Product')
                            .getElement('Geometric_Info')
                            .getElement('Product_Footprint')
                            .getElement('Product_Footprint')
                            .getElement('Global_Footprint')
                            .getAttribute('EXT_POS_LIST')
                            .getData()
                            .getElemString()).split()
    elif satellite == 's3':
        coordinates_list = str(metadata.getElement('Manifest')
                            .getElement('metadataSection')
                            .getElement('frameSet')
                            .getElement('footPrint')
                            .getAttribute('posList')
                            .getData()
                            .getElemString()).split()

    if coordinates_list is None:
        return None
    
    coordinates = [(float(coordinates_list[i]), float(coordinates_list[i+1])) for i in range(0, len(coordinates_list), 2)]
    return Polygon(coordinates)


def plot_polygons(s2_raw, s3_raw, tiles=None, figsize=(10,10)):
    print(type(tiles))
    s2_bbox = get_bbox_polygon(s2_raw)
    s3_bbox = get_bbox_polygon(s3_raw)
    s2_metadata = get_metadata_polygon(s2_raw, 's2')
    s3_metadata = get_metadata_polygon(s3_raw, 's3')

    tile_polygons = []
    if type(tiles) is list:
        for tile in tiles:
            tile_polygons.append(get_bbox_polygon(tile))
    else:
        tile_polygons.append(get_bbox_polygon(tiles))

    fig, ax = plt.subplots(figsize=figsize)

    # Plot the polygons
    ax.plot(*s2_bbox.exterior.xy, color='palegreen', linestyle='dashed', linewidth=1, label='S2 bbox')
    ax.plot(*s3_bbox.exterior.xy, color='lightblue', linestyle='dashed', linewidth=1, label='S3 bbox')
    ax.plot(*s2_metadata.exterior.xy, color='green', label='S2 actual')
    ax.plot(*s3_metadata.exterior.xy, color='blue', label='S3 actual')
    for index in range(len(tile_polygons)):
        ax.plot(*tile_polygons[index].exterior.xy, color='red', linewidth=1, label=f'Tile {index}')

    # Set plot limits
    # ax.set_xlim(45, 90)
    # ax.set_ylim(-90, -45)
    # ax.set_xlim(70, 74)
    # ax.set_ylim(-82, -72)

    ax.legend()
    plt.show()
