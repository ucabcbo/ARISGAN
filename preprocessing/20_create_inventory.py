# This script reads all Sentinel-2 files with a given Tilecode, and identifies all Sentinel-3 files
# taken within a given time window of the respective Sentinel-2 file.

# It reads and lists both .zip and .SEN3 files.
# For .SEN3 files, it reads the file and calculates the overlap with the Sentinel-2 file.

# It outputs the result in an inventory file in the _inventory subfolder of the data root directory.


### Arguments
import argparse
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--tile", required=True, help='S2 tilecode, e.g. T17XPA, T17XNA etc.')
parser.add_argument("--timediff", required=True, type=float, help='Maximum time difference between satellite images in hours')

a = parser.parse_args()
print(f'Arguments read: {a}')
### End arguments

import os
import sys
from datetime import datetime
import glob
from datetime import datetime
from datetime import timedelta
import pandas as pd
import zipfile

sys.path.append('.')
sys.path.append('..')
from environment import Environment
env = Environment()

import snap_toolbox as stbx

TILE = a.tile
MAX_TIMEDIFF = a.timediff
PATH_S2 = env.S2_ROOT
PATH_S3 = env.S3_ROOT

if not os.path.exists(os.path.join(env.DATA_ROOT, '_inventory')):
    os.makedirs(os.path.join(env.DATA_ROOT, '_inventory'))

inventory = pd.DataFrame(columns=['year','month','s2','s2date','s3','s3date','timediff','s3filetype','overlap'])
output_filename = f'inv_{TILE}_{str(MAX_TIMEDIFF).replace(".","")}h.csv'

for yearnum in range(2017, 2024):
    year = str(yearnum)
    for monthnum in range(12):
        month = str(monthnum + 1).zfill(2)

        # Construct the file pattern with variables using string formatting
        file_pattern_S2 = os.path.join(PATH_S2, f'*{TILE}*{year}{month}*')
        file_paths_S2 = glob.glob(file_pattern_S2)

        # Construct the file pattern with variables using string formatting
        file_pattern_S3zip = os.path.join(PATH_S3, year, month, 'S3A_OL_1_EFR____*.zip')
        file_paths_S3zip = glob.glob(file_pattern_S3zip)

        file_pattern_S3 = os.path.join(PATH_S3, year, month, 'S3A_OL_1_EFR____*.SEN3')

        # Loop through the matched files
        for file_path_S2 in file_paths_S2:
            filename_s2 = os.path.basename(file_path_S2)
            # print(filename_s2)
            acquisition_time_S2 = datetime.strptime(filename_s2.split("_")[2], "%Y%m%dT%H%M%S")
            s2_raw = None

            not_unpacked_zips = []

            # Loop through S3 zip files to unzip the relevant ones
            for file_path_S3zip in file_paths_S3zip:
                filename_S3zip = os.path.basename(file_path_S3zip)
                acquisition_time_S3zip = datetime.strptime(filename_S3zip.split("_")[7][:15], "%Y%m%dT%H%M%S")
                time_diff = abs(acquisition_time_S3zip - acquisition_time_S2)
                if time_diff < timedelta(hours=MAX_TIMEDIFF):
                    if not file_path_S3zip[:-4] in file_paths_S3:
                        try:
                            with zipfile.ZipFile(file_path_S3zip, 'r') as zip_ref:
                                for file_info in zip_ref.infolist():
                                    if file_info.filename.startswith(filename_S3zip[:-4]):
                                        zip_ref.extract(file_info, os.path.dirname(file_path_S3zip))
                            print(f'Unpacked {filename_S3zip}')
                        except PermissionError as e:
                            print(f'PermissionEror: {os.path.dirname(file_path_S3zip)}')
                            print(f'when trying to unpack {filename_S3zip}')
                            not_unpacked_zips.append(file_path_S3zip)
                        except zipfile.BadZipFile as e:
                            print(f'zipfile.BadZipFile')
                            print(f'when trying to unpack {filename_S3zip}')
                            not_unpacked_zips.append(file_path_S3zip)
                        except Exception as e:
                            print(f'Uncaught Exception')
                            print(f'when trying to unpack {filename_S3zip}')
                            not_unpacked_zips.append(file_path_S3zip)
            
            # Now loop through S3 .SEN3 files to determine inventory
            file_paths_S3 = glob.glob(file_pattern_S3) + not_unpacked_zips

            for file_path_S3 in file_paths_S3:
                filename_S3 = os.path.basename(file_path_S3)
                extension_S3 = os.path.splitext(file_path_S3)[1]
                acquisition_time_S3 = datetime.strptime(filename_S3.split("_")[7][:15], "%Y%m%dT%H%M%S")
                time_diff = abs(acquisition_time_S3 - acquisition_time_S2)
                if time_diff < timedelta(hours=MAX_TIMEDIFF):
                    overlap = None
                    if extension_S3 == '.SEN3':
                        if s2_raw is None:
                            s2_raw = stbx.read_product(file_path_S2)
                        s3_raw = stbx.read_product(file_path_S3)
                        #TODO: check why S3 is sometimes None
                        if s3_raw is not None:
                            overlap = stbx.check_overlap(s2_raw, s3_raw, 'metadata.s2', 'metadata.s3')
                            s3_raw.dispose()
                        else:
                            print(f's3_raw is none: {file_path_S3}')
                    inventory.loc[len(inventory)] = {'year': year,
                                                       'month': month,
                                                       's2': file_path_S2,
                                                       's2date': acquisition_time_S2,
                                                       's3': file_path_S3,
                                                       's3date': acquisition_time_S3,
                                                       'timediff': time_diff,
                                                       's3filetype': extension_S3,
                                                       'overlap': overlap}

            if s2_raw is not None:
                s2_raw.dispose()

        inventory.to_csv(os.path.join(env.DATA_ROOT, '_inventory', output_filename), index=True, index_label='index')
