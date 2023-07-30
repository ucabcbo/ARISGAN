### Arguments
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--inventory", required=True, help='Inventory code (without .csv extension)')
parser.add_argument("--overlap", required=False, default=0.5, type=float, help='Minimum overlap to consider S2/S3 match')
parser.add_argument("--downsample", required=False, default=20, type=int, help='Downsample pngs for smaller file size (to every nth pixel)')

a = parser.parse_args()
print(f'Arguments read: {a}')
### End arguments

INVENTORY = a.inventory
OVERLAP = a.overlap
DOWNSAMPLE = a.downsample

import os
import sys
import json
import pandas as pd

sys.path.append('.')
sys.path.append('..')
from environment import Environment
env = Environment()

import snap_toolbox as stbx

inventory = pd.read_csv(os.path.join(env.DATA_ROOT, '_inventory', f'{INVENTORY}.csv'), index_col='index')
s2_inventory = inventory[inventory['overlap'] >= OVERLAP]['s2'].drop_duplicates()

targetdir = os.path.join(env.DATA_ROOT, '_inventory', INVENTORY)
if not os.path.exists(targetdir):
    os.makedirs(targetdir)

for index, s2_file in s2_inventory.items():
    s2_raw = stbx.read_product(str(s2_file))
    saveas = os.path.join(targetdir,f'{index}.png')
    title = f'{index}: {inventory["year"][index]}-{inventory["month"][index]}'
    stbx.plot_tile(s2_raw, 's2', show=False, savefig=saveas, downsample=DOWNSAMPLE, title=title, subtitle=os.path.basename(s2_file))
    print(f'ID {index} plotted')

config = {'inventory': INVENTORY,
          'overlap': OVERLAP,
          'downsample': DOWNSAMPLE}

with open(os.path.join(targetdir, '_config.json'), "w") as json_file:
    json.dump(config, json_file, indent=4)
    