# ARISGAN
ARISGAN generates synthetic high-resolution satellite imagery from low-resolution inputs.

It has been developed and tested with 256x256 pixel Sentinel-2 images as ground truth, and either a collocated Sentinel-3 image,
or 12x downsampled versions of the Sentinel-2 image as input data.

ARISGAN showed superior results compared to state-of-the-art models in multiple metrics, e.g., Haar wavelet-based Perceptual Similarity Index (HPSI),
Enhanced Global Relative Error in Synthesis (ERGAS) and Spatial Correlation Coefficient (SCC).

The intention of the model is a realistic image that could be the ground truth. An exact recreation is not aspired or possible due to the extreme
resolution difference.

Sample results:
![image](https://github.com/ucabcbo/ARISGAN/assets/115581327/e44be4c0-a635-4a0b-9e5c-fc242075ec0a)

# Setup

## Preparation

### Environment

Adjust `environment.json` to your specific environment: name, and reference to directory structures

### Folder structure

Next to this repository, a directory for data and a directory for experiments are required.

#### Data

The data directory can be created anywhere in your file system. The following subdirectories are required (partially will be created
automatically:

- `_inventory`: This directory will contain all interim files produced during pre-processing, enabling tracing the steps performed, as well as performing a manual review and selection of images to be taken up into a dataset
- `_masks`: If masks shall be applied, they are to be stored here in kml format
- `_tif`: This folder will contain the output tif images after preprocessing step 40
- \<dataset(s)\>: After preprocessing step 50, the resulting `tfrecord` files will be stored in the dataset name as specified by the user, and from here can be consumed for training and evaluation

#### Experiments

The experiments directory can be created anywhere in your file system. It is up to the user to create a multi-level folder structure underneath to organize
experiments. Each leaf directory can contain one `experiment.json' file, which describes the experiment to be conducted. During training, further directories
will automatically be created at this location.

Further directories in the experiment root directory are:

- `_samples`: The latest sample image for each model will be stored centrally here, along with the respective experiment's subdirectory for easier access
- `_evaluation`: The final evaluation results for each experiment will be stored here

## Data preparation

This repository contains all code necessary to preprocess training and test data from native Sentinel-2 `.SAVE` and Sentinel-3 `.SEN3` files.
However, the code to read Sentinel-2 and -3 files is proprietary to the directory structure used for the preparation of this code, which is
a flat directory for Sentinel-2, and a YYYY\MM subdirectory structure for Sentinel-3.

Data Preparation is performed in the following five steps, all files are location in the `preprocessing` directory. For all but step 50, `snappy` is required.
For step 50, `TensorFlow` is required instead.

### `10_identify_unique_tilecodes.ipynb`

...

### `20_create_inventory.py`

...

### `30_create_s2pngs.py`

...

### `40_create_tifs.py`

...

### `50_create_tfrecords.py`

...
