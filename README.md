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

## Data Preprocessing

This repository contains all code necessary to preprocess training and test data from native Sentinel-2 `.SAVE` and Sentinel-3 `.SEN3` files.
However, the code to read Sentinel-2 and -3 files is proprietary to the directory structure used for the preparation of this code, which is
a flat directory for Sentinel-2, and a YYYY/MM subdirectory structure for Sentinel-3.

Data Preparation is performed in the following five steps. All files are located in the `preprocessing` directory. For all but step 50, the `snappy` module
is required. For step 50, `TensorFlow` is required instead.

See https://senbox.atlassian.net/wiki/spaces/SNAP/pages/24051781/Using+SNAP+in+your+Python+programs
for `snappy` installation (this seems to be much easier from SNAP 10 onward than it was up to SNAP 9).

#### `10_identify_unique_tilecodes.ipynb`

This code simply browses the Sentinel-2 directory and compiles a list of available tile codes with the number of files of each code. This is just used
to provide an overview.

#### `20_create_inventory.py`

This script reads all Sentinel-2 files with a given tile code and identifies all Sentinel-3 files
taken within a given time window of the respective Sentinel-2 file.

Tilecode and time window are provided as parameters.

It reads and lists both `.zip` and `.SEN3` files.
It attempts to unpack zip files but is error tolerant if it can't (e.g. due to missing write permissions)
For `.SEN3` files, it reads the file and calculates the overlap with the Sentinel-2 file.

It outputs the result in an inventory file in the `_inventory` subfolder of the data root directory, following
the naming convention `inv_<tile-code>_<time-window>.csv`

There is no need to open this file.

#### `30_create_s2pngs.py`

For each valid Sentinel-2 file in the inventory,
this script creates a downscaled png image to enable an easy manual review.
The purpose is that the user can judge which Sentinel-2 files should be further considered,
e.g., cloudy or distorted images can be excluded.

Inventory has to be specified. Overlap ratio and downsample factor can be provided as additional parameters.

The `inventory` parameter refers to the inventory code generated in step 20 (the filename without `.csv` extension).

The png files get saved in a directory named identically to the inventory. The individual png filenames represent
the IDs of the respective tile in the `.csv` file.

The user shall now review the PNGs and select the ones to include for further processing. To do so, create a
json file in the `_inventory` subdirectory. Include one property `inventory`, referring to the filename
of the respective inventory `.csv` file, and include a list of 's2_indices` to include. That is: the IDs of
good-quality Sentinel-2 images.

```json
{
    "inventory": "inv_T17XNA_10h",
    "s2_indices": [
        17,
        51,
        52,
        ...
    ]
}
```

#### `40_create_tifs.py`

This script creates tif tiles from the Sentinel-2/-3 image pairs selected by the user. For this purpose, it reads the
json file created in the previous step and looks up the specified IDs. In case the Sentinel-2 tile has multiple
possible Sentinel-3 counterparts, it chooses the one in closest temporal proximity.

Parameters are:

- `selection`: the name of the previously created json file, without `.json' extension
- `masks`: the filenames of the masks to apply, comma-separated if multiple, or an artificial term if none
- `tilesize` (optional): the size of a tif file in pixels, default is 256
- `quantity` (optional): the number of tif files to randomly generate, default is 5
- `overlap` (optional): the minimum Sentinel-2/-3 overlap ratio to consider, default is 0.5

It then performs the task end-to-end by selecting the correct bands (B2-B4 from Sentinel-2, all 21 OLCI bands for Sentinel-3),
collocating the images, thereby upsampling Sentinel-3 by nearest neighbour upsampling, cropping
random tiles from the result, and saving them as tif.

The resulting tif files have 26 channels: 0 and 1 are quality bands, 2-4 are the Sentinel-2 bands, and the rest are Sentinel-3 bands.

Masks can be specified. Masks must be stored as kml files (one polygon per file) in the data/_masks directory. Tif files are
created regardless, but the ones outside the mask get the suffix "notinmask".

The script also filters out those tiles that are out of the visible Sentinel-2/Sentinel-3 bounds, i.e. do not have an overlap
with the geographic extent. The Sentinel metadata are used for this check.

The tif files will be saved in a subdirectory named identical to the inventory name, in the `_tif` subdirectory of the data root folder.
An additional subdirectory corresponding to the `tilesize` will be created underneath.

The filenames follow the naming convention `<Sentinel-2-ID>_<start-x-pixel>x<start-y-pixel><suffix>.tif`.

In addition, a plot displaying the overlay, including the mask(s), will be saved for each Sentinel-2 ID in a subdirectory
corresponding to the selection filename, with a filename corresponding to the Sentinel-2 ID, in the `_inventory` subdirectory of the data root
directory.

Runtime for this script can easily be multiple hours.

#### `50_create_tfrecords.py`

This script converts tif files into `tfrecords`, for easy consumption by `tensorflow`.

It converts all tif files in the provided `_tif` subfolder. In its current setting, it automatically
creates various datasets, some in which downsampled versions of the Sentinel-2 image are used as input data, and one in
which the Sentinel-3 image is used as input data. In all cases, the original Sentinel-2 image is used
as ground truth.

It randomly assigns files to the training vs. test dataset - creating the different datasets simultaneously
ensures that identical files are used for training/testing across datasets.

Parameters are:

- `inventory`: the name of the input inventory, i.e. subdirectory in the `_tif` directory
- `dataset`: the name of the target dataset, i.e. output directory
- `tilesize` (optional): the size of a tif file in pixels, default is 256
- `train_ratio` (optional): the ratio to split between train and test datasets, default is 0.95

The resulting `tfrecords` will be saved in a subdirectory named after the `dataset` specified, directly in the data root directory.
The `tilesize` will be created as a subdirectory thereof.

Note that as of this step, files originating from multiple inventories will get combined in the same directory for the first time,
and tracing back to their origin is not possible anymore, other than through their filename and some forensic motivation.

The `tfrecords` are named identically to the tif files.

## Model setup

To setup a model for training, create a module in the `models` subfolder in the project directory - just copy any of the examples.

The module must contain a class `GAN` (no inheritance needed) with functions `Generator` and `Discriminator`, as well as `tf.function`-decorated `train_step`. A simple blueprint is shown below. The imports include:

- `layers`: includes pre-defined tf layers for easier architecture construction
- `losses`: contains predefined loss function methods for selection per experiment
- `Experiment`: represents an experiment (see below section)

```python
import tensorflow as tf

sys.path.append(os.getcwd())
import models.layers as layers
import models.losses as losses
from experiment import Experiment

class GAN:
    
    def __init__(self, experiment:Experiment):
        self.exp = experiment
        self.generator = self.Generator()
        self.discriminator = self.Discriminator()
        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.summary_writer = tf.summary.create_file_writer(self.exp.output.LOGS)
        

    def Generator(self) -> tf.keras.Model:
        inputs = tf.keras.layers.Input(shape=[self.exp.IMG_HEIGHT, self.exp.IMG_WIDTH, self.exp.INPUT_CHANNELS])
        ... # Insert layers to translate input to output `outputlayer`
        return tf.keras.Model(inputs=inputs, outputs=outputlayer)


    def Discriminator(self) -> tf.keras.Model:
        inp = tf.keras.layers.Input(shape=[self.exp.IMG_HEIGHT, self.exp.IMG_WIDTH, self.exp.INPUT_CHANNELS], name='input_image')
        tar = tf.keras.layers.Input(shape=[self.exp.IMG_HEIGHT, self.exp.IMG_WIDTH, self.exp.OUTPUT_CHANNELS], name='target_image')
        ... # Insert layers to translate inp/tar to output `outputlayer`
        return tf.keras.Model(inputs=[inp, tar], outputs=outputlayer)
    

    @tf.function
    def train_step(self, input_image, target, step):
        generator = self.generator
        discriminator = self.discriminator
        summary_writer = self.summary_writer

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = generator(input_image, training=True)
            disc_real_output = discriminator([input_image, target], training=True)
            disc_generated_output = discriminator([input_image, gen_output], training=True)
            total_gen_loss, gen_losses = losses.generator_loss(disc_generated_output, gen_output, target, self.exp.GEN_LOSS, self.loss_object)
            total_disc_loss, disc_losses = losses.discriminator_loss(disc_real_output, disc_generated_output, self.exp.DISC_LOSS, self.loss_object)

        generator_gradients = gen_tape.gradient(total_gen_loss, generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(total_disc_loss, discriminator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

        with summary_writer.as_default():
            tf.summary.scalar('total_gen_loss', total_gen_loss, step=step//1000)
            for gen_loss in list(gen_losses.keys()):
                tf.summary.scalar(gen_loss, gen_losses[gen_loss], step=step//1000)
            tf.summary.scalar('total_disc_loss', total_disc_loss, step=step//1000)
            for disc_loss in list(disc_losses.keys()):
                tf.summary.scalar(disc_loss, disc_losses[disc_loss], step=step//1000)
```

## Experiment setup

