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

Create a `environment.json` file in the project root folder and adjust it to your specific environment:

```json
{
    "environment": "cpom", # Environment name, for reference and logs
    "project_root": "/absolute/path/to/localrepo/",
    "data_root": "/absolute/path/to/data/",
    "experiment_root": "/absolute/path/to/experiments/",

    "s2_root": "/cpnet/projects/sikuttiaq/pond_inlet/Sentinel_2/DATA/", # Location of Sentinel-2 root data (if preprocessing is required)
    "s3_root": "/cpnet/projects/sikuttiaq/pond_inlet/Sentinel_3/OLCI/", # Location of Sentinel-2 root data (if preprocessing is required)

    "sample_freq": 1000, # every n-th training step, samples will be created
    "ckpt_freq": 10000, # every n-th training step, a checkpoint will be saved

    "max_shuffle_buffer": 500  # shuffle buffer size, depending on available memory

}
```

### Folder structure

Next to this repository, a directory for data and a directory for experiments are required.

#### Data

The data directory can be created anywhere in your file system. The following subdirectories are required (partially will be created
automatically):

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

Experiments get set up in the experiments root directory as json files. Create a meaningful subdirectory structure (no limitations) and store one
`experiment.json` file per leaf directory. An example subdirectory structure could be:

- model
  - lossfunction
    - dataset

The choice really depends on the best structure to keep an overview on models and their results. All output files will be stored in this directory.

The JSON contains all information needed to select model and dataset, various pre-processing parameters, loss functions etc. It also allows
using model-specific parameters. All dictionary entries of the `params` node get exposed to the model and thus can be used for model logic.

```json
{
    Model name, must be found as python module in `models` directory
    "model_name": "sis2",

    Dataset name, must be found as path in data root directory
    "dataset": "cur_masked",

    Tilesize (default: 256)
    "tilesize": 256,

    Image height/width (default: same as tilesize)
    "img_height": 256,
    "img_width": 256,

    Numer of training steps
    default: 40000
    "steps": 40000,

    Number of random sample images used for training/testing - null if the entire dataset shall be used (default: null)
    "sample_train": 10000,
    "sample_val": 1000,

    Batch size (default: 16)
    "batch_size": 16,

    Shuffle as bool (default: true)
    "shuffle": true,

    Random resize factor as float (default: 1.11)
    "random_resize": 1.2,

    Random rotate as bool (default: true)
    "random_rotate": true,

    Enforce that data filenames include a certain suffix (default: null)
    "enfore_suffix": "notinmask",

    Excelude data files with a certain suffix in the filename (default: null)
    "exclude_suffix": "notinmask",

    Dictionary of model-specific parameters
    "params": {
    },

    Generator loss functions to use and their weights (see `losses.py`)
    "gen_loss": {
        "gen_gan": null,
        "gen_nll": null,
        "gen_ssim": 50,
        "gen_l1": 50,
        "gen_l2": null,
        "gen_rmse": null,
        "gen_wstein": 100
    },

    Discriminator loss functions to use and their weights (see `losses.py`)
    "disc_loss": {
        "disc_bce": 1,
        "disc_nll": null
    }

}
```

# Training

To start training, simply run the `train.py` script.

The script takes the following parameters:

- `exp`: Link to the experiment subdirectory, e.g. `sis2/s3/l1loss`
- `timestamp` (optional, default: current time): Start timestamp to apply; for log purposes - supported format: `MMDD-hhmm`
- `restore` (optional, default: none): Timestamp of a checkpoint to restore and continue training

Training will run for as many steps as specified in the experiment. It will create the following subdirectories in the experiment directory:

#### `ckpt`

Checkpoint files. Checkpoints will be created every `n` steps, as specified in the `environment.json`.

Old checkpoint files will not be deleted - watch the size here manually.

A separate subdirectory will be created per `timestamp`, also when the training is continued.

#### `logs`

Log files, both Tensorflow logs, as well as an `experiment.json` output to verify parameters have been read correctly.

A separate subdirectory will be created per `timestamp`, also when the training is continued.

#### `model`

Graphical outputs of the generator and discriminator architecture.

#### `nohup`

Recommended subdirectory to store shell-level output files.

#### `samples`

A sample test image will be created every `n` steps, as specified in the `environment.json`. Those will be stored here.

In addition, the most current sample will be stored in the `_samples` subdirectory in the experiment root directory, for more convenient access.

# Evaluation

Evaluation will be performed automatically once training is complete. It can also be triggered manually by calling `evaluate.py`.

The script takes the following parameters:

- `exp`: Link to the experiment subdirectory, e.g. `sis2/s3/l1loss`
- `timestamp` (optional, default: latest experiment): Timestamp of an explicit experiment to restore

The script will evaluate the test dataset as specified in the respective `experiment.json` across all metrics implemented in `models.Metrics`.
It stores its results in the `_evaluation' subdirectory of the experiment root directory.

Results are stored on a per-sample csv file (according to the specified batch size).
`summary_eval.ipynb` shows some options to summarise and compare the evaluation results.
