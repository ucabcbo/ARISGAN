import os
import json
import sys

class Experiment:
    """This class provides experiment information and settings, 
    as specified in the `experiment.json` file in the
    provided subdirectory of the experiment root directory.
    A documentation of this file is provided at the end.
    """

    class Output:
        """Subclass to identify, create and provide access to output
        directories for\n
        logs: `LOGS`\n
        checkpoints: `CKPT`\n
        samples: `SAMPLES`\n
        model information: `MODEL`
        """
        def __init__(self, output_root:str, fullname:str, timestamp:str):
            self.LOGS = os.path.join(output_root, f'logs/{fullname}_{timestamp}/')
            self.CKPT = os.path.join(output_root, f'ckpt/{timestamp}')
            #TODO: should I delete all old samples first?
            self.SAMPLES = os.path.join(output_root, f'samples/')
            self.MODEL = os.path.join(output_root, f'model/')
            os.makedirs(self.LOGS, exist_ok=True)
            os.makedirs(self.CKPT, exist_ok=True)
            os.makedirs(self.SAMPLES, exist_ok=True)
            os.makedirs(self.MODEL, exist_ok=True)


    def __init__(self, path:str, experiment_name:str, timestamp:str, restore:bool=False):
        """Creates a new Experiment instance by reading the respective experiment.json,
        exposing its parameters, and preparing output directory structure.

        Args:
            path (str): Path to experiment root directory (from environment.json)
            experiment_name (str): Experiment name as sub-directories
            timestamp (str): Timestamp, typically `MMDD-hhmm`
            restore (bool, optional): Whether an existing experiment shall be restored. Defaults to False, which will create a new experiment.
        """        

        self.STRINGNAME = experiment_name.replace('/', '_')
        self.TIMESTAMP = timestamp

        possiblepaths = ['environment.json',
                        '../environment.json',
                        '/home/ucabcbo/sis2/environment.json']

        env = None
        for possiblepath in possiblepaths:
            if os.path.exists(possiblepath):
                with open(possiblepath) as f:
                    env = json.load(f)
                break
        assert env is not None
        environment:dict = env

        self.SAMPLE_FREQ = environment.get('sample_freq', 1000)
        self.CKPT_FREQ = environment.get('ckpt_freq', 5000)
        self.MAX_SHUFFLE_BUFFER = environment.get('max_shuffle_buffer', 500)

        self.RESTORE = restore


        with open(os.path.join(path, 'experiment.json')) as f:
            experiment:dict = json.load(f)


        assert 'model_name' in experiment
        self.MODEL_NAME:str = str(experiment['model_name'])
        self.DATASET:str = experiment.get('dataset', 'curated')
        self.TILESIZE:int = experiment.get('tilesize', 256)
        self.IMG_HEIGHT:int = experiment.get('img_height', self.TILESIZE)
        self.IMG_WIDTH:int = experiment.get('img_width', self.TILESIZE)

        self.RANDOM_RESIZE:float = experiment.get('random_resize', 1.11)
        #TODO: make random rotations the default
        self.RANDOM_ROTATE:bool = experiment.get('random_rotate', False)

        self.BATCH_SIZE:int = experiment.get('batch_size', 16)
        self.SHUFFLE:bool = experiment.get('shuffle', True)
        self.STEPS:int = experiment.get('steps', 40000)
        self.EXCLUDE_SUFFIX:str = experiment.get('exclude_suffix', None)
        self.ENFORCE_SUFFIX:str = experiment.get('enforce_suffix', None)

        if self.DATASET[:3] == 'alt':
            self.PARSEMODE = 'alt'
            self.INPUT_CHANNELS = 3
            self.OUTPUT_CHANNELS = 3
        else:
            self.PARSEMODE = None
            self.INPUT_CHANNELS = 21
            self.OUTPUT_CHANNELS = 3


        if experiment.get('sample_train', None) is not None and experiment.get('sample_val', None) is not None:
            self.DATA_SAMPLE = (experiment['sample_train'], experiment['sample_val'])
        else:
            self.DATA_SAMPLE = None
            if experiment.get('sample_train', None) is not None or experiment.get('sample_val', None) is not None:
                print('W: Both sample_train and sample_val must be set for any to take effect.')


        default_ganloss = {
            "gen_gan": 1,
            "gen_nll": 0,
            "gen_l1": 100,
            "gen_l2": 0,
            "gen_rmse": 0,
            "gen_wstein": 0
        }
        default_discloss = {
            "disc_bce": 1,
            "disc_nll": 0
        }
        self.GEN_LOSS = experiment.get('gen_loss', default_ganloss)
        self.DISC_LOSS = experiment.get('disc_loss', default_discloss)
        self.PARAMS = experiment.get('params', {})

        self.output = Experiment.Output(path, self.STRINGNAME, self.TIMESTAMP)

        assert all(key in environment for key in ['environment', 'project_root', 'data_root', 'experiment_root'])
        self.ENVIRONMENT = environment['environment']
        self.PROJECT_ROOT = environment['project_root']
        self.DATA_ROOT = os.path.join(environment['data_root'], self.DATASET, str(self.TILESIZE))
        self.EXPERIMENT_ROOT = environment['experiment_root']

        sys.path.append(self.PROJECT_ROOT)

"""
Structure of the `experiment.json` file:
{
    Model name, must be found as python module in `models` directory
    "model_name": "aris-a",

    Dataset name, must be found as path in data root directory
    "dataset": "cur_masked",

    Tilesize
    default: 256
    "tilesize": 256,

    Image height/width
    default: same as tilesize
    "img_height": 256,
    "img_width": 256,

    Numer of training steps
    default: 40000
    "steps": 40000,

    Number of random sample images used for training/testing - null if entire dataset shall be used
    default: null
    "sample_train": 10000,
    "sample_val": 1000,

    Batch size
    default: 16
    "batch_size": 16,

    Shuffle
    default: true
    "shuffle": true,

    Random resize factor (float)
    default: 1.11
    "random_resize": 1.2,

    Random rotate
    default: true
    "random_rotate": true,

    Enforce that filenames include a certain suffix
    default: null
    "enfore_suffix": "notinmask",

    Excelude files with a certain suffix in the filename
    default: null
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
"""