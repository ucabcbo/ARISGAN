import os
import json
import sys

class Environment:
    """This class provides environment information and settings, common
    across all experiments on this environment. It reads the `environment.json`
    file in the project root directory. A documentation of this
    file is provided at the end.
    """

    def __init__(self):
        """Initializes the instance and provides properties to
        the environment-level settings
        """

        # Read the configuration file
        possiblepaths = ['environment.json',
                        '../environment.json',
                        '/home/ucabcbo/sis2/environment.json']
        environment = None
        for possiblepath in possiblepaths:
            if os.path.exists(possiblepath):
                with open(possiblepath) as f:
                    environment = json.load(f)
                break

        assert environment is not None

        self.SAMPLE_FREQ = environment.get('sample_freq', 1000)
        self.CKPT_FREQ = environment.get('ckpt_freq', 5000)

        self.MAX_SHUFFLE_BUFFER = environment.get('max_shuffle_buffer', 500)

        self.ENVIRONMENT = environment.get('environment', None)

        self.PROJECT_ROOT = environment.get('project_root', None)
        self.DATA_ROOT = environment.get('data_root', None)
        self.EXPERIMENT_ROOT = environment.get('experiment_root', None)

        self.S2_ROOT = environment.get('s2_root', None)
        self.S3_ROOT = environment.get('s3_root', None)

        sys.path.append(self.PROJECT_ROOT)

        print('environment loaded')

"""
Structure of the `environment.json` file:
{
    Environment name, especially for logs
    "environment": "cpom",

    Root, data, and experiment in/output directories
    "project_root": "/home/cb/sis2/",
    "data_root": "/home/cb/sis2/data/",
    "experiment_root": "/home/cb/sis2/experiments/",

    Data directory for original S2 and S3 files
    "s2_root": "/cpnet/projects/sikuttiaq/pond_inlet/Sentinel_2/DATA/",
    "s3_root": "/cpnet/projects/sikuttiaq/pond_inlet/Sentinel_3/OLCI/",

    After each n-th step, a sample is created and saved to the samples subdirectory
    default: 1000
    "sample_freq": 1000,
    
    After each n-th step, a checkpoint is created
    default: 5000
    "ckpt_freq": 10000,

    Maximum shuffle buffer size
    default: 500
    "max_shuffle_buffer": 500

}
"""