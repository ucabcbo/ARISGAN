import os
import json
import sys

class Environment:

    def __init__(self):

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
