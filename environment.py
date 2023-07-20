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

        self.SAMPLE_FREQ = environment['sample_freq']
        self.CKPT_FREQ = environment['ckpt_freq']

        self.MAX_SHUFFLE_BUFFER = environment['max_shuffle_buffer']

        self.ENVIRONMENT = environment['environment']

        self.PROJECT_ROOT = environment['project_root']
        self.DATA_ROOT = environment['data_root']
        self.EXPERIMENT_ROOT = environment['experiment_root']

        sys.path.append(self.PROJECT_ROOT)

        print('environment loaded')
