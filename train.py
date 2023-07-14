### Arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--exp", required=True, help="Experiment name (without ending), must be located in /experiments folder")

a = parser.parse_args()
print(f'Arguments read: {a}')
### End arguments


import sys
import os
sys.path.append(os.getcwd())
import init

if init.ENVIRONMENT == 'blaze':
    import subprocess

    command = 'source /usr/local/cuda/CUDA_VISIBILITY.csh'
    process = subprocess.Popen(command, shell=True, executable="/bin/csh", stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    os.environ['CUDA_VISIBLE_DEVICES'] = stdout.decode()[-2]
    # os.environ['CUDA_HOME'] = '/opt/cuda/cuda-10.0'
    print(stdout.decode())

    command = 'source /server/opt/cuda/enable_cuda_11.0'
    process = subprocess.Popen(command, shell=True, executable="/bin/csh", stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    command = 'echo $CUDA_VISIBLE_DEVICES'
    process = subprocess.Popen(command, shell=True, executable="/bin/csh", stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    print(command)
    print(stdout.decode())

import tensorflow as tf

### GPU checks
from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if True]

available_gpus = get_available_gpus()

tf.config.list_physical_devices()

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

with tf.compat.v1.Session() as sess:
    device_name = tf.test.gpu_device_name()
    if device_name != '':
        print('TensorFlow is using GPU:', device_name)
    else:
        print('TensorFlow is not using GPU')
### End GPU checks

import time
import importlib
import json

from dataset.reader import Reader
import sis_toolbox as tbx


with open(os.path.join(init.PROJECT_ROOT, 'experiments', f'{a.exp}.json')) as f:
    experiment = json.load(f)

EXP = experiment

MODEL_NAME = experiment['model_name']
BATCH_SIZE = experiment['batch_size']
SHUFFLE = experiment['shuffle']
STEPS = experiment['steps']

DATA_SAMPLE = None
if experiment['sample_train'] is not None and experiment['sample_val'] is not None:
    DATA_SAMPLE = (experiment['sample_train'],experiment['sample_val'])
else:
    if experiment['sample_train'] is not None or experiment['sample_val'] is not None:
        print('W: Both sample_train and sample_val must be set for any to take effect.')

GEN_LOSS = experiment['gen_loss']
DISC_LOSS = experiment['disc_loss']
PARAMS = experiment['params']

SUBFOLDER = f'{init.TIMESTAMP}_{a.exp}_{BATCH_SIZE}x{init.TILESIZE}'
OUTPUT = dict()
OUTPUT['logs'] = os.path.join(init.OUTPUT_ROOT, f'{SUBFOLDER}/logs/')
OUTPUT['ckpt'] = os.path.join(init.OUTPUT_ROOT, f'{SUBFOLDER}/ckpt/')
OUTPUT['samples'] = os.path.join(init.OUTPUT_ROOT, f'{SUBFOLDER}/samples/')
OUTPUT['model'] = os.path.join(init.OUTPUT_ROOT, f'{SUBFOLDER}/model/')
os.makedirs(OUTPUT['logs'], exist_ok=True)
os.makedirs(OUTPUT['ckpt'], exist_ok=True)
os.makedirs(OUTPUT['samples'], exist_ok=True)
os.makedirs(OUTPUT['model'], exist_ok=True)

with open(os.path.join(OUTPUT['model'], 'experiment.json'), 'w') as f:
    experiment['environment'] = init.ENVIRONMENT
    experiment['PID'] = os.getpid()
    experiment['timestamp'] = init.TIMESTAMP
    experiment['output_root'] = os.path.join(init.OUTPUT_ROOT, SUBFOLDER)
    json.dump(experiment, f, indent=4)


# Dynamically import all classes in the directory
modules = []
for filename in os.listdir(os.path.join(init.PROJECT_ROOT, 'models')):
    if filename.endswith('.py'):
        module_name = filename[:-3]
        modules.append(importlib.import_module("." + module_name, package='models'))

model = None
for module in modules:
    if module.__name__[-len(MODEL_NAME):] == MODEL_NAME:
        model = module.GAN(OUTPUT, PARAMS, GEN_LOSS, DISC_LOSS)


generator = model.generator
discriminator = model.discriminator

req_tf_version = '2.8.0'
tf_gtet_280 = True
for reqPart, part in zip(map(int, req_tf_version.split(".")), map(int, tf.__version__.split("."))):
    if reqPart > part:
        tf_gtet_280 = False
        break
    if reqPart < part:
        break

if tf_gtet_280:
    tf.keras.utils.plot_model(generator, show_shapes=True, expand_nested=False, show_layer_activations=True, to_file=os.path.join(OUTPUT['model'], '10_gen.png'))
    tf.keras.utils.plot_model(generator, show_shapes=True, expand_nested=True, show_layer_activations=True, to_file=os.path.join(OUTPUT['model'], '11_gen_full.png'))
    tf.keras.utils.plot_model(discriminator, show_shapes=True, expand_nested=False, show_layer_activations=True, to_file=os.path.join(OUTPUT['model'], '20_disc.png'))
    tf.keras.utils.plot_model(discriminator, show_shapes=True, expand_nested=True, show_layer_activations=True, to_file=os.path.join(OUTPUT['model'], '21_disc_full.png'))
else:
    tf.keras.utils.plot_model(generator, show_shapes=True, expand_nested=False, to_file=os.path.join(OUTPUT['model'], '10_gen.png'))
    tf.keras.utils.plot_model(discriminator, show_shapes=True, expand_nested=False, to_file=os.path.join(OUTPUT['model'], '20_disc.png'))


dataset_reader = Reader(BATCH_SIZE, SHUFFLE, 'train.py', DATA_SAMPLE)
train_dataset = dataset_reader.train_dataset
test_dataset = dataset_reader.test_dataset

def fit(train_ds, test_ds, steps):
    # example_target, example_input = next(iter(test_ds.take(1)))
    start = time.time()
    example_targets = []
    example_inputs = []
    for example_target, example_input in test_dataset.take(5):
        example_targets.append(example_target[0])
        example_inputs.append(example_input[0])
    example_inputs = tf.stack(example_inputs, axis=0)
    example_targets = tf.stack(example_targets, axis=0)

    for step, (target, input_image) in train_ds.repeat().take(steps).enumerate():
        if (step == 0) or ((step + 1) % init.SAMPLE_FREQ == 0):
            if step != 0:
                print(f'Time taken for {init.SAMPLE_FREQ} steps: {time.time()-start:.2f} sec')
                start = time.time()
            
            tbx.generate_images(generator, example_inputs, example_targets, showimg=False, PATH_IMGS=OUTPUT['samples'], model_name=a.exp, iteration=(step + 1))
            print(f"Step: {step + 1}")

        model.train_step(input_image, target, step)

        # Training step
        if (step + 1) % 10 == 0:
            print('.', end='', flush=True)

        # Save (checkpoint) the model every 5k steps
        if (step + 1) % init.CKPT_FREQ == 0:
            print(f'Step + 1 = {step + 1} - saving checkpoint')
            model.save()

fit(train_dataset, test_dataset, steps=STEPS)
