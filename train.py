### Arguments
import argparse
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--exp", required=True, help="Experiment name, representing folder structure below experiment root directory")
parser.add_argument("--timestamp", required=False, default=datetime.now().strftime("%m%d-%H%M"), help="Timestamp")
parser.add_argument("--restore", required=False, default=None, help="Checkpoint to be restored (experiment output directory)")

a = parser.parse_args()
print(f'Arguments read: {a}')
### End arguments

import sys
import os
import json
sys.path.append(os.getcwd())

from environment import Environment

env = Environment()

if env.ENVIRONMENT == 'blaze':
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

from experiment import Experiment
from dataset import Reader
import toolbox as tbx

experiment_root = os.path.join(env.EXPERIMENT_ROOT, a.exp)
exp = Experiment(experiment_root, a.exp, a.timestamp)


# Dynamically import all classes in the directory
modules = []
for filename in os.listdir(os.path.join(exp.PROJECT_ROOT, 'models')):
    if filename.endswith('.py'):
        module_name = filename[:-3]
        modules.append(importlib.import_module("." + module_name, package='models'))

model = None
for module in modules:
    if module.__name__[-len(exp.MODEL_NAME):] == exp.MODEL_NAME:
        model = module.GAN(exp)
assert model is not None

generator:tf.keras.Model = model.generator
discriminator:tf.keras.Model = model.discriminator

req_tf_version = '2.8.0'
tf_gtet_280 = True
for reqPart, part in zip(map(int, req_tf_version.split(".")), map(int, tf.__version__.split("."))):
    if reqPart > part:
        tf_gtet_280 = False
        break
    if reqPart < part:
        break

if tf_gtet_280:
    tf.keras.utils.plot_model(generator, show_shapes=True, expand_nested=False, show_layer_activations=True, to_file=os.path.join(exp.output.MODEL, 'generator.png'))
    tf.keras.utils.plot_model(generator, show_shapes=True, expand_nested=True, show_layer_activations=True, to_file=os.path.join(exp.output.MODEL, 'generator_full.png'))
    tf.keras.utils.plot_model(discriminator, show_shapes=True, expand_nested=False, show_layer_activations=True, to_file=os.path.join(exp.output.MODEL, 'discriminator.png'))
    tf.keras.utils.plot_model(discriminator, show_shapes=True, expand_nested=True, show_layer_activations=True, to_file=os.path.join(exp.output.MODEL, 'discriminator_full.png'))
else:
    tf.keras.utils.plot_model(generator, show_shapes=True, expand_nested=False, to_file=os.path.join(exp.output.MODEL, 'generator.png'))
    tf.keras.utils.plot_model(discriminator, show_shapes=True, expand_nested=False, to_file=os.path.join(exp.output.MODEL, 'discriminator.png'))

dataset_reader = Reader(exp, 'train.py')
train_dataset = dataset_reader.train_dataset
test_dataset = dataset_reader.test_dataset

checkpoint = tf.train.Checkpoint(
    generator_optimizer=model.generator_optimizer,
    discriminator_optimizer=model.discriminator_optimizer,
    generator=generator,
    discriminator=discriminator,
    step=tf.Variable(0, dtype=tf.int64))

stepoffset = 0
latest_checkpoint = None
if a.restore is not None:
    print('Trying to restore: ' + os.path.join(exp.output.CKPT, '..', a.restore))
    latest_checkpoint = tf.train.latest_checkpoint(os.path.join(exp.output.CKPT, '..', a.restore))
    checkpoint.restore(latest_checkpoint).expect_partial()
    stepoffset = int(checkpoint.step)
    print("Loaded checkpoint:", latest_checkpoint)
    print("Continue at step:", stepoffset)


with open(os.path.join(exp.output.LOGS, f'experiment.json'), 'w') as f:
    logdict = {}
    logdict['environment'] = exp.ENVIRONMENT
    logdict['PID'] = os.getpid()
    logdict['timestamp'] = exp.TIMESTAMP
    logdict['experiment_root'] = env.EXPERIMENT_ROOT
    if a.restore is not None:
        logdict['ckpt_requested'] = a.restore
        if latest_checkpoint is not None:
            logdict['ckpt_loaded'] = f'{latest_checkpoint}:{stepoffset}'
    json.dump(logdict, f, indent=4)


def save_checkpoint(step:int):
    checkpoint.step.assign(step)
    print(f'Step + 1 = {step + 1} - saving checkpoint (saved: {int(checkpoint.step)})')
    checkpoint.save(os.path.join(exp.output.CKPT, 'ckpt'))


def fit(train_ds, test_ds, steps):
    start = time.time()
    example_targets = []
    example_inputs = []
    for example_target, example_input in test_ds.take(5):
        example_targets.append(example_target[0])
        example_inputs.append(example_input[0])
    example_inputs = tf.stack(example_inputs, axis=0)
    example_targets = tf.stack(example_targets, axis=0)

    for step, (target, input_image) in train_ds.repeat().take(steps).enumerate():
        if (step == 0) or ((step + stepoffset + 1) % exp.SAMPLE_FREQ == 0):
            if step != 0:
                print(f'Time taken for {exp.SAMPLE_FREQ} steps: {time.time()-start:.2f} sec')
                start = time.time()
            
            if exp.PARSEMODE == 'alt':
                tbx.generate_images_alt(generator,
                                    example_inputs,
                                    example_targets,
                                    showimg=False,
                                    PATH_IMGS=exp.output.SAMPLES,
                                    model_name=f'{exp.STRINGNAME}_{a.timestamp}',
                                    iteration=(step + stepoffset + 1))
            else:
                tbx.generate_images(generator,
                                    example_inputs,
                                    example_targets,
                                    showimg=False,
                                    PATH_IMGS=exp.output.SAMPLES,
                                    model_name=f'{exp.STRINGNAME}_{a.timestamp}',
                                    iteration=(step + stepoffset + 1))
            print(f"Step: {step + stepoffset + 1}")

        model.train_step(input_image, target, step)

        # Training step
        if (step + 1) % 10 == 0:
            print('.', end='', flush=True)

        # Save (checkpoint) the model every n steps
        if (step + 1) % exp.CKPT_FREQ == 0:
            save_checkpoint(step + stepoffset)


fit(train_dataset, test_dataset, steps=(exp.STEPS - stepoffset))

print('EXPERIMENT COMPLETED')