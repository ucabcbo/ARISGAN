import sys
import os
sys.path.append(os.getcwd())
import init

### Arguments
import argparse
parser = argparse.ArgumentParser()
# parser.add_argument("--m", default='pix2pix', help="model name")
parser.add_argument("--m", required=True, help="model name")
parser.add_argument("--b", type=int, default=1, help="batch size")
parser.add_argument("--shuffle", default='y', help="shuffle train/val data")
parser.add_argument("--prog_freq", type=int, default=1000, help="display progress every n steps")
parser.add_argument("--save_freq", type=int, default=5000, help="save model every n steps")
parser.add_argument("--suffix", default=None, help="suffix for output paths")

a = parser.parse_args()
print(f'Arguments read: {a}')

MODEL_NAME = a.m
BATCH_SIZE = a.b
SHUFFLE = a.shuffle
PROGRESS_FREQ = a.prog_freq
SAVE_FREQ = a.save_freq
SUFFIX = a.suffix


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
import time
import importlib

from dataset.reader import Reader
import sis_toolbox as tbx

SUBFOLDER = f'{init.TIMESTAMP}_{MODEL_NAME}_{BATCH_SIZE}x{init.TILESIZE}'
if SUFFIX is not None:
    SUBFOLDER += f'_{SUFFIX}'

path_logs = os.path.join(init.OUTPUT_ROOT, f'{SUBFOLDER}/logs/')
path_ckpt = os.path.join(init.OUTPUT_ROOT, f'{SUBFOLDER}/ckpt/')
path_imgs = os.path.join(init.OUTPUT_ROOT, f'{SUBFOLDER}/samples/')
path_model = os.path.join(init.OUTPUT_ROOT, f'{SUBFOLDER}/model/')
os.makedirs(path_logs, exist_ok=True)
os.makedirs(path_ckpt, exist_ok=True)
os.makedirs(path_imgs, exist_ok=True)
os.makedirs(path_model, exist_ok=True)

# Dynamically import all classes in the directory
directory = 'model'
modules = []
for filename in os.listdir(directory):
    if filename.endswith('.py'):
        module_name = filename[:-3]
        modules.append(importlib.import_module("." + module_name, package=directory))

model = None
for module in modules:
    if module.__name__[-len(MODEL_NAME):] == MODEL_NAME:
        model = module.GAN(path_logs, path_ckpt)


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
    tf.keras.utils.plot_model(generator, show_shapes=True, expand_nested=False, show_layer_activations=True, to_file=os.path.join(path_model, 'generator.png'))
    tf.keras.utils.plot_model(discriminator, show_shapes=True, expand_nested=False, show_layer_activations=True, to_file=os.path.join(path_model, 'discriminator.png'))
else:
    tf.keras.utils.plot_model(generator, show_shapes=True, expand_nested=False, to_file=os.path.join(path_model, 'generator.png'))
    tf.keras.utils.plot_model(discriminator, show_shapes=True, expand_nested=False, to_file=os.path.join(path_model, 'discriminator.png'))
None



shuffle = False if SHUFFLE == 'n' else True
dataset_reader = Reader(BATCH_SIZE, shuffle, 'train.py', (25000, 5000))
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
        if step % PROGRESS_FREQ == 0:
            # display.clear_output(wait=True)
            
            if step != 0:
                print(f'Time taken for {PROGRESS_FREQ} steps: {time.time()-start:.2f} sec\n')
                start = time.time()
            
            tbx.generate_images(generator, example_inputs, example_targets, showimg=False, PATH_IMGS=path_imgs, savemodel=MODEL_NAME, starttimestamp=init.TIMESTAMP, iteration=step)
            # for example_target, example_input in test_dataset.take(1):
            #     helper.generate_images(generator, example_input, example_target, showimg=False, PATH_IMGS=path_imgs, savemodel=model.name, starttimestamp=STARTTIME, iteration=step)

            print(f"Step: {step}")

        model.train_step(input_image, target, step)

        # Training step
        if (step + 1) % 10 == 0:
            print('.', end='', flush=True)

        # Save (checkpoint) the model every 5k steps
        if (step + 1) % SAVE_FREQ == 0:
            print(f'Step + 1 = {step + 1} - saving checkpoint')
            model.save()


fit(train_dataset, test_dataset, steps=40000)
