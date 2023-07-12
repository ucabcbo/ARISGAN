### Arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="tbd")
parser.add_argument("--batch_size", type=int, help="tbd", default=1)
parser.add_argument("--lmbda", type=int, help="tbd", default=100)
parser.add_argument("--progress_freq", type=int, default=1000, help="display progress every n steps")
parser.add_argument("--save_freq", type=int, default=5000, help="save model every n steps")
parser.add_argument("--suffix", default=None, help="suffix for output paths")
parser.add_argument("--shuffle", default='y', help="tbd")

# parser.add_argument("--lr", type=float, default=0.0001, help="initial learning rate for adam")
# parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
# parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
# parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")

# parser.add_argument("--ndf", type=int, default=32, help="number of generator filters in first conv layer")
# parser.add_argument("--train_count", type=int, default=64000, help="number of training data")
# parser.add_argument("--test_count", type=int, default=384, help="number of test data")
# parser.add_argument("--gpus", help="gpus to run", default="0")
# parser.add_argument("--blk", type=int, default=32, help="size of sample lr image")

a = parser.parse_args()
print(f'Arguments read: {a}')

assert a.model in ['psgan',
                   'pix2pix',
                   'pix2pix_psganloss',
                   'pix2pix_mse',
                   'psgan_mse',
                   'pix2pix_vgg',
                   'psgan_vgg',
                   'pix2pix_wstein']
### End arguments

import os
pid = os.getpid()
print("PID:", pid)

import sys
import os
sys.path.append(os.getcwd())
import init

print(f'running on environment: "{init.ENVIRONMENT}"')
assert init.ENVIRONMENT in ['blaze',
                       'colab',
                       'local',
                       'cpom']


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

import glob
import time
import datetime

import sis_toolbox as tbx

from models import pix2pix, psgan, pix2pix_psganloss, pix2pix_mse, psgan_mse, pix2pix_vgg, psgan_vgg, pix2pix_wstein
from dataset.reader import Reader

### GPU checks only
from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if True]

get_available_gpus()

tf.config.list_physical_devices()

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

with tf.compat.v1.Session() as sess:
    device_name = tf.test.gpu_device_name()
    if device_name != '':
        print('TensorFlow is using GPU:', device_name)
    else:
        print('TensorFlow is not using GPU')
### End GPU checks

path_subfolder = f'{init.TIMESTAMP}_{a.model}_{a.batch_size}x{init.TILESIZE}'
if a.suffix is not None:
    path_subfolder += f'_{a.suffix}'
path_log = os.path.join(init.OUTPUT_ROOT, f'{path_subfolder}/logs/')
path_ckpt = os.path.join(init.OUTPUT_ROOT, f'{path_subfolder}/ckpt/')
path_imgs = os.path.join(init.OUTPUT_ROOT, f'{path_subfolder}/samples/')
path_model = os.path.join(init.OUTPUT_ROOT, f'{path_subfolder}/model/')
os.makedirs(path_log, exist_ok=True)
os.makedirs(path_ckpt, exist_ok=True)
os.makedirs(path_imgs, exist_ok=True)
os.makedirs(path_model, exist_ok=True)

# The training set consist of n images
BUFFER_SIZE = len(glob.glob(os.path.join(init.TRAIN_DIR, '*')))

if a.model == 'pix2pix':
    model = pix2pix.Model(a.lmbda, path_log, path_ckpt)
elif a.model == 'psgan':
    model = psgan.Model(a.lmbda, path_log, path_ckpt)
elif a.model == 'pix2pix_psganloss':
    model = pix2pix_psganloss.Model(a.lmbda, path_log, path_ckpt)
elif a.model == 'pix2pix_mse':
    model = pix2pix_mse.Model(a.lmbda, path_log, path_ckpt)
elif a.model == 'psgan_mse':
    model = psgan_mse.Model(a.lmbda, path_log, path_ckpt)
elif a.model == 'pix2pix_vgg':
    model = pix2pix_vgg.Model(a.lmbda, path_log, path_ckpt)
elif a.model == 'psgan_vgg':
    model = psgan_vgg.Model(path_log, path_ckpt)
elif a.model == 'pix2pix_wstein':
    model = pix2pix_wstein.Model(a.lmbda, path_log, path_ckpt)
else:
    model = pix2pix.Model(a.lmbda, path_log, path_ckpt)


shuffle = False if a.shuffle == 'n' else True
dataset_reader = Reader(a.batch_size, shuffle, 'train.py', 25000)
train_dataset = dataset_reader.train_dataset
test_dataset = dataset_reader.test_dataset

generator = model.generator
discriminator = model.discriminator

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
        if step % a.progress_freq == 0:
            # display.clear_output(wait=True)
            
            if step != 0:
                print(f'Time taken for {a.progress_freq} steps: {time.time()-start:.2f} sec\n')
                start = time.time()
            
            tbx.generate_images(generator, example_inputs, example_targets, showimg=False, PATH_IMGS=path_imgs, savemodel=model.name, starttimestamp=init.TIMESTAMP, iteration=step)
            # for example_target, example_input in test_dataset.take(1):
            #     helper.generate_images(generator, example_input, example_target, showimg=False, PATH_IMGS=path_imgs, savemodel=model.name, starttimestamp=STARTTIME, iteration=step)

            print(f"Step: {step}")

        model.train_step(input_image, target, step)

        # Training step
        if (step + 1) % 10 == 0:
            print('.', end='', flush=True)

        # Save (checkpoint) the model every 5k steps
        if (step + 1) % a.save_freq == 0:
            print(f'Step + 1 = {step + 1} - saving checkpoint')
            model.save()

fit(train_dataset, test_dataset, steps=40000)