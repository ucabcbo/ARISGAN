### Arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="tbd")
parser.add_argument("--tilesize", type=int, help="tbd", default=256)
parser.add_argument("--img_width", type=int, help="tbd", default=256)
parser.add_argument("--img_height", type=int, help="tbd", default=256)
parser.add_argument("--batch_size", type=int, help="tbd", default=16)
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
                   'pix2pix_psganloss']
### End arguments

import os
pid = os.getpid()
print("PID:", pid)

with open('env.txt') as f:
    ENVIRONMENT = f.readlines()[0][:-1]
print(f'running on environment: "{ENVIRONMENT}"')
assert ENVIRONMENT in ['blaze',
                       'colab',
                       'local',
                       'cpom']


if ENVIRONMENT == 'blaze':
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

import sis_helper as helper

from models import pix2pix, psgan, pix2pix_psganloss
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


INPUT_CHANNELS = 21
OUTPUT_CHANNELS = 3
STARTTIME = datetime.datetime.now().strftime('%m%d-%H%M')

if ENVIRONMENT == 'blaze':
    path_prefix = '/cs/student/msc/aisd/2022/cboehm/projects/li1_data/'
elif ENVIRONMENT == 'colab':
    path_prefix = f'/content/drive/MyDrive/sis2/data/'
elif ENVIRONMENT == 'local':
    path_prefix = f'/Users/christianboehm/projects/sis2/data/'
elif ENVIRONMENT == 'cpom':
    path_prefix = f'/home/cb/sis2/data/'
else:
    path_prefix = f'~/projects/sis2/data'

path_train = os.path.join(path_prefix, f'tfrecords{a.tilesize}/')
path_val = os.path.join(path_prefix, f'tfrecords{a.tilesize}/')

path_subfolder = f'{STARTTIME}_{a.model}_{a.batch_size}x{a.tilesize}'
if a.suffix is not None:
    path_subfolder += f'_{a.suffix}'
path_log = os.path.join(path_prefix, f'logs/{path_subfolder}/')
path_ckpt = os.path.join(path_prefix, f'checkpoints/{path_subfolder}/')
path_imgs = os.path.join(path_prefix, f'images/{path_subfolder}/')
os.makedirs(path_log, exist_ok=True)
os.makedirs(path_ckpt, exist_ok=True)
os.makedirs(path_imgs, exist_ok=True)

# The training set consist of n images
BUFFER_SIZE = len(glob.glob(os.path.join(path_train, '*')))

if a.model == 'pix2pix':
    model = pix2pix.Model(a.img_width, a.img_height, INPUT_CHANNELS, OUTPUT_CHANNELS, a.lmbda, path_log, path_ckpt)
elif a.model == 'psgan':
    model = psgan.Model(a.img_width, a.img_height, INPUT_CHANNELS, OUTPUT_CHANNELS, a.lmbda, path_log, path_ckpt)
elif a.model == 'pix2pix_psganloss':
    model = pix2pix_psganloss.Model(a.img_width, a.img_height, INPUT_CHANNELS, OUTPUT_CHANNELS, a.lmbda, path_log, path_ckpt)

shuffle = False if a.shuffle == 'n' else True
dataset_reader = Reader(a.tilesize, a.img_width, a.img_height, path_train, path_val, BUFFER_SIZE, a.batch_size, shuffle)
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
        if (step) % a.progress_freq == 0:
            # display.clear_output(wait=True)
            
            if step != 0:
                print(f'Time taken for {a.progress_freq} steps: {time.time()-start:.2f} sec\n')
                start = time.time()
            
            helper.generate_images(generator, example_input, example_target, showimg=False, PATH_IMGS=path_imgs, savemodel=model.name, starttimestamp=STARTTIME, iteration=step)
            # for example_target, example_input in test_dataset.take(1):
            #     helper.generate_images(generator, example_input, example_target, showimg=False, PATH_IMGS=path_imgs, savemodel=model.name, starttimestamp=STARTTIME, iteration=step)

            print(f"Step: {step}")

        model.train_step(input_image, target, step)

        # Training step
        if (step + 1) % 10 == 0:
            print('.', end='', flush=True)

        # Save (checkpoint) the model every 5k steps
        if (step + 1) % a.save_freq == 0:
            model.save()

fit(train_dataset, test_dataset, steps=40000)