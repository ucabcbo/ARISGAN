# This script reads the latest checkpoint of a given experiment, and performs evaluation of the metrics across
# all images in the test dataset. Results are saved in the `_evaluation` subdirectory.
# If no timestamp is specified, the overall latest run will be chosen. Otherwise the latest checkpoint within
# the specified run.


### Arguments
import argparse
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--exp", required=True, help="Experiment name, representing folder structure below experiment root directory")
parser.add_argument("--timestamp", required=False, help="Explicit checkpoint to restore")

a = parser.parse_args()
print(f'Arguments read: {a}')
### End arguments

import sys
import os
sys.path.append(os.getcwd())

from environment import Environment

env = Environment()

import tensorflow as tf

import importlib
import pandas as pd

from experiment import Experiment
from dataset import Reader

from models import metrics

EXPERIMENT = a.exp

experiment_root = os.path.join(env.EXPERIMENT_ROOT, EXPERIMENT)

if a.timestamp is not None:
    timestamp = a.timestamp
else:
    timestamp = max(os.listdir(os.path.join(experiment_root, 'ckpt')))

exp = Experiment(experiment_root, EXPERIMENT, timestamp, restore=True)

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

dataset_reader = Reader(exp, 'evaluate.py')
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

print('Trying to restore: ' + os.path.join(exp.output.CKPT))
latest_checkpoint = tf.train.latest_checkpoint(os.path.join(exp.output.CKPT))
checkpoint.restore(latest_checkpoint).expect_partial()
stepoffset = int(checkpoint.step)
print("Loaded checkpoint:", latest_checkpoint)


def evaluate(test_ds:tf.data.Dataset) -> pd.DataFrame:
    """Evaluation of metrics using the `Metrics` class. All images in the test set are looped through.

    Parameters
    ----------
    test_ds : tf.data.Dataset
        The test dataset

    Returns
    -------
    pd.DataFrame
        The resulting metrics as dataframe
    """

    metric_results = pd.DataFrame()

    for step, (target, input_image) in test_ds.enumerate():
        # print(input_image.shape)
        prediction = generator(input_image, training=False)

        for (single_target, single_prediction) in zip(target, prediction):

            metric = metrics.Metrics(single_target.numpy(), single_prediction.numpy())
            df = pd.DataFrame([metric.getall()])
            metric_results = pd.concat([metric_results, df], ignore_index=True)
            # metric_results.append(metric.getall())
    
    return metric_results

# Calling the evaluation on the test dataset
eval_results = evaluate(test_dataset)

os.makedirs(os.path.join(exp.EXPERIMENT_ROOT, '_evaluation'), exist_ok=True)
eval_results.to_csv(os.path.join(exp.EXPERIMENT_ROOT, '_evaluation', f'{EXPERIMENT.replace("/", "_")}_{timestamp}.csv'), index_label='index')
