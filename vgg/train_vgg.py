import sys
import os
sys.path.append(os.getcwd())
import environment

from dataset import Reader
import tensorflow as tf
from tensorflow.keras.applications import VGG19
import numpy as np
import os

if environment.ENVIRONMENT == 'blaze':
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


# Set the image dimensions and batch size
image_height = environment.TILESIZE
image_width = environment.TILESIZE

batch_size = 1
epochs = 10

# Custom data generator for loading TFRECORDS
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_height, image_width, batch_size):
        self.image_height = image_height
        self.image_width = image_width
        self.batch_size = batch_size
        # self.tfrecord_files = [f for f in os.listdir(init.TRAIN_DIR) if f.endswith('.tfrecord')]

        dataset = Reader(256, self.image_height, self.image_width,
                         environment.TRAIN_DIR,
                         environment.VAL_DIR,
                         batch_size,
                         False,
                         caller='train_vgg',
                         random_sample_size=20000)
        self.dataset = dataset.train_dataset.repeat()
        self.num_samples = len(dataset)

    def __len__(self):
        # return (self.num_samples + self.batch_size - 1) // self.batch_size
        return (self.num_samples // self.batch_size)
    
    def __getitem__(self, idx):
        batch_images = []
        batch = self.dataset.take(1)
        for step, (target, input_image) in batch.enumerate():
            # print(target[0])
            # tbx.plot_tensor(target[0], rgb.S2)
            batch_images = target
        return batch_images, np.ones(len(batch_images))  # Assuming all images belong to the same class (1)
        
# Create the custom data generator
data_generator = DataGenerator(image_height, image_width, batch_size)

# Load the VGG-19 model without the top classification layer
base_model = VGG19(include_top=False, weights=None, input_shape=(image_height, image_width, 3))

# Freeze the layers in the base model
# for layer in base_model.layers:
#     layer.trainable = False

# Add your custom top layers for classification
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
predictions = tf.keras.layers.Dense(1, activation='sigmoid')(x)

# Create the final model
model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# Set the number of steps per epoch based on the size of the dataset and batch size
steps_per_epoch = len(data_generator)

# Train the model
model.fit(
    data_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs
)

model.save(f'vgg/ckpt/VGG_{environment.TIMESTAMP}')
