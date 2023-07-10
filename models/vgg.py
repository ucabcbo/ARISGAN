import sys
sys.path.append('/home/cb/sis2/')
from dataset.reader import Reader
import tensorflow as tf
from tensorflow.keras.applications import VGG19
import numpy as np
import os

# Set the path to your image data directory
data_dir = '/home/cb/sis2/data/tfrecords256/'

# Set the image dimensions and batch size
image_height = 256
image_width = 256
batch_size = 16

# Custom data generator for loading TIFF images
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_dir, image_height, image_width, batch_size):
        self.data_dir = data_dir
        self.image_height = image_height
        self.image_width = image_width
        self.batch_size = batch_size
        self.tfrecord_files = [f for f in os.listdir(data_dir) if f.endswith('.tfrecord')]
        self.num_samples = len(self.tfrecord_files)

        dataset = Reader(256, 256, 256,
                         '/home/cb/sis2/data/tfrecords256/',
                         '/home/cb/sis2/data/tfrecords256_val/',
                         self.num_samples, batch_size, False)
        self.dataset = dataset.train_dataset.repeat()
    
    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size
    
    def __getitem__(self, idx):
        batch_images = []
        batch = self.dataset.take(1)
        for step, (target, input_image) in batch.enumerate():
            # print(target[0])
            # tbx.plot_tensor(target[0], rgb.S2)
            batch_images = target
        return batch_images, np.ones(len(batch_images))  # Assuming all images belong to the same class (1)
        
# Create the custom data generator
data_generator = DataGenerator(data_dir, image_height, image_width, batch_size)

# Load the VGG-19 model without the top classification layer
base_model = VGG19(include_top=False, weights=None, input_shape=(image_height, image_width, 3))

# Freeze the layers in the base model
for layer in base_model.layers:
    layer.trainable = False

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

# Set the number of steps per epoch based on the size of the dataset and batch size
steps_per_epoch = len(data_generator)

# Train the model
model.fit(
    data_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=10
)

model.save('model256')