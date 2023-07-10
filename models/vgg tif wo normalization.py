import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import rasterio
import cv2

# Set the path to your image data directory
data_dir = '/home/cb/sis2/data/tif256/'

# Set the image dimensions and batch size
image_height = 256
image_width = 256
batch_size = 32

# Custom data generator for loading TIFF images
class TiffDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_dir, image_height, image_width, batch_size):
        self.data_dir = data_dir
        self.image_height = image_height
        self.image_width = image_width
        self.batch_size = batch_size
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith('.tif')]
        self.num_samples = len(self.image_files)
    
    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size
    
    def __getitem__(self, idx):
        batch_files = self.image_files[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_images = []
        
        for file in batch_files:
            image_path = os.path.join(self.data_dir, file)
            # image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            # image_channels = image[:, :, 2:5]
            # image_resized = cv2.resize(image_channels, (self.image_width, self.image_height))
            with rasterio.open(image_path) as dataset:
                image = dataset.read([3, 4, 5])  # Read channels 2, 3, and 4
                image = image.transpose((1, 2, 0))  # Transpose to (height, width, channels)
                # image_resized = cv2.resize(image, (256, 256))
                batch_images.append(image)
            # batch_images.append(image_resized)
        
        batch_images = np.array(batch_images)
        return batch_images, np.ones(len(batch_images))  # Assuming all images belong to the same class (1)
    
# Create the custom data generator
data_generator = TiffDataGenerator(data_dir, image_height, image_width, batch_size)

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
