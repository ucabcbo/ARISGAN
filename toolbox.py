
def normalize_numpy(array):
    array_min, array_max = array.min(), array.max()
    return (array - array_min) / (array_max - array_min)

from enum import Enum

class RGBProfile(Enum):
    S2 = [3,2,1]
    S3 = [17,6,3]
    S3_TRISTIMULUS = [17,5,2]

tiff_bandoffset = {RGBProfile.S2.name: 2,
                   RGBProfile.S3.name: 5,
                   RGBProfile.S3_TRISTIMULUS.name: 5}

tensor_bandoffset = {RGBProfile.S2.name: -1,
                     RGBProfile.S3.name: -1,
                     RGBProfile.S3_TRISTIMULUS.name: -1}


def plot_tiff(raw_tiff, rgbprofile, ax=None):

    bands = rgbprofile.value
    bands = [item + tiff_bandoffset[rgbprofile.name] for item in bands]

    import numpy as np
    from matplotlib import pyplot as plt

    red_band = normalize_numpy(raw_tiff.read(bands[0]))
    green_band = normalize_numpy(raw_tiff.read(bands[1]))
    blue_band = normalize_numpy(raw_tiff.read(bands[2]))

    # Stack the bands to create the RGB image
    # rgb_image = rasterio.plot.reshape_as_image([red_band, green_band, blue_band])
    rgb_image = np.stack([red_band, green_band, blue_band], axis=-1)

    # Display the RGB image
    if ax is None:
        plt.figure(figsize=(10,10))
        plt.imshow(rgb_image)
        plt.axis('off')
        plt.show()
    else:
        ax.imshow(rgb_image)
        ax.axis('off')


def plot_tiff_channel(raw_tiff, channel:int, ax=None):

    import numpy as np
    from matplotlib import pyplot as plt

    channel = normalize_numpy(raw_tiff.read(channel))

    # Display the RGB image
    if ax is None:
        plt.figure(figsize=(10,10))
        plt.imshow(channel, cmap='gray')
        plt.axis('off')
        plt.show()
    else:
        ax.imshow(channel, cmap='gray')
        ax.axis('off')



def plot_tiff_sbs(raw_tiff, left_rgbprofile=RGBProfile.S2, right_rgbprofile=RGBProfile.S3, title=None):
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    if title is not None:
        fig.suptitle(title)
    plot_tiff(raw_tiff, left_rgbprofile, ax=ax[0])
    plot_tiff(raw_tiff, right_rgbprofile, ax=ax[1])
    ax[0].set_title(left_rgbprofile.name)
    ax[1].set_title(right_rgbprofile.name)
    plt.tight_layout()
    plt.show()

def plot_tensor(tensor, rgbprofile, ax=None):

    # print(banddelta)
    # print(type(rgbprofile))
    # print(type(RGBProfile.S3))
    bands = rgbprofile.value
    bands = [item + tensor_bandoffset[rgbprofile.name] for item in bands]

    import numpy as np
    from matplotlib import pyplot as plt

    nptensor = tensor.numpy()

    red_band = normalize_numpy(nptensor[:,:,bands[0]])
    green_band = normalize_numpy(nptensor[:,:,bands[1]])
    blue_band = normalize_numpy(nptensor[:,:,bands[2]])

    # Stack the bands to create the RGB image
    # rgb_image = rasterio.plot.reshape_as_image([red_band, green_band, blue_band])
    rgb_image = np.stack([red_band, green_band, blue_band], axis=-1)

    if ax is None:
        # Display the RGB image
        plt.figure(figsize=(10,10))
        plt.imshow(rgb_image)
        plt.axis('off')
        plt.show()
    else:
        ax.imshow(rgb_image)
        ax.axis('off')


def plot_tensor_sbs(tensor, tilesize, s3_rgbprofile=RGBProfile.S3, title=None):
    from matplotlib import pyplot as plt

    s2_tensor, s3_tensor = parse_tfrecord(tensor, tilesize)
    
    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    if title is not None:
        fig.suptitle(title)
    plot_tensor(s2_tensor, RGBProfile.S2, ax=ax[0])
    plot_tensor(s3_tensor, s3_rgbprofile, ax=ax[1])
    ax[0].set_title('Sentinel-2')
    ax[1].set_title('Sentinel-3')
    plt.tight_layout()
    plt.show()


def plot_tensor_sbs_alt(tensor, tilesize, title=None):
    from matplotlib import pyplot as plt

    s2_tensor, s3_tensor = parse_tfrecord_alt(tensor, tilesize)
    
    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    if title is not None:
        fig.suptitle(title)
    plot_tensor(s2_tensor, RGBProfile.S2, ax=ax[0])
    plot_tensor(s3_tensor, RGBProfile.S2, ax=ax[1])
    ax[0].set_title('Sentinel-2')
    ax[1].set_title('Sentinel-2 downsampled')
    plt.tight_layout()
    plt.show()


def save_tfrecord(raw_tiff, filepath):
    import numpy as np
    import tensorflow as tf

    raw_np = np.transpose(raw_tiff.read(), (1, 2, 0))
    #TODO: adjust if tiff structure changes
    raw_s2 = raw_np[:,:,2:5]
    raw_s3 = raw_np[:,:,5:26]

    writer = tf.io.TFRecordWriter(filepath)

    sample = tf.train.Example(features=tf.train.Features(feature={
        'raw_s2': tf.train.Feature(float_list=tf.train.FloatList(value=raw_s2.flatten())),
        'raw_s3': tf.train.Feature(float_list=tf.train.FloatList(value=raw_s3.flatten())),
    }))

    writer.write(sample.SerializeToString())
    writer.close()


def save_tfrecord_alt(raw_tiff, filepath, downsample:int=6):
    import numpy as np
    import tensorflow as tf
    from skimage.measure import block_reduce

    raw_np = np.transpose(raw_tiff.read(), (1, 2, 0))
    #TODO: adjust if tiff structure changes
    raw_s2 = raw_np[:,:,2:5]

    #ChatGPT MEAN downsampling:
    # blocks = raw_s2.reshape(raw_s2.shape[0] // downsample, downsample,
    #                         raw_s2.shape[1] // downsample, downsample)
    # average_blocks = blocks.mean(axis=(1, 3))

    downsampled_array = block_reduce(raw_s2, downsample, np.mean)

    # array_85x85 = raw_s2[::downsample, ::downsample]
    array_85x85 = downsampled_array
    expanded_array = np.kron(array_85x85, np.ones((downsample, downsample, 1)))
    cropped_array = expanded_array[:256, :256, :]

    writer = tf.io.TFRecordWriter(filepath)

    sample = tf.train.Example(features=tf.train.Features(feature={
        'raw_s2': tf.train.Feature(float_list=tf.train.FloatList(value=raw_s2.flatten())),
        'raw_s2_alt': tf.train.Feature(float_list=tf.train.FloatList(value=cropped_array.flatten())),
    }))

    writer.write(sample.SerializeToString())
    writer.close()


def parse_tfrecord(tfrecord, tilesize):
    import tensorflow as tf

    record_description = {
        'raw_s2': tf.io.FixedLenFeature([tilesize, tilesize, 3], tf.float32),
        'raw_s3': tf.io.FixedLenFeature([tilesize, tilesize, 21], tf.float32),
    }
    sample = tf.io.parse_single_example(tfrecord, record_description)
    raw_s2 = sample['raw_s2']
    raw_s3 = sample['raw_s3']
    return raw_s2, raw_s3


def parse_tfrecord_alt(tfrecord, tilesize):
    import tensorflow as tf

    record_description = {
        'raw_s2': tf.io.FixedLenFeature([tilesize, tilesize, 3], tf.float32),
        'raw_s2_alt': tf.io.FixedLenFeature([tilesize, tilesize, 3], tf.float32),
    }
    sample = tf.io.parse_single_example(tfrecord, record_description)
    raw_s2 = sample['raw_s2']
    raw_s2_alt = sample['raw_s2_alt']
    return raw_s2, raw_s2_alt


def generate_images(model, example_input, example_target, num_images=5, showimg=True, ROOTPATH_IMGS=None, PATH_IMGS=None, model_name=None, iteration=None):

    import os
    from matplotlib import pyplot as plt

    prediction = model(example_input, training=False)

    num_images = min(num_images, len(example_input))

    fig, ax = plt.subplots(num_images, 3, figsize=(15,5*num_images))

    for i in range(num_images):
        display_list = [example_input[i], example_target[i], prediction[i]]
        tempax = ax[i] if num_images > 1 else ax

        plot_tensor(display_list[0], RGBProfile.S3, ax=tempax[0])
        tempax[0].set_title('Input Image')
        tempax[0].axis('off')

        plot_tensor(display_list[1], RGBProfile.S2, ax=tempax[1])
        tempax[1].set_title('Ground Truth')
        tempax[1].axis('off')

        plot_tensor(display_list[2], RGBProfile.S2, ax=tempax[2])
        tempax[2].set_title('Predicted Image')
        tempax[2].axis('off')

    plt.tight_layout()
    if PATH_IMGS is not None:
        plt.savefig(os.path.join(PATH_IMGS, f'{model_name}_{iteration:05d}.png'))
    if ROOTPATH_IMGS is not None:
        plt.savefig(os.path.join(ROOTPATH_IMGS, f'{model_name}.png'))
    if showimg:
        plt.show()
    else:
        plt.close()
    

def generate_images_alt(model, example_input, example_target, num_images=5, showimg=True, ROOTPATH_IMGS=None, PATH_IMGS=None, model_name=None, iteration=None):

    import os
    from matplotlib import pyplot as plt

    prediction = model(example_input, training=False)

    num_images = min(num_images, len(example_input))

    fig, ax = plt.subplots(num_images, 3, figsize=(15,5*num_images))

    for i in range(num_images):
        display_list = [example_input[i], example_target[i], prediction[i]]
        tempax = ax[i] if num_images > 1 else ax

        plot_tensor(display_list[0], RGBProfile.S2, ax=tempax[0])
        tempax[0].set_title('Input Image')
        tempax[0].axis('off')

        plot_tensor(display_list[1], RGBProfile.S2, ax=tempax[1])
        tempax[1].set_title('Ground Truth')
        tempax[1].axis('off')

        plot_tensor(display_list[2], RGBProfile.S2, ax=tempax[2])
        tempax[2].set_title('Predicted Image')
        tempax[2].axis('off')

    plt.tight_layout()
    if PATH_IMGS is not None:
        plt.savefig(os.path.join(PATH_IMGS, f'{model_name}_{iteration:05d}.png'))
    if ROOTPATH_IMGS is not None:
        plt.savefig(os.path.join(ROOTPATH_IMGS, f'{model_name}.png'))
    if showimg:
        plt.show()
    else:
        plt.close()

