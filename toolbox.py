import numpy as np
from enum import Enum
from matplotlib import pyplot as plt
import rasterio
import tensorflow as tf
from skimage.transform import resize
from rasterio.enums import Resampling

class RGBProfile(Enum):
    """Class to define RGB profiles for the display of multi-channel tif images.\n
    Specifies which channel is used for red, green, blue bands
    """
    S2 = [3,2,1]
    S3 = [17,6,3]
    S3_TRISTIMULUS = [17,5,2]

tiff_bandoffset = {RGBProfile.S2.name: 2,
                   RGBProfile.S3.name: 5,
                   RGBProfile.S3_TRISTIMULUS.name: 5}
"""Related to the SIS2 standard tif structure, what is the channel offset to display the correct bands.\n
SIS2 TIFF channels:\n
0-1: collocation and quality masks\n
2-4: Sentinel-2 bands B2, B3, B4\n
5-26: Sentinel-2 bands 01-21
"""

tensor_bandoffset = {RGBProfile.S2.name: -1,
                     RGBProfile.S3.name: -1,
                     RGBProfile.S3_TRISTIMULUS.name: -1}
"""Related to the SIS2 standard tfrecord structure, what is the channel offset to display the correct bands.\n
In the concrete case, Sentinel-2 and Sentinel-3 images are separated in tensors already, and each starts at band 0.
"""


def normalize_numpy(array:np.ndarray):
    """Normalize a numpy array

    Parameters
    ----------
    array : numpy.ndarray
        Array of floats

    Returns
    -------
    numpy.ndarray
        Array of floats, normalized
    """
    array_min, array_max = array.min(), array.max()
    return (array - array_min) / (array_max - array_min)


def plot_tiff(raw_tiff:rasterio.io.DatasetReader, rgbprofile:RGBProfile, ax:plt.Axes=None):
    """Plot an image from a tif file

    Parameters
    ----------
    raw_tiff : rasterio.io.DatasetReader
        Tif file, as read by `rasterio.open()`
    rgbprofile : RGBProfile
        RGB profile to use for selecting RGB bands for display
    ax : plt.Axes, optional
        Axis to plot the result on, by default None, which will create a standalone plot
    """
    bands = rgbprofile.value
    bands = [item + tiff_bandoffset[rgbprofile.name] for item in bands]

    red_band = normalize_numpy(raw_tiff.read(bands[0]))
    green_band = normalize_numpy(raw_tiff.read(bands[1]))
    blue_band = normalize_numpy(raw_tiff.read(bands[2]))

    rgb_image = np.stack([red_band, green_band, blue_band], axis=-1)

    if ax is None:
        plt.figure(figsize=(10,10))
        plt.imshow(rgb_image)
        plt.axis('off')
        plt.show()
    else:
        ax.imshow(rgb_image)
        ax.axis('off')


def plot_tiff_channel(raw_tiff:rasterio.io.DatasetReader, channel:int, ax:plt.Axes=None):
    """Plot a single channel of a tif file

    Parameters
    ----------
    raw_tiff : rasterio.io.DatasetReader
        Tif file, as read by `rasterio.open()`
    channel : int
        Which channel to plot
    ax : plt.Axes, optional
        Axis to plot the result on, by default None, which will create a standalone plot
    """

    channel = normalize_numpy(raw_tiff.read(channel))

    if ax is None:
        plt.figure(figsize=(10,10))
        plt.imshow(channel, cmap='gray')
        plt.axis('off')
        plt.show()
    else:
        ax.imshow(channel, cmap='gray')
        ax.axis('off')


def plot_tiff_sbs(raw_tiff:rasterio.io.DatasetReader, left_rgbprofile:RGBProfile=RGBProfile.S2, right_rgbprofile:RGBProfile=RGBProfile.S3, title:str=None):
    """Convenience function: plot two images of the same tif side by side.

    Parameters
    ----------
    raw_tiff : rasterio.io.DatasetReader
        Tif file, as read by `rasterio.open()`
    left_rgbprofile : RGBProfile, optional
        RGB profile for the left image, by default RGBProfile.S2
    right_rgbprofile : RGBProfile, optional
        RGB profile for the right image, by default RGBProfile.S3
    title : str, optional
        Title, by default None
    """

    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    if title is not None:
        fig.suptitle(title)
    plot_tiff(raw_tiff, left_rgbprofile, ax=ax[0])
    plot_tiff(raw_tiff, right_rgbprofile, ax=ax[1])
    ax[0].set_title(left_rgbprofile.name)
    ax[1].set_title(right_rgbprofile.name)
    plt.tight_layout()
    plt.show()


def plot_tensor(tensor:tf.Tensor, rgbprofile:RGBProfile, ax:plt.Axes=None):
    """Plot an image of a tensor - tensor must be unpacked already

    Parameters
    ----------
    tensor : tf.Tensor
        Tensor to plot
    rgbprofile : RGBProfile
        Which RGB profile to use for the tensor plot
    ax : plt.Axes, optional
        Axis to plot the result on, by default None, which will create a standalone plot
    """
    bands = rgbprofile.value
    bands = [item + tensor_bandoffset[rgbprofile.name] for item in bands]

    nptensor = tensor.numpy()

    red_band = normalize_numpy(nptensor[:,:,bands[0]])
    green_band = normalize_numpy(nptensor[:,:,bands[1]])
    blue_band = normalize_numpy(nptensor[:,:,bands[2]])

    rgb_image = np.stack([red_band, green_band, blue_band], axis=-1)

    if ax is None:
        plt.figure(figsize=(10,10))
        plt.imshow(rgb_image)
        plt.axis('off')
        plt.show()
    else:
        ax.imshow(rgb_image)
        ax.axis('off')


def plot_tensor_sbs(tensor:tf.Tensor, tilesize:int, s3_rgbprofile:RGBProfile=RGBProfile.S3, title:str=None):
    """Convenience function: plot two images of one raw tensor side-by-side.\n
    Only supports Sentinel-2/Sentinel-3 combination, use `plot_tensor_sbs_alt()` for Sentinel-2/Sentinel-2 combination.

    Parameters
    ----------
    tensor : tf.Tensor
        Tensor to plot (packed)
    tilesize : int
        Tilesize in the tensor (required for unpacking)
    s3_rgbprofile : RGBProfile, optional
        RGB profile for the right image, by default RGBProfile.S3
    title : str, optional
        Title, by default None
    """

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


def plot_tensor_sbs_alt(tensor:tf.Tensor, tilesize:int, title:str=None):
    """Convenience function: plot two images of the same tensor side-by-side. 
    Only supports Sentinel-2/Sentinel-2 combination, use `plot_tensor_sbs()` for Sentinel-2/Sentinel-3 combination.

    Parameters
    ----------
    tensor : tf.Tensor
        Tensor to plot (packed)
    tilesize : int
        Tilesize in the tensor (required for unpacking)
    title : str, optional
        Title, by default None
    """
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


def save_tfrecord(raw_tiff:rasterio.io.DatasetReader, filepath:str):
    """Save tif file as tfrecord. Note that the tif structure is hardcoded and this function needs to be updated if it changes.
    This function is for Sentinel-2/Sentinel-3 tifs with 26 channels. Use `save_tfrecord_alt()` for Sentinel-2/Sentinel-2 combination.

    Parameters
    ----------
    raw_tiff : rasterio.io.DatasetReader
        Tif file, as read by `rasterio.open()`
    filepath : str
        Path+filename to save the tfrecord as
    """
    try:
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

    except Exception as e:
        print(f'Unexpected Exception in toolbox.save_tfrecord: {e}')


def save_tfrecord_alt(raw_tiff:rasterio.io.DatasetReader, filepath:str, downsample:int):
    """Save tif file as tfrecord. Note that the tif structure is hardcoded and this function needs to be updated if it changes.
    This function is for Sentinel-2/Sentinel-2 tifs with 26 channels. Use `save_tfrecord()` for Sentinel-2/Sentinel-3 combination.

    Parameters
    ----------
    raw_tiff : rasterio.io.DatasetReader
        Tif file, as read by `rasterio.open()`
    filepath : str
        Path+filename to save the tfrecord as
    downsample : int
        Factor by which to downsample the input image (function doesn't make sense without downsampling).
    """
    try:
        #TODO: adjust if tiff structure changes
        raw_s2 = np.transpose(raw_tiff.read(), (1, 2, 0))[:,:,2:5]
        tilesize = raw_s2.shape[1]

        data = raw_tiff.read(
            out_shape=(
                raw_tiff.count,
                int(raw_tiff.height / downsample),
                int(raw_tiff.width / downsample)
            ),
            resampling=Resampling.average
        )
        #TODO: adjust if tiff structure changes
        raw_np = np.transpose(data, (1, 2, 0))[:,:,2:5]

        output_array = resize(raw_np, (tilesize, tilesize), order=0, anti_aliasing=False)

        writer = tf.io.TFRecordWriter(filepath)

        sample = tf.train.Example(features=tf.train.Features(feature={
            'raw_s2': tf.train.Feature(float_list=tf.train.FloatList(value=raw_s2.flatten())),
            'raw_s2_alt': tf.train.Feature(float_list=tf.train.FloatList(value=output_array.flatten())),
        }))

        writer.write(sample.SerializeToString())
        writer.close()
    except Exception as e:
        print(f'Unexpected Exception in toolbox.save_tfrecord_alt: {e}')


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


def send_email(subject, body, sender_email:str='c49040@gmail.com', receiver_email:str='ucabcbo@ucl.ac.uk', password:str=None):
    import os
    import smtplib
    from email.mime.text import MIMEText

    message = MIMEText(body)
    message['Subject'] = subject
    message['From'] = sender_email
    message['To'] = receiver_email

    if password is None:
        password = os.environ.get('EMAIL_PASSWORD')

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, [receiver_email], message.as_string())
        print('Email sent successfully!')
    except Exception as e:
        print(f'Error sending email: {e}')
    finally:
        server.quit()

