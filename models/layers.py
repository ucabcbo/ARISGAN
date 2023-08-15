# This modules contains layers regularly used in various architectures, for ease of use

import tensorflow as tf
from typing import List

def conv(kernel_size:int, filters:int, stride:int, batchnorm:bool, lrelu:bool, padding:str='same'):
    """Conv2D layers, optionally with batch normalisation and/or leakyReLU
    If any of these is added, packs everything in a `tf.keras.Sequential` layer

    Parameters
    ----------
    kernel_size : int
        Kernel size
    filters : int
        Number of filters
    stride : int
        Stride
    batchnorm : bool
        Batch normalisation yes/np
    lrelu : bool
        LeakyReLU yes/no
    padding : str, optional
        Padding, by default 'same'

    Returns
    -------
    tf.Tensor
        Output tensor
    """    
    initializer = tf.random_normal_initializer(0., 0.02)
    x = tf.keras.layers.Conv2D(filters,
                               kernel_size,
                               strides=stride,
                               padding=padding,
                               kernel_initializer=initializer,
                               use_bias=False)
    if lrelu or batchnorm:
        result = tf.keras.Sequential()
        result.add(x)
        if batchnorm:
            result.add(tf.keras.layers.BatchNormalization())
        if lrelu:
            result.add(tf.keras.layers.LeakyReLU())
    else:
        result = x
    return result


def deconv(kernel_size:int, filters:int, stride:int, batchnorm:bool, relu:bool, dropout:float, activation:str=None):
    """Conv2DTranspose layer, optionally with batch normalisation, ReLU and/or Dropout
    If any of these is added, packs everything in a `tf.keras.Sequential` layer

    Parameters
    ----------
    kernel_size : int
        Kernel size
    filters : int
        Number of filters
    stride : int
        Stride
    batchnorm : bool
        Batch normalisation yes/no
    relu : bool
        ReLU yes/no
    dropout : float
        Droput yes/no
    activation : str, optional
        _description_, by default None

    Returns
    -------
    tf.Tensor
        Output tensor
    """    
    initializer = tf.random_normal_initializer(0., 0.02)
    x = tf.keras.layers.Conv2DTranspose(filters,
                                        kernel_size,
                                        strides=stride,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False,
                                        activation=activation)
    if relu or batchnorm or (dropout is not None):
        result = tf.keras.Sequential()
        result.add(x)
        if batchnorm:
            result.add(tf.keras.layers.BatchNormalization())
        if dropout is not None:
            result.add(tf.keras.layers.Dropout(dropout))
        if relu:
            result.add(tf.keras.layers.ReLU())
    else:
        result = x
    return result


def lrelu():
    """LeakyReLU layer

    Returns
    -------
    tf.Tensor
        Result tensor
    """    
    result = tf.keras.layers.LeakyReLU()
    return result


def sigmoid():
    """Signmoid activation layer

    Returns
    -------
    tf.Tensor
        Result tensor
    """    
    result = tf.keras.layers.Activation('sigmoid')
    return result


def batchnorm():
    """Batch normalisation layer

    Returns
    -------
    tf.Tensor
        Result tensor
    """    
    result = tf.keras.layers.BatchNormalization()
    return result


def dropout(rate:float):
    """Dropout layer

    Parameters
    ----------
    rate : float
        Dropout rate - currently unused

    Returns
    -------
    tf.Tensor
        Result tensor
    """    
    result = tf.keras.layers.Dropout(dropout)
    return result


def relu(max_value=None):
    """ReLU layer

    Parameters
    ----------
    max_value : float, optional
        Max value, by default None

    Returns
    -------
    tf.Tensor
        Result tensor
    """    
    result = tf.keras.layers.ReLU(max_value=max_value)
    return result


def prelu(alpha_initializer:float=0.25):
    """PReLU layer (see class documentation)

    Parameters
    ----------
    alpha_initializer : float, optional
        Initial alpha, by default 0.25

    Returns
    -------
    tf.Tensor
        Result tensor
    """    
    result = PReLU(tf.keras.initializers.Constant(alpha_initializer))
    return result


def pixelshuffler(scale_factor:float):
    """Pixel Shuffler layer (see class documentation)

    Parameters
    ----------
    scale_factor : float
        Scale factor

    Returns
    -------
    tf.Tensor
        Result tensor
    """    
    result = PixelShuffler(scale_factor=scale_factor)
    return result


def multiply_lambda(units:int):
    """Multiplication with learnable lambda value

    Parameters
    ----------
    units : int
        Number of channels

    Returns
    -------
    tf.Tensor
        Result tensor
    """    
    result = tf.keras.layers.Lambda(lambda x: x * tf.keras.backend.random_normal(shape=(1, 1, 1, units)))
    # result = MultiplyLayer(units=units)
    return result


def multiply_layer(units):
    """Tensor multiplication

    Parameters
    ----------
    units : int
        Not used - deprecated

    Returns
    -------
    tf.Tensor
        Result tensor
    """    
    result = tf.keras.layers.Lambda(lambda x: tf.multiply(x[0], x[1]))
    return result


def residual_block_srgan(kernel_size:int=3, filters:int=64, stride:int=1):
    """Residual Block as defined in SR-GAN
    Consists if two Conv2D with Batch normalisations, plus one PReLU layer

    Parameters
    ----------
    kernel_size : int, optional
        Kernel size, by default 3
    filters : int, optional
        Number of filters, by default 64
    stride : int, optional
        Stride, by default 1

    Returns
    -------
    tf.Tensor
        Result tensor
    """
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()

    result.add(tf.keras.layers.Conv2D(filters,
                            kernel_size,
                            strides=stride,
                            padding='same',
                            kernel_initializer=initializer,
                            use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    result.add(prelu())

    result.add(tf.keras.layers.Conv2D(filters,
                            kernel_size,
                            strides=stride,
                            padding='same',
                            kernel_initializer=initializer,
                            use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())

    return result


def residual_block_dsen2(kernel_size:int = 3, filters:int = 64, stride:int = 1):
    """Residual block, as defined by DSen2-WGAN
    Consists of three Conv2D layers, one LeakyReLU and one batch normalisation

    Parameters
    ----------
    kernel_size : int, optional
        Kernel size, by default 3
    filters : int, optional
        Number of filters, by default 64
    stride : int, optional
        Stride, by default 1

    Returns
    -------
    tf.Tensor
        Result tensor
    """
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters,
                                      kernel_size,
                                      strides=stride,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))
    result.add(lrelu())
    result.add(tf.keras.layers.Conv2D(filters,
                                      kernel_size,
                                      strides=stride,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))
    result.add(batchnorm())
    result.add(tf.keras.layers.Conv2D(filters,
                                      kernel_size,
                                      strides=stride,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))
    return result


def dense_block(input_tensor:tf.Tensor, layers:int=5, kernel_size:int=3, filters:int=32, stride:int=1):
    """Dense Block, as defined by TARSGAN (and others)

    Parameters
    ----------
    input_tensor : tf.Tensor
        Input tensor
    layers : int, optional
        Number of layers, by default 5
    kernel_size : int, optional
        Kernel size, by default 3
    filters : int, optional
        Number of filters, by default 32
    stride : int, optional
        Stride, by default 1

    Returns
    -------
    tf.Tensor
        Result tensor
    """
    x = input_tensor
    previous_xs = [input_tensor]
    for _ in range(max(layers-1, 1)):
        x = conv(kernel_size, filters, stride, batchnorm=False, lrelu=True)(x)
        temp = x
        for previous_x in previous_xs:
            x = tf.keras.layers.concatenate([x, previous_x])
        previous_xs.append(temp)

    x = conv(kernel_size, filters, stride, batchnorm=False, lrelu=False)(x)

    return x


def experimental_dense_block(input_tensor:tf.Tensor, layers:int=4, kernel_size:int=3, filters:int=32, stride:int=2):
    """Experimental adjustment of the TARSGAN Dense Block, introducing strided and dilated convolution

    Parameters
    ----------
    input_tensor : tf.Tensor
        Input tensor
    layers : int, optional
        Number of layers, by default 4
    kernel_size : int, optional
        Kernel size, by default 3
    filters : int, optional
        Number of filters, by default 32
    stride : int, optional
        Stride, by default 2

    Returns
    -------
    tf.Tensor
        Result tensor
    """
    x = input_tensor
    previous_ups = [input_tensor]
    previous_downs = []

    for _ in range(max(layers-1, 1)):
        x = conv(kernel_size, filters, stride, batchnorm=False, lrelu=True)(x)
        temp = x
        for previous_down in previous_downs:
            x = tf.keras.layers.concatenate([x, previous_down])
        previous_downs.append(temp)

        x = deconv(kernel_size, filters, stride, batchnorm=False, relu=True, dropout=0.5)(x)
        temp = x
        for previous_up in previous_ups:
            x = tf.keras.layers.concatenate([x, previous_up])
        previous_ups.append(temp)

    x = conv(kernel_size, filters, 1, batchnorm=False, lrelu=False)(x)
    
    return x


# Based on DMNet
def sis2_dense_multireceptive_field_dmnet(input_tensor:tf.Tensor, kernel_sizes:List[int], filters:int):
    """Multireceptive field as defined by DMNet (parallel).
    For serial version, use `sis2_dense_multireceptive_field_srs3`.
    The number of convolution layers is determined by the length of th provided list of kernel sizes.

    Parameters
    ----------
    input_tensor : tf.Tensor
        Input tensor
    kernel_sizes : List[int]
        Kernel sizes as list of integers
    filters : int
        Number of filters

    Returns
    -------
    tf.Tensor
        Result tensor
    """
    results = []
    for kernel_size in kernel_sizes:
        x1 = conv(kernel_size, filters, 1, batchnorm=False, lrelu=False)(input_tensor)
        x1 = relu()(x1)
        results.append(x1)

    x = tf.keras.layers.concatenate(results)

    return x


def sis2_dense_multireceptive_field_srs3(input_tensor:tf.Tensor, kernel_sizes:List[int], filters:int):
    """Multireceptive field as defined by SRS3 (serial).
    For parallel version, use `sis2_dense_multireceptive_field_dmnet`.
    The number of convolution layers is determined by the length of th provided list of kernel sizes.

    Parameters
    ----------
    input_tensor : tf.Tensor
        Input tensor
    kernel_sizes : List[int]
        Kernel sizes for each layer
    filters : int
        Number of filters

    Returns
    -------
    tf.Tensor
        Result tensor
    """
    for_concat = []
    prev_tensor = input_tensor
    for kernel_size in kernel_sizes:
        x1 = conv(kernel_size, filters, 1, batchnorm=False, lrelu=False)(prev_tensor)
        x1 = relu()(x1)
        for_concat.append(x1)
        prev_tensor = x1

    x = tf.keras.layers.concatenate(for_concat)
    return x


def awrrdb_block(input_tensor:tf.Tensor, tilesize:int, filters:int):
    """Single AWRRDB Block, as defined by TARSGAN

    Parameters
    ----------
    input_tensor : tf.Tensor
        Input tensor
    tilesize : int
        Tilesize of the tensor
    filters : int
        Number of filters

    Returns
    -------
    int
        Result tensor
    """
    x = input_tensor
    for i in range(3):
        l1input = x
        x = dense_block(x, filters=filters)
        l1input = multiply_lambda(filters)(l1input)
        x = multiply_lambda(filters)(x)
        l1noise = tf.keras.layers.GaussianNoise(stddev=0.1, batch_input_shape=(None, tilesize, tilesize, 32))(x)
        l1noise = multiply_lambda(filters)(l1noise)
        x = tf.keras.layers.Add()([l1input, x, l1noise])
    x = multiply_lambda(filters)(x)
    y = multiply_lambda(filters)(input_tensor)
    x = tf.keras.layers.Add()([y, x])
    return x


def sis2_pix2pix(input_tensor:tf.Tensor, pxshape:int, output_channels:int, kernel_size:int=4):
    """Entire Pix2Pix network block, as defined by Isola et al.

    Parameters
    ----------
    input_tensor : tf.Tensor
        Input tensor
    pxshape : int
        Tilesize of the input data
    output_channels : int
        Number of output channels
    kernel_size : int, optional
        Kernel size, by default 4

    Returns
    -------
    tf.Tensor
        Output tensor
    """
    x = input_tensor

    if pxshape >= 256:
        x = conv(kernel_size, 64, 2, lrelu=True, batchnorm=False)(x)           # 128,128,64
        skip128 = x

    if pxshape >= 128:
        x = conv(kernel_size, 128, 2, lrelu=True, batchnorm=True)(x)           # 64,64,128
        skip64 = x

    if pxshape >= 64:
        x = conv(kernel_size, 256, 2, lrelu=True, batchnorm=True)(x)           # 32,32,256
        skip32 = x

    x = conv(kernel_size, 512, 2, lrelu=True, batchnorm=True)(x)           # 16,16,512
    skip16 = x
    x = conv(kernel_size, 512, 2, lrelu=True, batchnorm=True)(x)           # 8,8,512
    skip8 = x
    x = conv(kernel_size, 512, 2, lrelu=True, batchnorm=True)(x)           # 4,4,512
    skip4 = x
    x = conv(kernel_size, 512, 2, lrelu=True, batchnorm=True)(x)           # 2,2,512
    skip2 = x
    x = conv(kernel_size, 512, 2, lrelu=True, batchnorm=True)(x)           # 1,1,512

    x = deconv(kernel_size, 512, 2, relu=True, batchnorm=True, dropout=0.5)(x)   # 2,2,512/1024
    x = tf.keras.layers.concatenate([x, skip2])

    x = deconv(kernel_size, 512, 2, relu=True, batchnorm=True, dropout=0.5)(x)   # 4,4,512/1024
    x = tf.keras.layers.concatenate([x, skip4])

    x = deconv(kernel_size, 512, 2, relu=True, batchnorm=True, dropout=0.5)(x)   # 8,8,512/1024
    x = tf.keras.layers.concatenate([x, skip8])

    x = deconv(kernel_size, 512, 2, relu=True, batchnorm=True, dropout=None)(x)  # 16,16,512/1024
    x = tf.keras.layers.concatenate([x, skip16])

    if pxshape >= 64:
        x = deconv(kernel_size, 256, 2, relu=True, batchnorm=True, dropout=None)(x)  # 32,32,256/512
        x = tf.keras.layers.concatenate([x, skip32])

    if pxshape >= 128:
        x = deconv(kernel_size, 128, 2, relu=True, batchnorm=True, dropout=None)(x)  # 64,64,128/256
        x = tf.keras.layers.concatenate([x, skip64])

    if pxshape >= 256:
        x = deconv(kernel_size, 64, 2, relu=True, batchnorm=True, dropout=None)(x)  # 128,128,64/128
        x = tf.keras.layers.concatenate([x, skip128])

    x = deconv(kernel_size, output_channels, 2, relu=False, batchnorm=False, dropout=None, activation='tanh')(x)    # 256,256,3

    return x


class PReLU(tf.keras.layers.Layer):
    """PReLU class, as defined by SR-GAN

    Parameters
    ----------
    tf : tensorflow
        TensorFlow
    """    
    def __init__(self, alpha_initializer, **kwargs):
        super(PReLU, self).__init__(**kwargs)
        self.alpha_initializer = tf.keras.initializers.get(alpha_initializer)

    def build(self, input_shape):
        self.alpha = self.add_weight(name='alpha', shape=(input_shape[-1],),
                                     initializer=self.alpha_initializer,
                                     trainable=True)
        super(PReLU, self).build(input_shape)

    def call(self, inputs):
        pos = tf.nn.relu(inputs)
        neg = self.alpha * (inputs - tf.abs(inputs)) * 0.5
        return pos + neg

    def get_config(self):
        config = super(PReLU, self).get_config()
        config.update({'alpha_initializer': tf.keras.initializers.serialize(self.alpha_initializer)})
        return config


from tensorflow.keras import backend as K

class PixelShuffler(tf.keras.layers.Layer):
    """PixelShuffler class, as defined by SR-GAN

    Parameters
    ----------
    tf : tensorflow
        TensorFlow
    """
    def __init__(self, scale_factor, **kwargs):
        super(PixelShuffler, self).__init__(**kwargs)
        self.scale_factor = scale_factor

    def call(self, inputs):
        shape = K.int_shape(inputs)
        batch_size, height, width, channels = shape
        # Calculate the output shape
        new_height = height * self.scale_factor
        new_width = width * self.scale_factor
        new_channels = channels // (self.scale_factor ** 2)

        # Reshape the input tensor
        reshaped = tf.reshape(tf.cast(inputs, tf.float32), (tf.shape(inputs)[0], height, width, self.scale_factor, self.scale_factor, new_channels))
        transposed = tf.transpose(reshaped, (0, 1, 2, 5, 3, 4))

        # Reshape to the desired output shape
        output = tf.reshape(transposed, (tf.shape(inputs)[0], new_height, new_width, new_channels))
        return output

    def get_config(self):
        config = super(PixelShuffler, self).get_config()
        config.update({'scale_factor': self.scale_factor})
        return config
    
