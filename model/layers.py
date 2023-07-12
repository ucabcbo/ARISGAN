import tensorflow as tf

def conv(kernel_size:int, filters:int, stride:int, lrelu:bool, batchnorm:bool):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters,
                                    kernel_size,
                                    strides=stride,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))
    if lrelu:
        result.add(tf.keras.layers.LeakyReLU())
    if batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    return result


def deconv(kernel_size:int, out_channels:int, stride:int, lrelu:bool, batchnorm:bool, dropout:int):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(out_channels,
                                                kernel_size,
                                                strides=stride,
                                                padding='same',
                                                kernel_initializer=initializer,
                                                use_bias=False))
    if batchnorm:
       result.add(tf.keras.layers.BatchNormalization())
    if dropout is not None:
        result.add(tf.keras.layers.Dropout(dropout))
    if lrelu:
        result.add(tf.keras.layers.LeakyReLU())
    return result


def lrelu():
    result = tf.keras.layers.LeakyReLU()
    return result


def sigmoid():
    result = tf.keras.layers.Activation('sigmoid')
    return result


def batchnorm():
    result = tf.keras.layers.BatchNormalization()
    return result


def dropout(rate:int):
    result = tf.keras.layers.Dropout(dropout)
    return result
