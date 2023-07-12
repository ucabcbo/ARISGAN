import tensorflow as tf

def conv(kernel_size:int, filters:int, stride:int, lrelu:bool, batchnorm:bool, padding:str='same'):
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


def deconv(kernel_size:int, filters:int, stride:int, relu:bool, batchnorm:bool, dropout:float, activation:str=None):
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
    result = tf.keras.layers.LeakyReLU()
    return result


def sigmoid():
    result = tf.keras.layers.Activation('sigmoid')
    return result


def batchnorm():
    result = tf.keras.layers.BatchNormalization()
    return result


def dropout(rate:float):
    result = tf.keras.layers.Dropout(dropout)
    return result
