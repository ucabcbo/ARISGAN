import tensorflow as tf

def conv(kernel_size:int, filters:int, stride:int, batchnorm:bool, lrelu:bool, padding:str='same'):
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


def prelu(alpha_initializer:float=0.25):
    result = PReLU(tf.keras.initializers.Constant(alpha_initializer))
    return result


def pixelshuffler(scale_factor:float):
    result = PixelShuffler(scale_factor=scale_factor)
    return result


def residual_block(kernel_size:int=3, filters:int=64, stride:int=1):
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


# From SR-GAN
class PReLU(tf.keras.layers.Layer):
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


# From SR-GAN
# class PixelShuffler(tf.keras.layers.Layer):
#     def __init__(self, scale_factor, **kwargs):
#         super(PixelShuffler, self).__init__(**kwargs)
#         self.scale_factor = scale_factor

#     def call(self, inputs):
#         batch_size, height, width, channels = tf.shape(inputs)
#         # Calculate the output shape
#         new_height = height * self.scale_factor
#         new_width = width * self.scale_factor
#         new_channels = channels // (self.scale_factor ** 2)

#         # Reshape the input tensor
#         reshaped = tf.reshape(inputs, (batch_size, height, width, self.scale_factor, self.scale_factor, new_channels))
#         transposed = tf.transpose(reshaped, (0, 1, 2, 5, 3, 4))

#         # Reshape to the desired output shape
#         output = tf.reshape(transposed, (batch_size, new_height, new_width, new_channels))
#         return output

#     def get_config(self):
#         config = super(PixelShuffler, self).get_config()
#         config.update({'scale_factor': self.scale_factor})
#         return config

from tensorflow.keras import backend as K

class PixelShuffler(tf.keras.layers.Layer):
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