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


def relu(max_value=None):
    result = tf.keras.layers.ReLU(max_value=max_value)
    return result


def prelu(alpha_initializer:float=0.25):
    result = PReLU(tf.keras.initializers.Constant(alpha_initializer))
    return result


def pixelshuffler(scale_factor:float):
    result = PixelShuffler(scale_factor=scale_factor)
    return result


def multiply_lambda(units:int):
    result = tf.keras.layers.Lambda(lambda x: x * tf.keras.backend.random_normal(shape=(1, 1, 1, units)))
    # result = MultiplyLayer(units=units)
    return result

def multiply_layer(units):
    result = tf.keras.layers.Lambda(lambda x: tf.multiply(x[0], x[1]))
    return result

# From SR-GAN
def residual_block_srgan(kernel_size:int=3, filters:int=64, stride:int=1):
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


# From DSen2
#TODO: kernel sizes and "scaling" not clear
def residual_block_dsen2(kernel_size: int = 3, filters: int = 64, stride: int = 1):
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


# From TARSGAN
def dense_block(input_tensor, kernel_size:int=3, filters:int=32, stride:int=1):

    x = input_tensor
    previous_xs = [input_tensor]
    for _ in range(4):
        x = conv(kernel_size, filters, stride, batchnorm=False, lrelu=True)(x)
        for previous_x in previous_xs:
            x = tf.keras.layers.concatenate([x, previous_x])
        previous_xs.append(x)

    x = conv(kernel_size, filters, stride, batchnorm=False, lrelu=False)(x)

    return x


def experimental_dense_block(input_tensor, kernel_size:int=3, filters:int=32, stride:int=2):

    x = input_tensor
    previous_xs = [input_tensor]
    for _ in range(2):
        x = conv(kernel_size, filters, stride, batchnorm=False, lrelu=True)(x)
        for previous_x in previous_xs:
            x = tf.keras.layers.concatenate([x, previous_x])
        previous_xs.append(x)

    x = conv(kernel_size, filters, 1, batchnorm=False, lrelu=True)
    for previous_x in previous_xs:
        x = tf.keras.layers.concatenate([x, previous_x])
    previous_xs.append(x)

    x = deconv(kernel_size, filters, stride, batchnorm=False, relu=True, dropout=None)
    for previous_x in previous_xs:
        x = tf.keras.layers.concatenate([x, previous_x])
    previous_xs.append(x)

    x = deconv(kernel_size, filters, stride, batchnorm=False, relu=False, dropout=None)
    
    return x


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


from tensorflow.keras import backend as K

#From SR-GAN
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
    
# # From TARSGAN
# class MultiplyLayer(tf.keras.layers.Layer):
#     def __init__(self, units, **kwargs):
#         super(MultiplyLayer, self).__init__(**kwargs)
#         self.units = units

#     def build(self, input_shape):
#         self.kernel = self.add_weight(
#             shape=(1, 1, 1, input_shape[-1]),
#             initializer='glorot_uniform',
#             trainable=True,
#             name='kernel'
#         )
#         super(MultiplyLayer, self).build(input_shape)

#     def call(self, inputs):
#         return inputs * self.kernel

#     def compute_output_shape(self, input_shape):
#         return input_shape

#     def get_config(self):
#         config = super(MultiplyLayer, self).get_config()
#         config.update({'units': self.units})
#         return config
    
# class MultiplyLayer(tf.keras.layers.Layer):
#     def __init__(self, units=1, **kwargs):
#         super(MultiplyLayer, self).__init__(**kwargs)
#         self.units = units

#     def build(self, input_shape):
#         self.kernel = self.add_weight(
#             shape=(input_shape[-1], self.units),
#             initializer='glorot_uniform',
#             trainable=True,
#             name='kernel'
#         )
#         super(MultiplyLayer, self).build(input_shape)

#     def call(self, inputs):
#         return inputs * self.kernel

#     def compute_output_shape(self, input_shape):
#         return input_shape

#     def get_config(self):
#         config = super(MultiplyLayer, self).get_config()
#         config.update({'units': self.units})
#         return config
