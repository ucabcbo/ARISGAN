import tensorflow as tf

class Model:
    
    def __init__(self, IMG_WIDTH, IMG_HEIGHT, INPUT_CHANNELS, OUTPUT_CHANNELS):
        self.IMG_WIDTH = IMG_WIDTH
        self.IMG_HEIGHT = IMG_HEIGHT
        self.INPUT_CHANNELS = INPUT_CHANNELS
        self.OUTPUT_CHANNELS = OUTPUT_CHANNELS

    def downsample(filters, size, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02)
        
        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2D(
                filters, size,
                strides=2,
                padding='same',
                kernel_initializer=initializer,
                use_bias=False))
        
        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())
            
        result.add(tf.keras.layers.LeakyReLU())
        
        return result

    def upsample(filters, size, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)
        
        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2DTranspose(
                filters, size,
                strides=2,
                padding='same',
                kernel_initializer=initializer,
                use_bias=False))
            
        result.add(tf.keras.layers.BatchNormalization())
        
        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))
            
        result.add(tf.keras.layers.ReLU())
        
        return result

    def Generator(self):
        
        inputs = tf.keras.layers.Input(shape=[self.IMG_HEIGHT, self.IMG_WIDTH, self.INPUT_CHANNELS])
        
        down_stack = [
            Model.downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
            Model.downsample(128, 4),  # (batch_size, 64, 64, 128)
            Model.downsample(256, 4),  # (batch_size, 32, 32, 256)
            Model.downsample(512, 4),  # (batch_size, 16, 16, 512)
            Model.downsample(512, 4),  # (batch_size, 8, 8, 512)
            Model.downsample(512, 4),  # (batch_size, 4, 4, 512)
            Model.downsample(512, 4),  # (batch_size, 2, 2, 512)
            Model.downsample(512, 4),  # (batch_size, 1, 1, 512)
        ]

        up_stack = [
            Model.upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
            Model.upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
            Model.upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
            Model.upsample(512, 4),  # (batch_size, 16, 16, 1024)
            Model.upsample(256, 4),  # (batch_size, 32, 32, 512)
            Model.upsample(128, 4),  # (batch_size, 64, 64, 256)
            Model.upsample(64, 4),  # (batch_size, 128, 128, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(
            self.OUTPUT_CHANNELS, 4,
            strides=2,
            padding='same',
            kernel_initializer=initializer,
            activation='tanh')  # (batch_size, 256, 256, 3)
            
        x = inputs
        
        # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)
            
        skips = reversed(skips[:-1])
        
        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])
        
        x = last(x)

        return tf.keras.Model(inputs=inputs, outputs=x)
        

    def Discriminator(self):
        initializer = tf.random_normal_initializer(0., 0.02)

        inp = tf.keras.layers.Input(shape=[self.IMG_HEIGHT, self.IMG_WIDTH, self.INPUT_CHANNELS], name='input_image')
        tar = tf.keras.layers.Input(shape=[self.IMG_HEIGHT, self.IMG_WIDTH, self.OUTPUT_CHANNELS], name='target_image')

        x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

        down1 = Model.downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
        down2 = Model.downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
        down3 = Model.downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
        conv = tf.keras.layers.Conv2D(
            512, 4, strides=1,
            kernel_initializer=initializer,
            use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

        last = tf.keras.layers.Conv2D(
            1, 4, strides=1,
            kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

        return tf.keras.Model(inputs=[inp, tar], outputs=last)
    