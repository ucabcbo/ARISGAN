import tensorflow as tf
import datetime
import os

class Model:
    
    def __init__(self, IMG_WIDTH, IMG_HEIGHT, INPUT_CHANNELS, OUTPUT_CHANNELS, LAMBDA, PATH_LOGS, PATH_CKPT):

        self.name = 'psgan'

        self.IMG_WIDTH = IMG_WIDTH
        self.IMG_HEIGHT = IMG_HEIGHT
        self.INPUT_CHANNELS = INPUT_CHANNELS
        self.OUTPUT_CHANNELS = OUTPUT_CHANNELS
        self.LAMBDA = LAMBDA

        self.generator = Model.Generator(IMG_WIDTH, IMG_HEIGHT, INPUT_CHANNELS, OUTPUT_CHANNELS)
        self.discriminator = Model.Discriminator(IMG_HEIGHT, IMG_WIDTH, INPUT_CHANNELS, OUTPUT_CHANNELS)
        
        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        self.summary_writer = tf.summary.create_file_writer(PATH_LOGS + f'{self.name}_fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

        self.checkpoint_prefix = os.path.join(PATH_CKPT, f'{self.name}_ckpt')
        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            generator=self.generator,
            discriminator=self.discriminator)


    def conv(kernel_size, filters, stride, lrelu):
        
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()

        if lrelu:
            result.add(tf.keras.layers.LeakyReLU())

        result.add(tf.keras.layers.Conv2D(filters,
                                        kernel_size,
                                        strides=stride,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))
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
    

    def strided_conv(kernel_size, out_channels, lrelu):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()

        if lrelu:
            result.add(tf.keras.layers.LeakyReLU())

        result.add(tf.keras.layers.Conv2DTranspose(out_channels,
                                                 kernel_size,
                                                 strides=2,
                                                 padding='same',
                                                 kernel_initializer=initializer,
                                                 use_bias=False))
        return result


    def Generator(IMG_HEIGHT, IMG_WIDTH, INPUT_CHANNELS, OUTPUT_CHANNELS):

        inputs = tf.keras.layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, INPUT_CHANNELS])

        x = inputs

        x = Model.batchnorm()(x)

        #encoder 1_2
        x = Model.conv(3, 32, 1, lrelu=False)(x)    # 256x256x32
        #encoder 2_2
        x = Model.conv(3, 32, 1, lrelu=True)(x)     # 256x256x32
        encoder_2_2 = x
        #encoder 3_2
        x = Model.conv(2, 64, 2, lrelu=True)(x)     # 128x128x64
        encoder_3_2 = x

        #encoder_4
        x = Model.conv(3, 128, 1, lrelu=True)(x)    # 128x128x128
        #encoder_5
        x = Model.conv(3, 128, 1, lrelu=True)(x)    # 128x128x128
        encoder5 = x
        #encoder_6
        x = Model.conv(3, 256, 2, lrelu=True)(x)    # 64x64x256
        
        #decoder_7
        x = Model.conv(1, 256, 1, lrelu=True)(x)    # 64x64x256
        #decoder_8
        x = Model.conv(3, 256, 1, lrelu=True)(x)    # 64x64x256

        #decoder_9
        x = Model.strided_conv(2, 128, lrelu=True)(x)   # 128x128x128

        x = tf.keras.layers.Concatenate(axis=3)([x, encoder5])    # 128x128x256

        #decoder_10
        x = Model.conv(3, 128, 1, lrelu=True)(x)    # 128x128x128
        #decoder_11
        x = Model.strided_conv(2, 128, lrelu=True)(x)       # 256x256x128

        x = tf.keras.layers.Concatenate(axis=3)([x, encoder_2_2])    # 256x256x192

        #decoder_12
        x = Model.conv(3, 64, 1, lrelu=True)(x)     # 256x256x64
        #decoder_13
        x = Model.conv(3, OUTPUT_CHANNELS, 1, lrelu=True)(x)    # 256x256xOUTPUT_CHANNELS

        return tf.keras.Model(inputs=inputs, outputs=x)
        

    def Discriminator(IMG_HEIGHT, IMG_WIDTH, INPUT_CHANNELS, OUTPUT_CHANNELS):
        # n_layers = 3
        # layers = []

        inp = tf.keras.layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, INPUT_CHANNELS], name='input_image')
        tar = tf.keras.layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, OUTPUT_CHANNELS], name='target_image')

        # input = tf.concat([inp, tar], 3)  # 128*128*8
        inputs = tf.keras.layers.Concatenate(axis=3)([inp, tar]) # 256x256x24

        #layer_1
        x = Model.conv(3, 32, 2, lrelu=False)(inputs)    # 128x128x32
        x = Model.lrelu()(x)

        #layer_2
        x = Model.conv(3, 64, 2, lrelu=False)(x)    # 64x64x64
        x = Model.lrelu()(x)

        #layer_3
        x = Model.conv(3, 128, 2, lrelu=False)(x)    # 32x32x128
        x = Model.lrelu()(x)
        
        #layer_4
        x = Model.conv(3, 256, 1, lrelu=False)(x)    # 32x32x256
        x = Model.lrelu()(x)

        #layer_5
        x = Model.conv(3, 1, 1, lrelu=False)(x)    # 32x32x1
        x = Model.sigmoid()(x)

        return tf.keras.Model(inputs=[inp, tar], outputs=x)


    def generator_loss(self, disc_generated_output, gen_output, target):
        EPS = 1e-8
        gen_loss_GAN = tf.reduce_mean(-tf.math.log(disc_generated_output + EPS))
        gen_loss_L1 = tf.reduce_mean(tf.abs(target - gen_output))
        #TODO: check weightings
        gen_loss = gen_loss_GAN * 1.0 + gen_loss_L1 * 100.0
        
        return gen_loss_GAN, gen_loss_L1, gen_loss


    def discriminator_loss(self, disc_real_output, disc_generated_output):
        EPS = 1e-8
        discrim_loss = tf.reduce_mean(-(tf.math.log(disc_real_output + EPS) + tf.math.log(1 - disc_generated_output + EPS)))

        return discrim_loss


    @tf.function
    def train_step(self, input_image, target, step):
        generator = self.generator
        discriminator = self.discriminator
        summary_writer = self.summary_writer

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = generator(input_image, training=True)

            disc_real_output = discriminator([input_image, target], training=True)
            disc_generated_output = discriminator([input_image, gen_output], training=True)

            gen_loss_GAN, gen_loss_L1, gen_loss = self.generator_loss(disc_generated_output, gen_output, target)
            discrim_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_loss_GAN,
                                                generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(discrim_loss,
                                                    discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients,
                                                generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                    discriminator.trainable_variables))

        with summary_writer.as_default():
            tf.summary.scalar('gen_loss_GAN', gen_loss_GAN, step=step//1000)
            tf.summary.scalar('gen_loss_L1', gen_loss_L1, step=step//1000)
            tf.summary.scalar('gen_loss', gen_loss, step=step//1000)
            tf.summary.scalar('disc_loss', discrim_loss, step=step//1000)
    

    def save(self):
        self.checkpoint.save(file_prefix=self.checkpoint_prefix)

