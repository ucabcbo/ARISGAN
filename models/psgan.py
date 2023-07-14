import sys
import os
sys.path.append(os.getcwd())
import init

import tensorflow as tf

import models.layers as layers
import models.losses as losses


class GAN:
    
    def __init__(self, OUTPUT, PARAMS, GEN_LOSS, DISC_LOSS):

        self.OUTPUT = OUTPUT
        self.PARAMS = PARAMS
        self.GEN_LOSS = GEN_LOSS
        self.DISC_LOSS = DISC_LOSS

        self.generator = self.Generator()
        self.discriminator = self.Discriminator()
        
        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        
        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.summary_writer = tf.summary.create_file_writer(self.OUTPUT['logs'])

        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            generator=self.generator,
            discriminator=self.discriminator)


    def Generator(self):
        
        inputs = tf.keras.layers.Input(shape=[init.IMG_HEIGHT, init.IMG_WIDTH, init.INPUT_CHANNELS])

        x = inputs

        x = layers.batchnorm()(x)

        #encoder 1_2
        x = layers.conv(3, 32, 1, lrelu=True, batchnorm=False)(x)    # 256x256x32
        #encoder 2_2
        x = layers.conv(3, 32, 1, lrelu=False, batchnorm=False)(x)     # 256x256x32
        encoder_2_2 = x
        x = layers.lrelu()(x)

        #encoder 3_2
        x = layers.conv(2, 64, 2, lrelu=True, batchnorm=False)(x)     # 128x128x64

        #encoder_4
        x = layers.conv(3, 128, 1, lrelu=True, batchnorm=False)(x)    # 128x128x128
        #encoder_5
        x = layers.conv(3, 128, 1, lrelu=False, batchnorm=False)(x)    # 128x128x128
        encoder5 = x
        x = layers.lrelu()(x)
        #encoder_6
        x = layers.conv(3, 256, 2, lrelu=True, batchnorm=False)(x)    # 64x64x256
        
        #decoder_7
        x = layers.conv(1, 256, 1, lrelu=True, batchnorm=False)(x)    # 64x64x256
        #decoder_8
        x = layers.conv(3, 256, 1, lrelu=False, batchnorm=False)(x)    # 64x64x256
        decoder8 = x
        x = layers.lrelu()(x)

        #decoder_9
        x = layers.deconv(2, 128, 2, relu=False, batchnorm=False, dropout=None)(x)   # 128x128x128

        # According to the paper, it should be decoder8, but sizes don't match
        x = tf.keras.layers.Concatenate(axis=3)([x, encoder5])    # 128x128x256
        x = layers.lrelu()(x)

        #decoder_10
        x = layers.conv(3, 128, 1, lrelu=True, batchnorm=False)(x)    # 128x128x128
        #decoder_11
        x = layers.deconv(2, 128, 2, relu=False, batchnorm=False, dropout=None)(x)       # 256x256x128

        x = tf.keras.layers.Concatenate(axis=3)([x, encoder_2_2])    # 256x256x192
        x = layers.lrelu()(x)

        #decoder_12
        x = layers.conv(3, 64, 1, lrelu=True, batchnorm=False)(x)     # 256x256x64
        #decoder_13
        x = layers.conv(3, init.OUTPUT_CHANNELS, 1, lrelu=False, batchnorm=False)(x)    # 256x256xOUTPUT_CHANNELS

        last = tf.keras.layers.ReLU()(x)

        return tf.keras.Model(inputs=inputs, outputs=last)
        

    def Discriminator(self):
        initializer = tf.random_normal_initializer(0., 0.02)

        inp = tf.keras.layers.Input(shape=[init.IMG_HEIGHT, init.IMG_WIDTH, init.INPUT_CHANNELS], name='input_image')
        tar = tf.keras.layers.Input(shape=[init.IMG_HEIGHT, init.IMG_WIDTH, init.OUTPUT_CHANNELS], name='target_image')

        # input = tf.concat([inp, tar], 3)  # 128*128*8
        inputs = tf.keras.layers.Concatenate(axis=3)([inp, tar]) # 256x256x24

        #layer_1
        x = layers.conv(3, 32, 2, lrelu=True, batchnorm=False)(inputs)    # 128x128x32

        #layer_2
        x = layers.conv(3, 64, 2, lrelu=True, batchnorm=False)(x)    # 64x64x64

        #layer_3
        x = layers.conv(3, 128, 2, lrelu=True, batchnorm=False)(x)    # 32x32x128
        
        #layer_4
        x = layers.conv(3, 256, 1, lrelu=True, batchnorm=False)(x)    # 32x32x256

        #layer_5
        x = layers.conv(3, 1, 1, lrelu=False, batchnorm=False)(x)    # 32x32x1
        x = layers.sigmoid()(x)

        return tf.keras.Model(inputs=[inp, tar], outputs=x)
    

    @tf.function
    def train_step(self, input_image, target, step):
        generator = self.generator
        discriminator = self.discriminator
        summary_writer = self.summary_writer

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = generator(input_image, training=True)

            disc_real_output = discriminator([input_image, target], training=True)
            disc_generated_output = discriminator([input_image, gen_output], training=True)

            total_gen_loss, gen_losses = losses.generator_loss(disc_generated_output, gen_output, target, self.GEN_LOSS, self.loss_object)
            total_disc_loss, disc_losses = losses.discriminator_loss(disc_real_output, disc_generated_output, self.DISC_LOSS, self.loss_object)

        generator_gradients = gen_tape.gradient(total_gen_loss, generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(total_disc_loss, discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

        with summary_writer.as_default():
            tf.summary.scalar('total_gen_loss', total_gen_loss, step=step//1000)
            for gen_loss in list(gen_losses.keys()):
                tf.summary.scalar(gen_loss, gen_losses[gen_loss], step=step//1000)
            tf.summary.scalar('total_disc_loss', total_disc_loss, step=step//1000)
            for disc_loss in list(disc_losses.keys()):
                tf.summary.scalar(disc_loss, disc_losses[disc_loss], step=step//1000)

    def save(self):
        self.checkpoint.save(file_prefix=self.OUTPUT['ckpt'])

