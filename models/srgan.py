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
        
        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False)

        self.summary_writer = tf.summary.create_file_writer(self.OUTPUT['logs'])

        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            generator=self.generator,
            discriminator=self.discriminator)
    

    def Generator(self):
        
        inputs = tf.keras.layers.Input(shape=[init.IMG_HEIGHT, init.IMG_WIDTH, init.INPUT_CHANNELS])

        x = inputs                                                          # 256,256,21

        # Original paper: stride 1. Introduced 4 to decrease my image size
        x = layers.conv(9, 64, 4, lrelu=False, batchnorm=False)(x)

        x = layers.prelu(alpha_initializer=0.25)(x)
        block0 = x

        x = layers.residual_block()(x)
        x = tf.keras.layers.Add()([x, block0])
        block1 = x

        x = layers.residual_block()(x)
        x = tf.keras.layers.Add()([x, block1])
        block2 = x

        x = layers.residual_block()(x)
        x = tf.keras.layers.Add()([x, block2])
        block3 = x

        x = layers.residual_block()(x)
        x = tf.keras.layers.Add()([x, block3])
        block4 = x

        x = layers.residual_block()(x)
        x = tf.keras.layers.Add()([x, block4])

        x = layers.conv(3, 64, 1, lrelu=False, batchnorm=True)(x)
        x = tf.keras.layers.Add()([x, block0])

        x = layers.conv(3, 256, 1, lrelu=False, batchnorm=False)(x)
        x = layers.pixelshuffler(scale_factor=2)(x)
        x = layers.prelu()(x)

        x = layers.conv(3, 256, 1, lrelu=False, batchnorm=False)(x)
        x = layers.pixelshuffler(scale_factor=2)(x)
        x = layers.prelu()(x)

        last = layers.conv(9, init.OUTPUT_CHANNELS, 1, lrelu=False, batchnorm=False)(x)

        return tf.keras.Model(inputs=inputs, outputs=last)
        

    def Discriminator(self):
        initializer = tf.random_normal_initializer(0., 0.02)

        inp = tf.keras.layers.Input(shape=[init.IMG_HEIGHT, init.IMG_WIDTH, init.INPUT_CHANNELS], name='input_image')
        tar = tf.keras.layers.Input(shape=[init.IMG_HEIGHT, init.IMG_WIDTH, init.OUTPUT_CHANNELS], name='target_image')

        x = tf.keras.layers.concatenate([inp, tar])  # 256,256,24

        x = layers.conv(3, 64, 1, lrelu=True, batchnorm=False)(x)   # 256,256,64

        x = layers.conv(3, 64, 2, lrelu=True, batchnorm=True)(x)    # 128,128,64

        x = layers.conv(3, 64, 2, lrelu=True, batchnorm=True)(x)    # 64,64,64
        x = layers.conv(3, 128, 1, lrelu=True, batchnorm=True)(x)   # 64,64,128
        x = layers.conv(3, 128, 2, lrelu=True, batchnorm=True)(x)   # 32,32,128
        x = layers.conv(3, 256, 1, lrelu=True, batchnorm=True)(x)   # 32,32,256
        x = layers.conv(3, 256, 2, lrelu=True, batchnorm=True)(x)   # 16,16,256
        x = layers.conv(3, 512, 1, lrelu=True, batchnorm=True)(x)   # 16,16,512
        x = layers.conv(3, 512, 2, lrelu=True, batchnorm=True)(x)   # 8,8,512

        x = tf.keras.layers.Dense(1024)(x)
        x = layers.lrelu()(x)
        x = tf.keras.layers.Dense(1)(x)
        last = layers.sigmoid()(x)

        return tf.keras.Model(inputs=[inp, tar], outputs=last)
    

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
