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

        x = inputs                                                          # 256,256,21

        # Only difference to regular pix2pix:
        x = layers.batchnorm()(x)

        x = layers.conv(4, 64, 2, lrelu=True, batchnorm=False)(x)           # 128,128,64
        skip128 = x
        x = layers.conv(4, 128, 2, lrelu=True, batchnorm=True)(x)           # 64,64,128
        skip64 = x
        x = layers.conv(4, 256, 2, lrelu=True, batchnorm=True)(x)           # 32,32,256
        skip32 = x
        x = layers.conv(4, 512, 2, lrelu=True, batchnorm=True)(x)           # 16,16,512
        skip16 = x
        x = layers.conv(4, 512, 2, lrelu=True, batchnorm=True)(x)           # 8,8,512
        skip8 = x
        x = layers.conv(4, 512, 2, lrelu=True, batchnorm=True)(x)           # 4,4,512
        skip4 = x
        x = layers.conv(4, 512, 2, lrelu=True, batchnorm=True)(x)           # 2,2,512
        skip2 = x
        x = layers.conv(4, 512, 2, lrelu=True, batchnorm=True)(x)           # 1,1,512

        x = layers.deconv(4, 512, 2, relu=True, batchnorm=True, dropout=0.5)(x)   # 2,2,512/1024
        x = tf.keras.layers.Concatenate()([x, skip2])

        x = layers.deconv(4, 512, 2, relu=True, batchnorm=True, dropout=0.5)(x)   # 4,4,512/1024
        x = tf.keras.layers.Concatenate()([x, skip4])

        x = layers.deconv(4, 512, 2, relu=True, batchnorm=True, dropout=0.5)(x)   # 8,8,512/1024
        x = tf.keras.layers.Concatenate()([x, skip8])

        x = layers.deconv(4, 512, 2, relu=True, batchnorm=True, dropout=None)(x)  # 16,16,512/1024
        x = tf.keras.layers.Concatenate()([x, skip16])

        x = layers.deconv(4, 256, 2, relu=True, batchnorm=True, dropout=None)(x)  # 32,32,256/512
        x = tf.keras.layers.Concatenate()([x, skip32])

        x = layers.deconv(4, 128, 2, relu=True, batchnorm=True, dropout=None)(x)  # 64,64,128/256
        x = tf.keras.layers.Concatenate()([x, skip64])

        x = layers.deconv(4, 64, 2, relu=True, batchnorm=True, dropout=None)(x)  # 128,128,64/128
        x = tf.keras.layers.Concatenate()([x, skip128])

        last = layers.deconv(4, 3, 2, relu=False, batchnorm=False, dropout=None, activation='tanh')(x)    # 256,256,3

        return tf.keras.Model(inputs=inputs, outputs=last)
        

    def Discriminator(self):
        initializer = tf.random_normal_initializer(0., 0.02)

        inp = tf.keras.layers.Input(shape=[init.IMG_HEIGHT, init.IMG_WIDTH, init.INPUT_CHANNELS], name='input_image')
        tar = tf.keras.layers.Input(shape=[init.IMG_HEIGHT, init.IMG_WIDTH, init.OUTPUT_CHANNELS], name='target_image')

        x = tf.keras.layers.concatenate([inp, tar])  # 256,256,24

        x = layers.conv(4, 64, 2, lrelu=True, batchnorm=False)(x)   # 128,128,64
        x = layers.conv(4, 128, 2, lrelu=True, batchnorm=True)(x)   # 64,64,128
        x = layers.conv(4, 256, 2, lrelu=True, batchnorm=True)(x)   # 32,32,256

        x = tf.keras.layers.ZeroPadding2D()(x)                      # 34,34,256
        x = layers.conv(4, 512, 1, lrelu=False, batchnorm=False, padding='valid')(x) # 31,31,512
        x = layers.batchnorm()(x)                                   # 31,31,512
        x = layers.lrelu()(x)                                       # 31,31,512
        x = tf.keras.layers.ZeroPadding2D()(x)                      # 30,30,1

        x = layers.conv(4, 1, 1, lrelu=False, batchnorm=False, padding='valid')(x)                   # 30,30,1

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
