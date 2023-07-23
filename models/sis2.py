import sys
import os
import tensorflow as tf

sys.path.append(os.getcwd())
import models.layers as layers
import models.losses as losses
from experiment import Experiment

class GAN:
    
    def __init__(self, experiment:Experiment):

        self.exp = experiment

        self.generator = self.Generator()
        self.discriminator = self.Discriminator()
        
        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        
        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        
        self.summary_writer = tf.summary.create_file_writer(self.exp.output.LOGS)
        

    def Generator(self):
        
        inputs = tf.keras.layers.Input(shape=[self.exp.IMG_HEIGHT, self.exp.IMG_WIDTH, self.exp.INPUT_CHANNELS])

        x = inputs                                                          # 256,256,21

        x = layers.sis2_dense_multireceptive_field(x, kernel_sizes=[5, 9, 13], filters=64)
        concat_x = x

        x = layers.sis2_pix2pix(x, self.exp.TILESIZE, 64)

        x = tf.keras.layers.Concatenate()([x, concat_x])

        x = layers.dense_block(x)

        last = layers.conv(1, self.exp.OUTPUT_CHANNELS, 1, batchnorm=False, lrelu=False)(x)

        return tf.keras.Model(inputs=inputs, outputs=last)
        

    def Discriminator(self):

        inp = tf.keras.layers.Input(shape=[self.exp.IMG_HEIGHT, self.exp.IMG_WIDTH, self.exp.INPUT_CHANNELS], name='input_image')
        tar = tf.keras.layers.Input(shape=[self.exp.IMG_HEIGHT, self.exp.IMG_WIDTH, self.exp.OUTPUT_CHANNELS], name='target_image')

        x = tf.keras.layers.Concatenate()([inp, tar])  # 256,256,24

        x = layers.conv(4, 64, 2, lrelu=True, batchnorm=False)(x)   # 128,128,64
        x = layers.conv(4, 128, 2, lrelu=True, batchnorm=True)(x)   # 64,64,128
        x = layers.conv(4, 256, 2, lrelu=True, batchnorm=True)(x)   # 32,32,256

        x = tf.keras.layers.ZeroPadding2D()(x)                      # 34,34,256
        x = layers.conv(4, 512, 1, lrelu=False, batchnorm=False, padding='valid')(x) # 31,31,512
        x = layers.batchnorm()(x)                                   # 31,31,512
        x = layers.lrelu()(x)                                       # 31,31,512
        x = tf.keras.layers.ZeroPadding2D()(x)                      # 30,30,1

        last = layers.conv(4, 1, 1, lrelu=False, batchnorm=False, padding='valid')(x)                   # 30,30,1

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

            total_gen_loss, gen_losses = losses.generator_loss(disc_generated_output, gen_output, target, self.exp.GEN_LOSS, self.loss_object)
            total_disc_loss, disc_losses = losses.discriminator_loss(disc_real_output, disc_generated_output, self.exp.DISC_LOSS, self.loss_object)

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
