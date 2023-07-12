import sys
import os
sys.path.append(os.getcwd())
import init

import tensorflow as tf

import model.layers as layers


class GAN:
    
    def __init__(self, PATH_LOGS, PATH_CKPT):

        self.generator = self.Generator()
        self.discriminator = self.Discriminator()
        
        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        
        self.summary_writer = tf.summary.create_file_writer(PATH_LOGS)

        self.checkpoint_prefix = os.path.join(PATH_CKPT)
        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            generator=self.generator,
            discriminator=self.discriminator)


    def Generator(self):
        
        inputs = tf.keras.layers.Input(shape=[init.IMG_HEIGHT, init.IMG_WIDTH, 21])

        x = inputs

        x = layers.batchnorm()(x)

        #encoder 1_2
        x = layers.conv(3, 32, 1, lrelu=False, batchnorm=False)(x)    # 256x256x32
        #encoder 2_2
        x = layers.conv(3, 32, 1, lrelu=True, batchnorm=False)(x)     # 256x256x32
        encoder_2_2 = x
        #encoder 3_2
        x = layers.conv(2, 64, 2, lrelu=True, batchnorm=False)(x)     # 128x128x64
        encoder_3_2 = x

        #encoder_4
        x = layers.conv(3, 128, 1, lrelu=True, batchnorm=False)(x)    # 128x128x128
        #encoder_5
        x = layers.conv(3, 128, 1, lrelu=True, batchnorm=False)(x)    # 128x128x128
        encoder5 = x
        #encoder_6
        x = layers.conv(3, 256, 2, lrelu=True, batchnorm=False)(x)    # 64x64x256
        
        #decoder_7
        x = layers.conv(1, 256, 1, lrelu=True, batchnorm=False)(x)    # 64x64x256
        #decoder_8
        x = layers.conv(3, 256, 1, lrelu=True, batchnorm=False)(x)    # 64x64x256

        #decoder_9
        #TODO: lReLU and no droput in the original version
        x = layers.deconv(2, 128, 2, relu=True, batchnorm=False, dropout=0.5)(x)   # 128x128x128

        x = tf.keras.layers.Concatenate(axis=3)([x, encoder5])    # 128x128x256

        #decoder_10
        x = layers.conv(3, 128, 1, lrelu=True, batchnorm=False)(x)    # 128x128x128
        #decoder_11
        #TODO: lReLU and no droput in the original version
        x = layers.deconv(2, 128, 2, relu=True, batchnorm=False, dropout=0.5)(x)       # 256x256x128

        x = tf.keras.layers.Concatenate(axis=3)([x, encoder_2_2])    # 256x256x192

        #decoder_12
        x = layers.conv(3, 64, 1, lrelu=True, batchnorm=False)(x)     # 256x256x64
        #decoder_13
        x = layers.conv(3, 3, 1, lrelu=True, batchnorm=False)(x)    # 256x256xOUTPUT_CHANNELS

        return tf.keras.Model(inputs=inputs, outputs=x)
        

    def Discriminator(self):
        initializer = tf.random_normal_initializer(0., 0.02)

        inp = tf.keras.layers.Input(shape=[init.IMG_HEIGHT, init.IMG_WIDTH, 21], name='input_image')
        tar = tf.keras.layers.Input(shape=[init.IMG_HEIGHT, init.IMG_WIDTH, 3], name='target_image')

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
    

    def generator_loss(self, disc_generated_output, gen_output, target):

        GENWGT = 1
        LMBDA = 100
        EPS = 1e-8

        gen_loss_GAN = tf.reduce_mean(-tf.math.log(disc_generated_output + EPS))
        gen_loss_L1 = tf.reduce_mean(tf.abs(target - gen_output))
        #TODO: check weightings
        total_gen_loss = gen_loss_GAN * GENWGT + gen_loss_L1 * LMBDA
        
        return total_gen_loss, gen_loss_GAN, gen_loss_L1


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

            total_gen_loss, gen_loss_GAN, gen_loss_L1 = self.generator_loss(disc_generated_output, gen_output, target)
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(total_gen_loss,
                                                generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                    discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients,
                                                generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                    discriminator.trainable_variables))

        with summary_writer.as_default():
            tf.summary.scalar('total_gen_loss', total_gen_loss, step=step//1000)
            tf.summary.scalar('gen_loss_GAN', gen_loss_GAN, step=step//1000)
            tf.summary.scalar('gen_loss_L1', gen_loss_L1, step=step//1000)
            tf.summary.scalar('disc_loss', disc_loss, step=step//1000)

    def save(self):
        self.checkpoint.save(file_prefix=self.checkpoint_prefix)

