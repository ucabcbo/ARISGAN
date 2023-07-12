import sys
import os
sys.path.append(os.getcwd())
import init

import tensorflow as tf

import model.layers as layers


class GAN:
    
    def __init__(self, PATH_LOGS, PATH_CKPT, LAMBDA):

        self.name = 'pix2pix'
        
        self.LAMBDA = LAMBDA

        self.generator = self.Generator()
        self.discriminator = self.Discriminator()

        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        
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

        x = inputs                                                          # 256,256,21
        x = layers.conv(4, 64, 2, lrelu=True, batchnorm=False)(x)           # 128,128,64
        skip128 = x
        x = layers.conv(4, 128, 2, lrelu=True, batchnorm=True)(x)           # 64,64,128
        skip64 = x
        # x = layers.conv(4, 256, 2, lrelu=True, batchnorm=True)(x)           # 32,32,256
        # skip32 = x
        # x = layers.conv(4, 512, 2, lrelu=True, batchnorm=True)(x)           # 16,16,512
        # skip16 = x
        # x = layers.conv(4, 512, 2, lrelu=True, batchnorm=True)(x)           # 8,8,512
        # skip8 = x
        # x = layers.conv(4, 512, 2, lrelu=True, batchnorm=True)(x)           # 4,4,512
        # skip4 = x
        # x = layers.conv(4, 512, 2, lrelu=True, batchnorm=True)(x)           # 2,2,512
        # skip2 = x
        # x = layers.conv(4, 512, 2, lrelu=True, batchnorm=True)(x)           # 1,1,512
        # # skip1 = x

        # x = layers.deconv(4, 512, 2, lrelu=True, batchnorm=True, dropout=True)(x)   # 2,2,512/1024
        # x = tf.keras.layers.Concatenate()([x, skip2])

        # x = layers.deconv(4, 512, 2, lrelu=True, batchnorm=True, dropout=True)(x)   # 4,4,512/1024
        # x = tf.keras.layers.Concatenate()([x, skip4])

        # x = layers.deconv(4, 512, 2, lrelu=True, batchnorm=True, dropout=True)(x)   # 8,8,512/1024
        # x = tf.keras.layers.Concatenate()([x, skip8])

        # x = layers.deconv(4, 512, 2, lrelu=True, batchnorm=True, dropout=False)(x)  # 16,16,512/1024
        # x = tf.keras.layers.Concatenate()([x, skip16])

        # x = layers.deconv(4, 256, 2, lrelu=True, batchnorm=True, dropout=False)(x)  # 32,32,256/512
        # x = tf.keras.layers.Concatenate()([x, skip32])

        # x = layers.deconv(4, 128, 2, lrelu=True, batchnorm=True, dropout=False)(x)  # 64,64,128/256
        # x = tf.keras.layers.Concatenate()([x, skip64])

        x = layers.deconv(4, 64, 2, lrelu=True, batchnorm=True, dropout=False)(x)  # 128,128,64/128
        x = tf.keras.layers.Concatenate()([x, skip128])

        initializer = tf.random_normal_initializer(0., 0.02)
        # First "3" is for output channels
        last = tf.keras.layers.Conv2DTranspose(
            3, 4,
            strides=2,
            padding='same',
            kernel_initializer=initializer,
            activation='tanh')  # (batch_size, 256, 256, 3)
        
        x = last(x)
            
        return tf.keras.Model(inputs=inputs, outputs=x)
        

    def Discriminator(self):
        initializer = tf.random_normal_initializer(0., 0.02)

        inp = tf.keras.layers.Input(shape=[init.IMG_HEIGHT, init.IMG_WIDTH, 21], name='input_image')
        tar = tf.keras.layers.Input(shape=[init.IMG_HEIGHT, init.IMG_WIDTH, 3], name='target_image')

        x = tf.keras.layers.concatenate([inp, tar])  # 256,256,24

        x = layers.conv(4, 64, 2, True, False)(x)   # 128,128,64
        x = layers.conv(4, 128, 2, True, True)(x)   # 64,64,128
        x = layers.conv(4, 256, 2, True, True)(x)   # 32,32,256

        x = tf.keras.layers.ZeroPadding2D()(x)      # 34,34,256
        x = layers.conv(4, 512, 1, False, False)(x) # 31,31,512
        x = layers.batchnorm()(x)                   # 31,31,512
        x = layers.lrelu()(x)                       # 31,31,512
        x = tf.keras.layers.ZeroPadding2D(name='HeyHo')(x)      # 30,30,1

        x = layers.conv(4, 1, 1, False, False)(x)   # 30,30,1

        return tf.keras.Model(inputs=[inp, tar], outputs=x)
    

    def generator_loss(self, disc_generated_output, gen_output, target):
        loss_object = self.loss_object
        gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
        rmse_loss = tf.reduce_mean((target - gen_output) ** 2) ** 1/2
        total_gen_loss = gan_loss + (self.LAMBDA * l1_loss)
        return total_gen_loss, gan_loss, l1_loss, rmse_loss


    def discriminator_loss(self, disc_real_output, disc_generated_output):
        loss_object = self.loss_object
        real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
        generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss


    @tf.function
    def train_step(self, input_image, target, step):
        generator = self.generator
        discriminator = self.discriminator
        summary_writer = self.summary_writer

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = generator(input_image, training=True)

            disc_real_output = discriminator([input_image, target], training=True)
            disc_generated_output = discriminator([input_image, gen_output], training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss, gen_rmse_loss = self.generator_loss(disc_generated_output, gen_output, target)
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss,
                                                generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                    discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients,
                                                generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                    discriminator.trainable_variables))

        with summary_writer.as_default():
            tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//1000)
            tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
            tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//1000)
            tf.summary.scalar('gen_rmse_loss', gen_rmse_loss, step=step//1000)
            tf.summary.scalar('disc_loss', disc_loss, step=step//1000)
    

    def save(self):
        self.checkpoint.save(file_prefix=self.checkpoint_prefix)

