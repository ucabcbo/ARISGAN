import sys
import os
sys.path.append(os.getcwd())
import init

import tensorflow as tf
from tensorflow.keras.models import Model as KerasModel

class Model:
    
    def __init__(self, LAMBDA, PATH_LOGS, PATH_CKPT):
        self.name = 'pix2pix_vgg'
        self.LAMBDA = LAMBDA

        self.generator = Model.Generator()
        self.discriminator = Model.Discriminator()

        # self.VGG = tf.keras.models.load_model('vgg/ckpt/VGG_0710-1612_b16_e10')
        self.VGG = tf.keras.models.load_model('vgg/ckpt/VGG_0710-1630_b1_e10')
        self.VGG.summary()
        # Set the desired layers for computing features
        feature_layers = ['block3_conv3', 'block4_conv3', 'block5_conv3']
        # Create a new model with selected feature layers as outputs
        self.vgg_features_model = KerasModel(inputs=self.VGG.input, outputs=[self.VGG.get_layer(layer).output for layer in feature_layers])

        self.loss_object = tf.keras.losses.BinaryCrossentropy()
        
        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        self.summary_writer = tf.summary.create_file_writer(PATH_LOGS + f'{self.name}_fit/{init.TIMESTAMP}')

        self.checkpoint_prefix = os.path.join(PATH_CKPT, f'{self.name}_ckpt')
        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            generator=self.generator,
            discriminator=self.discriminator)


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


    def Generator():
        
        inputs = tf.keras.layers.Input(shape=[init.IMG_HEIGHT, init.IMG_WIDTH, 21])
        
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
            3, 4,
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
        

    def Discriminator():
        initializer = tf.random_normal_initializer(0., 0.02)

        inp = tf.keras.layers.Input(shape=[init.IMG_HEIGHT, init.IMG_WIDTH, 21], name='input_image')
        tar = tf.keras.layers.Input(shape=[init.IMG_HEIGHT, init.IMG_WIDTH, 3], name='target_image')

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
    

    def generator_loss(self, disc_generated_output, gen_output, target):
        loss_object = self.loss_object
        gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
        rmse_loss = tf.reduce_mean((target - gen_output) ** 2) ** 1/2
        total_gen_loss = l1_loss + (self.LAMBDA * rmse_loss)
        return total_gen_loss, gan_loss, l1_loss, rmse_loss


    def gen_stylegan_loss(self, gen_output, target):
        # Compute perceptual loss (L1 distance between VGG features)
        perceptual_loss = tf.reduce_mean(tf.abs(self.vgg_features(target) - self.vgg_features(gen_output)))

        # Compute pixel-wise L1 loss
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

        # Compute StyleGAN loss
        stylegan_loss = 0.5 * perceptual_loss + l1_loss

        return stylegan_loss, perceptual_loss, l1_loss


    def discriminator_loss(self, disc_real_output, disc_generated_output):
        loss_object = self.loss_object
        real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
        generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss


    def disc_stylegan_loss(self, disc_real_output, disc_generated_output):
        loss_object = self.loss_object
        real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
        generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

        dro_resized = tf.image.resize(disc_real_output, (256, 256))
        dro_reshaped = tf.tile(dro_resized, [1, 1, 1, 3])        

        dgo_resized = tf.image.resize(disc_real_output, (256, 256))
        dgo_reshaped = tf.tile(dgo_resized, [1, 1, 1, 3])        

        # Compute perceptual loss (L1 distance between VGG features)
        perceptual_loss = tf.reduce_mean(tf.abs(self.vgg_features(dro_reshaped) - self.vgg_features(dgo_reshaped)))

        # Scale the perceptual loss to balance with other losses
        scaled_perceptual_loss = 0.5 * perceptual_loss
        
        # Combine the losses
        total_disc_loss = real_loss + generated_loss + scaled_perceptual_loss
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

            stylegan_loss_value, perceptual_loss, l1_loss = self.gen_stylegan_loss(gen_output, target)
            # gen_total_loss, gen_gan_loss, gen_l1_loss, gen_rmse_loss = self.generator_loss(disc_generated_output, gen_output, target)
            # disc_loss = self.disc_stylegan_loss(disc_real_output, disc_generated_output)
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(stylegan_loss_value,
                                                generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                    discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients,
                                                generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                    discriminator.trainable_variables))

        with summary_writer.as_default():
            tf.summary.scalar('stylegan_loss_value', stylegan_loss_value, step=step//1000)
            tf.summary.scalar('perceptual_loss', perceptual_loss, step=step//1000)
            tf.summary.scalar('l1_loss', l1_loss, step=step//1000)
            tf.summary.scalar('disc_loss', disc_loss, step=step//1000)
    

    def save(self):
        self.checkpoint.save(file_prefix=self.checkpoint_prefix)


    def vgg_features(self, input_tensor):

        # # Preprocess the input tensor for VGG-19
        # preprocessed_input = tf.keras.applications.vgg19.preprocess_input(input_tensor)

        # Compute VGG features for the preprocessed input
        vgg_features_output = self.vgg_features_model(input_tensor)

        # Resize the feature maps to have the same spatial dimensions
        resized_features = []
        target_shape = vgg_features_output[0].shape[1:3]  # Get the spatial dimensions of the first feature map
        for feature_map in vgg_features_output:
            resized_feature = tf.image.resize(feature_map, target_shape, method='bilinear')
            resized_features.append(resized_feature)

        # Concatenate the resized feature maps into a single tensor
        combined_features = tf.concat(resized_features, axis=-1)

        return combined_features
