### VERSION 1

def stylegan_loss(gen_output, target, vgg_features):
    # Compute perceptual loss (L1 distance between VGG features)
    perceptual_loss = tf.reduce_mean(tf.abs(vgg_features(target) - vgg_features(gen_output)))

    # Compute pixel-wise L1 loss
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    # Compute StyleGAN loss
    stylegan_loss = perceptual_loss + 0.01 * l1_loss

    return stylegan_loss, perceptual_loss, l1_loss


def train_step(input_image, target, generator, discriminator, vgg_features,
               generator_optimizer, discriminator_optimizer):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        stylegan_loss_value, perceptual_loss, l1_loss = stylegan_loss(gen_output, target, vgg_features)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(stylegan_loss_value, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

    return stylegan_loss_value, perceptual_loss, l1_loss


### VERSION 2

def stylegan_loss(gen_output, target, vgg_features):
    # Compute perceptual loss (L1 distance between VGG features)
    perceptual_loss = tf.reduce_mean(tf.abs(vgg_features(target) - vgg_features(gen_output)))

    # Compute pixel-wise L1 loss
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    # Compute StyleGAN loss
    stylegan_loss = perceptual_loss + 0.01 * l1_loss

    return stylegan_loss, perceptual_loss, l1_loss


def train_step(input_image, target, generator, discriminator, vgg_features,
               generator_optimizer, discriminator_optimizer):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        stylegan_loss_value, perceptual_loss, l1_loss = stylegan_loss(gen_output, target, vgg_features)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(stylegan_loss_value, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

    return stylegan_loss_value, perceptual_loss, l1_loss
