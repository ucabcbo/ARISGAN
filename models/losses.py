import tensorflow as tf


def generator_loss(disc_generated_output, gen_output, target, GEN_LOSS:dict, loss_object=None):

    total_gen_loss = None
    gen_losses = dict()

    if GEN_LOSS['gan'] is not None:
        assert loss_object is not None
        gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
        total_gen_loss = (total_gen_loss + (GEN_LOSS['gan'] * gan_loss)) if total_gen_loss is not None else GEN_LOSS['gan'] * gan_loss
        gen_losses['gan'] = gan_loss

    if GEN_LOSS['nll'] is not None:
        EPS = 1e-8
        nll_loss = tf.reduce_mean(-tf.math.log(disc_generated_output + EPS))
        total_gen_loss = (total_gen_loss + (GEN_LOSS['nll'] * nll_loss)) if total_gen_loss is not None else GEN_LOSS['nll'] * nll_loss
        gen_losses['nll'] = nll_loss

    if GEN_LOSS['l1'] is not None:
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
        total_gen_loss = (total_gen_loss + (GEN_LOSS['l1'] * l1_loss)) if total_gen_loss is not None else GEN_LOSS['l1'] * l1_loss
        gen_losses['l1'] = l1_loss

    if GEN_LOSS['rmse'] is not None:
        rmse_loss = tf.reduce_mean((target - gen_output) ** 2) ** 1/2
        total_gen_loss = (total_gen_loss + (GEN_LOSS['rmse'] * rmse_loss)) if total_gen_loss is not None else GEN_LOSS['rmse'] * rmse_loss
        gen_losses['rmse'] = rmse_loss

    if GEN_LOSS['wstein'] is not None:
        wstein_loss = -tf.reduce_mean(disc_generated_output)
        total_gen_loss = (total_gen_loss + (GEN_LOSS['wstein'] * wstein_loss)) if total_gen_loss is not None else GEN_LOSS['wstein'] * wstein_loss
        gen_losses['wstein'] = wstein_loss

    return total_gen_loss, gen_losses


def discriminator_loss(disc_real_output, disc_generated_output, DISC_LOSS:dict, loss_object=None):

    total_disc_loss = None
    disc_losses = dict()

    if DISC_LOSS['bce'] is not None:
        assert loss_object is not None
        real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
        generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
        bce_loss = real_loss + generated_loss
        total_disc_loss = (total_disc_loss + (DISC_LOSS['bce'] * bce_loss)) if total_disc_loss is not None else DISC_LOSS['bce'] * bce_loss
        disc_losses['bce'] = bce_loss

    if DISC_LOSS['nll'] is not None:
        EPS = 1e-8
        nll_loss = tf.reduce_mean(-(tf.math.log(disc_real_output + EPS) + tf.math.log(1 - disc_generated_output + EPS)))
        total_disc_loss = (total_disc_loss + (DISC_LOSS['nll'] * nll_loss)) if total_disc_loss is not None else DISC_LOSS['nll'] * nll_loss
        disc_losses['nll'] = nll_loss

    return total_disc_loss, disc_losses
