### Needed Imports
import tensorflow as tf
from keras import backend

# File with repeating functions used for different types of GANs



# Gradient penalty functions are based on the function in https://keras.io/examples/generative/wgan_gp/#wasserstein-gan-wgan-with-gradient-penalty-gp
# gradient_penalty_cwgan was slightly modified to include the labels of each generated/real/interpolated sample
def gradient_penalty(batch_size, real_samples, gen_samples, critic):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.uniform([batch_size, 1], 0.0, 1.0)
        diff = gen_samples - real_samples
        interpolated = real_samples + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # Get the discriminator output for the interpolated samples.
            pred = critic(interpolated, training=True)

        # Calculate the gradients.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1]))
        gp = tf.reduce_mean((norm - 1.0) ** 2) # Average distance to norm 1
        return gp


def gradient_penalty_cwgan(batch_size, real_samples, gen_samples, real_labels, fake_labels, critic):
        """Calculates the gradient penalty for Conditional Wasserstein GAN models.

           This loss is calculated on an interpolated sample and added to the discriminator loss.
        """
        # Get the interpolated sample
        alpha = tf.random.uniform([batch_size, 1], 0.0, 1.0)
        diff = gen_samples - real_samples
        interpolated = real_samples + alpha * diff
        
        # Get the 'interpolated label', labels from real and generated samples are in the same order, 
        # therefore they are the same when the models are built
        if len(real_labels.shape) > 1:
            #dim = real_labels.shape[1]
            diff_labels = real_labels - fake_labels
            inter_labels = real_labels + tf.reshape(alpha, [batch_size, 1]) * diff_labels
        else:
            diff_labels = real_labels - fake_labels
            inter_labels = real_labels + tf.reshape(alpha, [batch_size, ]) * diff_labels
        #alpha = tf.reshape(alpha, [batch_size,1]) # Same alphas but reshapen
        #diff_labels = real_labels - fake_labels
        #print(diff.shape)
        #inter_labels = real_labels + tf.reshape(alpha, [batch_size, ]) * diff_labels
    
        with tf.GradientTape() as gp_tape:
            gp_tape.watch([interpolated, inter_labels])
            # Get the discriminator output for the interpolated samples.
            pred = critic([interpolated, inter_labels], training=True)

        # Calculate the gradients.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1]))
        gp = tf.reduce_mean((norm - 1.0) ** 2) # Average distance to norm 1
        return gp


def gradient_penalty_cwgan_bin(batch_size, real_samples, gen_samples, real_labels, fake_labels, critic):
        """Calculates the gradient penalty for Conditional Wasserstein GAN models for binary / feat. occurrence data.

           This loss is calculated on an interpolated sample and added to the discriminator loss.
        """
        # Get the interpolated sample
        alpha = tf.random.uniform([batch_size, 1, 1], 0.0, 1.0)
        diff = gen_samples - real_samples
        interpolated = real_samples + alpha * diff
        
        # Get the 'interpolated label', labels from real and generated samples are in the same order, 
        # therefore they are the same when the models are built
        if len(real_labels.shape) > 1:
            diff_labels = real_labels - fake_labels
            #print(diff_labels.shape)
            inter_labels = real_labels + tf.reshape(alpha, [batch_size, 1]) * diff_labels
        else:
            alpha = tf.reshape(alpha, [batch_size,]) # Same alphas but reshapen
            diff_labels = real_labels - fake_labels
            inter_labels = real_labels + alpha * diff_labels

        with tf.GradientTape() as gp_tape:
            gp_tape.watch([interpolated, inter_labels])
            # Get the discriminator output for the interpolated samples.
            pred = critic([interpolated, inter_labels], training=True)

        # Calculate the gradients
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # Calculate the norm of the gradients
        norm_grad = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1]))
        grad_pen = tf.reduce_mean((norm_grad - 1.0) ** 2) # Average distance to norm 1
        return grad_pen


### Different Loss functions

# Based on https://www.tensorflow.org/tutorials/generative/dcgan 
# Define the loss function for the discriminator in GAN
def discriminator_loss(real_output, fake_output):
    "Calculates discriminator Binary Cross Entropy loss."
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

# Define the loss function for the generator in GAN
def generator_loss(fake_output):
    "Calculates generator Binary Cross Entropy loss."
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Function from https://machinelearningmastery.com/how-to-code-a-wasserstein-generative-adversarial-network-wgan-from-scratch/
# Implementation of wasserstein loss
def wasserstein_loss(y_true, y_pred):
    "Calculates Wasserstein loss."
    return backend.mean(y_true * y_pred)

# Functions from https://keras.io/examples/generative/wgan_gp/#wasserstein-gan-wgan-with-gradient-penalty-gp
# Define the loss functions for the discriminator
def critic_loss_wgan(real_samples_preds, gen_samples_preds):
    "Calculates Wasserstein loss for the critic only."
    real_loss = tf.reduce_mean(real_samples_preds)
    fake_loss = tf.reduce_mean(gen_samples_preds)
    return fake_loss - real_loss

# Define the loss functions for the generator in WGAN.
def generator_loss_wgan(gen_samples_preds):
    "Calculates Wasserstein loss for the generator only."
    return -tf.reduce_mean(gen_samples_preds)

# Softmax function (a layer) based on https://github.com/igul222/improved_wgan_training/blob/master/gan_language.py
def softmax(logits, len_output):
    "Softmaxx output layer for binary / feature occurrence CWGAN-GP models."
    return tf.nn.softmax(
            tf.reshape(logits,[-1, len_output, 2])
        )