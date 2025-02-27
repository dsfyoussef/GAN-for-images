# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 18:41:31 2024

@author: youss
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class Generator(tf.keras.Model):
    def __init__(self, noise_dim):
        super(Generator, self).__init__()
        self.dense1 = layers.Dense(8 * 8 * 256, use_bias=False)
        self.batch_norm1 = layers.BatchNormalization()
        self.leaky_relu1 = layers.LeakyReLU()
        self.reshape = layers.Reshape((8, 8, 256))

        self.conv2d_transpose1 = layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False)
        self.batch_norm2 = layers.BatchNormalization()
        self.leaky_relu2 = layers.LeakyReLU()

        self.conv2d_transpose2 = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)
        self.batch_norm3 = layers.BatchNormalization()
        self.leaky_relu3 = layers.LeakyReLU()

        # Final layer to produce (64, 64, 3) image
        self.conv2d_transpose3 = layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.batch_norm1(x)
        x = self.leaky_relu1(x)
        x = self.reshape(x)
        x = self.conv2d_transpose1(x)
        x = self.batch_norm2(x)
        x = self.leaky_relu2(x)
        x = self.conv2d_transpose2(x)
        x = self.batch_norm3(x)
        x = self.leaky_relu3(x)
        x = self.conv2d_transpose3(x)
        return x







class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Adjust input shape to (64, 64, 3)
        self.conv2d1 = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=(64, 64, 3))
        self.leaky_relu1 = layers.LeakyReLU()
        self.dropout1 = layers.Dropout(0.3)

        self.conv2d2 = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')
        self.leaky_relu2 = layers.LeakyReLU()
        self.dropout2 = layers.Dropout(0.3)

        self.flatten = layers.Flatten()
        self.dense = layers.Dense(1)  # Output single value for real/fake classification

    def call(self, inputs):
        x = self.conv2d1(inputs)
        x = self.leaky_relu1(x)
        x = self.dropout1(x)
        x = self.conv2d2(x)
        x = self.leaky_relu2(x)
        x = self.dropout2(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x


class GAN:
    def __init__(self, noise_dim):
        self.noise_dim = noise_dim
        self.generator = Generator(noise_dim)
        self.discriminator = Discriminator()
        self.cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator_optimizer = keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = keras.optimizers.Adam(1e-4)

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        return real_loss + fake_loss
    def save(self, filepath):
        """Saves the GAN model by saving its generator and discriminator."""
        os.makedirs(filepath, exist_ok=True)
        self.generator.save(os.path.join(filepath, 'generator.keras')) # Added .h5 extension to the filename
        self.discriminator.save(os.path.join(filepath, 'discriminator.keras')) # Added .h5 extension to the filename

    def load(self, filepath):
        """Loads the GAN model by loading its generator and discriminator."""
        self.generator = keras.models.load_model(os.path.join(filepath, 'generator.keras')) # Added .h5 extension to the filename
        self.discriminator = keras.models.load_model(os.path.join(filepath, 'discriminator.keras')) # Added .h5 extension to the filename
