import numpy as np
import tensorflow as tf
from models.ops import *
from functools import reduce

conv = tf.layers.conv2d
deconv = tf.layers.conv2d_transpose
dropout = tf.layers.dropout

w_init = lambda:tf.truncated_normal_initializer(mean=0.0, stddev=0.001)

slope = 0.2
num_pieces = 2

lr = 0.0001
beta1 = 0.5
beta2 = 0.999

class ALI:

    def __init__(self, z_dim=64, image_shape=(32, 32, 3), raw_marginal=None):

        self.z_dim = z_dim
        self.image_shape = image_shape

        if raw_marginal is not None:
            marginal = cal_marginal(raw_marginal)
            self.marginal_initializer = tf.constant_initializer(marginal, tf.float32)
        else:
            self.marginal_initializer = None

        self.build()

    def build(self):

        self.input_x = tf.placeholder(tf.float32, (None,) + self.image_shape)
        self.input_z = tf.placeholder(tf.float32, (None, 1, 1, self.z_dim))

        self.train_g = tf.placeholder(tf.bool, [])
        self.train_d = tf.placeholder(tf.bool, [])

        self.G_x = self.generator_x(self.input_z, train=self.train_g, reuse=False)
        self.G_z = self.generator_z(self.input_x, train=self.train_g, reuse=False)

        self.D_G_x, self.D_G_x_logits = self.discriminator(self.G_x, self.input_z, train=self.train_d, reuse=False)
        self.D_G_z, self.D_G_z_logits = self.discriminator(self.input_x, self.G_z, train=self.train_d, reuse=True)

        self.d_loss = tf.reduce_mean(tf.nn.softplus(-self.D_G_z_logits) + tf.nn.softplus(self.D_G_x_logits))
        self.g_loss = tf.reduce_mean(tf.nn.softplus(self.D_G_z_logits) + tf.nn.softplus(-self.D_G_x_logits))

        self.d_vars = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
        self.gx_vars = [var for var in tf.trainable_variables() if 'generator_x' in var.name]
        self.gz_vars = [var for var in tf.trainable_variables() if 'generator_z' in var.name]

        self.d_optim = tf.train.AdamOptimizer(lr, beta1, beta2).minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(lr, beta1, beta2).minimize(self.g_loss, var_list=self.gx_vars+self.gz_vars)

        self.optims = [self.d_optim, self.g_optim]

        print('----- Discriminator Variables -----')
        for var in self.d_vars:
            print(var.name)
        print('----- Discriminator Variables -----')
        print('----- Generator X Variables -----')
        for var in self.gx_vars:
            print(var.name)
        print('----- Generator X Variables -----')
        print('----- Generator Z Variables -----')
        for var in self.gz_vars:
            print(var.name)
        print('----- Generator Z Variables -----')

        resampler = self.generator_x(self.generator_z(self.input_x, train=False, reuse=True), train=False, reuse=True)
        self.resampler = tf.concat([self.input_x, resampler], axis=2)


    def generator_x(self, input_, train=True, reuse=False):

        with tf.variable_scope('generator_x', reuse=reuse) as scope:

            h = input_

            with tf.variable_scope('block1'):
                h = deconv(h, filters=256, kernel_size=(4, 4), strides=(1,1), padding='valid', use_bias=False, kernel_initializer=w_init())
                h = batch_norm(h, training=train)
                h = lrelu(h, slope=slope)

            with tf.variable_scope('block2'):
                h = deconv(h, filters=128, kernel_size=(4, 4), strides=(2,2), padding='valid', use_bias=False, kernel_initializer=w_init())
                h = batch_norm(h, training=train)
                h = lrelu(h, slope=slope)

            with tf.variable_scope('block3'):
                h = deconv(h, filters=64, kernel_size=(4, 4), strides=(1,1), padding='valid', use_bias=False, kernel_initializer=w_init())
                h = batch_norm(h, training=train)
                h = lrelu(h, slope=slope)

            with tf.variable_scope('block4'):
                h = deconv(h, filters=32, kernel_size=(4, 4), strides=(2,2), padding='valid', use_bias=False, kernel_initializer=w_init())
                h = batch_norm(h, training=train)
                h = lrelu(h, slope=slope)

            with tf.variable_scope('block5'):
                h = deconv(h, filters=32, kernel_size=(5, 5), strides=(1,1), padding='valid', use_bias=False, kernel_initializer=w_init())
                h = batch_norm(h, training=train)
                h = lrelu(h, slope=slope)

            with tf.variable_scope('block6'):
                h = conv(h, filters=32, kernel_size=(1, 1), strides=(1,1), padding='valid', use_bias=False, kernel_initializer=w_init())
                h = batch_norm(h, training=train)
                h = lrelu(h, slope=slope)

            with tf.variable_scope('block7'):
                h = conv(h, filters=3, kernel_size=(1, 1), strides=(1,1), padding='valid', use_bias=False, kernel_initializer=w_init())
                h = add_nontied_bias(h, initializer=self.marginal_initializer)
                h = tf.nn.sigmoid(h)

            output = h

        return output


    def generator_z(self, input_, train=True, reuse=False):

        self.input = input_

        with tf.variable_scope('generator_z', reuse=reuse) as scope:

            h = self.input

            with tf.variable_scope('block1'):
                h = conv(h, filters=32, kernel_size=(5, 5), strides=(1,1), padding='valid', use_bias=False, kernel_initializer=w_init())
                h = batch_norm(h, training=train)
                h = lrelu(h, slope=slope)

            with tf.variable_scope('block2'):
                h = conv(h, filters=64, kernel_size=(4, 4), strides=(2,2), padding='valid', use_bias=False, kernel_initializer=w_init())
                h = batch_norm(h, training=train)
                h = lrelu(h, slope=slope)

            with tf.variable_scope('block3'):
                h = conv(h, filters=128, kernel_size=(4, 4), strides=(1,1), padding='valid', use_bias=False, kernel_initializer=w_init())
                h = batch_norm(h, training=train)
                h = lrelu(h, slope=slope)

            with tf.variable_scope('block4'):
                h = conv(h, filters=256, kernel_size=(4, 4), strides=(2,2), padding='valid', use_bias=False, kernel_initializer=w_init())
                h = batch_norm(h, training=train)
                h = lrelu(h, slope=slope)

            with tf.variable_scope('block5'):
                h = conv(h, filters=512, kernel_size=(4, 4), strides=(1,1), padding='valid', use_bias=False, kernel_initializer=w_init())
                h = batch_norm(h, training=train)
                h = lrelu(h, slope=slope)

            with tf.variable_scope('block6'):
                h = conv(h, filters=512, kernel_size=(1, 1), strides=(1,1), padding='valid', use_bias=False, kernel_initializer=w_init())
                h = batch_norm(h, training=train)
                h = lrelu(h, slope=slope)

            with tf.variable_scope('block7_mu'):
                h_mu = conv(h, filters=self.z_dim, kernel_size=(1, 1), strides=(1,1), padding='valid', use_bias=False, kernel_initializer=w_init())
                h_mu = add_nontied_bias(h_mu)
                self.G_z_mu = h_mu

            with tf.variable_scope('block7_sigma'):
                h_sigma = conv(h, filters=self.z_dim, kernel_size=(1, 1), strides=(1,1), padding='valid', use_bias=False, kernel_initializer=w_init())
                h_sigma = add_nontied_bias(h_sigma)
                self.G_z_sigma = h_sigma
                h_sigma = tf.exp(h_sigma)

            rng = tf.random_normal(shape=tf.shape(h_mu))

            output = (rng * h_sigma) + h_mu

        return output

    def discriminator(self, input_x, input_z, train=True, reuse=False):

        with tf.variable_scope('discriminator', reuse=reuse):

            h_x = input_x
            h_z = input_z

            with tf.variable_scope('x'):

                with tf.variable_scope('block1'):
                    h_x = dropout(h_x, rate=0.2, training=train)
                    h_x = conv(h_x, filters=32, kernel_size=(5, 5), strides=(1,1), padding='valid', use_bias=True, kernel_initializer=w_init())
                    h_x = conv_maxout(h_x, num_pieces)

                with tf.variable_scope('block2'):
                    h_x = dropout(h_x, rate=0.5, training=train)
                    h_x = conv(h_x, filters=64, kernel_size=(4, 4), strides=(2,2), padding='valid', use_bias=True, kernel_initializer=w_init())
                    h_x = conv_maxout(h_x, num_pieces)

                with tf.variable_scope('block3'):
                    h_x = dropout(h_x, rate=0.5, training=train)
                    h_x = conv(h_x, filters=128, kernel_size=(4, 4), strides=(1,1), padding='valid', use_bias=True, kernel_initializer=w_init())
                    h_x = conv_maxout(h_x, num_pieces)

                with tf.variable_scope('block4'):
                    h_x = dropout(h_x, rate=0.5, training=train)
                    h_x = conv(h_x, filters=256, kernel_size=(4, 4), strides=(2,2), padding='valid', use_bias=True, kernel_initializer=w_init())
                    h_x = conv_maxout(h_x, num_pieces)

                with tf.variable_scope('block5'):
                    h_x = dropout(h_x, rate=0.5, training=train)
                    h_x = conv(h_x, filters=512, kernel_size=(4, 4), strides=(1,1), padding='valid', use_bias=True, kernel_initializer=w_init())
                    h_x = conv_maxout(h_x, num_pieces)

            with tf.variable_scope('z'):

                with tf.variable_scope('block1'):
                    h_z = dropout(h_z, rate=0.2, training=train)
                    h_z = conv(h_z, filters=512, kernel_size=(1, 1), strides=(1,1), padding='valid', use_bias=False, kernel_initializer=w_init())
                    h_z = conv_maxout(h_z, num_pieces)

                with tf.variable_scope('block2'):
                    h_z = dropout(h_z, rate=0.5, training=train)
                    h_z = conv(h_z, filters=512, kernel_size=(1, 1), strides=(1,1), padding='valid', use_bias=False, kernel_initializer=w_init())
                    h_z = conv_maxout(h_z, num_pieces)

            with tf.variable_scope('xz'):

                h_xz = tf.concat([h_x, h_z], axis=h_x.get_shape().ndims-1)

                with tf.variable_scope('block1'):
                    h_xz = dropout(h_xz, rate=0.5, training=train)
                    h_xz = conv(h_xz, filters=1024, kernel_size=(1, 1), strides=(1,1), padding='valid', use_bias=True, kernel_initializer=w_init())
                    h_xz = conv_maxout(h_xz, num_pieces)

                with tf.variable_scope('block2'):
                    h_xz = dropout(h_xz, rate=0.5, training=train)
                    h_xz = conv(h_xz, filters=1024, kernel_size=(1, 1), strides=(1,1), padding='valid', use_bias=True, kernel_initializer=w_init())
                    h_xz = conv_maxout(h_xz, num_pieces)

                with tf.variable_scope('block3'):
                    h_xz = dropout(h_xz, rate=0.5, training=train)
                    h_xz = conv(h_xz, filters=1, kernel_size=(1, 1), strides=(1,1), padding='valid', use_bias=True, kernel_initializer=w_init())

            logits = h_xz
            output = tf.nn.sigmoid(h_xz)

            return output, logits
