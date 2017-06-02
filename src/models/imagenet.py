import numpy as np
import tensorflow as tf
from models.ops import *
from functools import reduce
from collections import namedtuple

from cifar10 import ALI as ALI_Base

conv = tf.layers.conv2d
deconv = tf.layers.conv2d_transpose
dropout = tf.layers.dropout

w_init = lambda:tf.truncated_normal_initializer(mean=0.0, stddev=0.001)

slope = 0.01

lr = 0.0001
beta1 = 0.5
beta2 = 0.999

class ALI(ALI_Base):

    def __init__(self, z_dim=256, image_shape=(64, 64, 3), raw_marginal=None):

        self.z_dim = z_dim
        self.image_shape = image_shape

        if raw_marginal is not None:
            marginal = cal_marginal(raw_marginal)
            self.marginal_initializer = tf.constant_initializer(marginal, tf.float32)
        else:
            self.marginal_initializer = None

        self.build()


    def generator_x(self, input_, train=True, reuse=False):

        with tf.variable_scope('generator_x', reuse=reuse) as scope:

            h = input_

            with tf.variable_scope('block0'):
                h = conv(h, filters=2048, kernel_size=(1, 1), strides=(1,1), padding='valid', use_bias=False, kernel_initializer=w_init())
                h = batch_norm(h, training=train)
                h = lrelu(h, slope=slope)

            with tf.variable_scope('block1'):
                h = conv(h, filters=256, kernel_size=(1, 1), strides=(1,1), padding='valid', use_bias=False, kernel_initializer=w_init())
                h = batch_norm(h, training=train)
                h = lrelu(h, slope=slope)


            Block = namedtuple('Block', ['i', 'filter_num', 'kernel_size', 'stride'])
            blocks = [
                Block(2, 256, 4, 1),
                Block(3, 128, 4, 2),
                Block(4, 128, 4, 1),
                Block(5, 64, 4, 2),
                Block(6, 64, 4, 1),
                Block(7, 64, 4, 2),
            ]

            for i, b in enumerate(blocks):
                with tf.variable_scope('block{}'.format(b.i)):
                    h = deconv(h, filters=b.filter_num, kernel_size=(b.kernel_size, b.kernel_size), strides=(b.stride,b.stride), padding='valid', use_bias=False, kernel_initializer=w_init())
                    h = batch_norm(h, training=train)
                    h = lrelu(h, slope=slope)

            with tf.variable_scope('block8'):
                h = conv(h, filters=3, kernel_size=(1, 1), strides=(1,1), padding='valid', use_bias=False, kernel_initializer=w_init())
                h = add_nontied_bias(h, initializer=self.marginal_initializer)
                h = tf.nn.sigmoid(h)

            output = h

        return output


    def generator_z(self, input_, train=True, reuse=False):

        self.input = input_

        with tf.variable_scope('generator_z', reuse=reuse) as scope:

            h = self.input

            Block = namedtuple('Block', ['i', 'filter_num', 'kernel_size', 'stride'])
            blocks = [
                Block(0, 64, 4, 2),
                Block(1, 64, 4, 1),
                Block(2, 128, 4, 2),
                Block(3, 128, 4, 1),
                Block(4, 256, 4, 2),
                Block(5, 256, 4, 1),
                Block(6, 2048, 1, 1),
                Block(7, 2048, 1, 1),
            ]

            for b in blocks:
                with tf.variable_scope('block{}'.format(b.i)):
                    h = conv(h, filters=b.filter_num, kernel_size=(b.kernel_size,b.kernel_size), strides=(b.stride,b.stride), padding='valid', use_bias=False, kernel_initializer=w_init())
                    h = batch_norm(h, training=train)
                    h = lrelu(h, slope=slope)

            with tf.variable_scope('block8'):
                with tf.variable_scope('mu'):
                    h_mu = conv(h, filters=self.z_dim, kernel_size=(1, 1), strides=(1,1), padding='valid', use_bias=False, kernel_initializer=w_init())
                    h_mu = add_nontied_bias(h_mu)
                    self.G_z_mu = h_mu
                with tf.variable_scope('sigma'):
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

                Block = namedtuple('Block', ['i', 'filter_num', 'kernel_size', 'stride', 'is_bn'])
                blocks = [
                    Block(0, 64, 4, 2, False),
                    Block(1, 64, 4, 1, True),
                    Block(2, 128, 4, 2, True),
                    Block(3, 128, 4, 1, True),
                    Block(4, 256, 4, 2, True),
                    Block(5, 256, 4, 1, True),
                ]

                for b in blocks:
                    with tf.variable_scope('block{}'.format(b.i)):
                        h_x = dropout(h_x, rate=0.2, training=train)
                        h_x = conv(h_x, filters=b.filter_num, kernel_size=(b.kernel_size, b.kernel_size), strides=(b.stride,b.stride), padding='valid', use_bias=not b.is_bn, kernel_initializer=w_init())
                        if b.is_bn:
                            h_x = batch_norm(h_x, training=train)
                        h_x = lrelu(h_x, slope=slope)

            with tf.variable_scope('z'):

                with tf.variable_scope('block0'):
                    h_z = dropout(h_z, rate=0.2, training=train)
                    h_z = conv(h_z, filters=2048, kernel_size=(1, 1), strides=(1,1), padding='valid', use_bias=True, kernel_initializer=w_init())
                    h_z = lrelu(h_z, slope=slope)

                with tf.variable_scope('block1'):
                    h_z = dropout(h_z, rate=0.2, training=train)
                    h_z = conv(h_z, filters=2048, kernel_size=(1, 1), strides=(1,1), padding='valid', use_bias=True, kernel_initializer=w_init())
                    h_z = lrelu(h_z, slope=slope)

            with tf.variable_scope('xz'):

                h_xz = tf.concat([h_x, h_z], axis=h_x.get_shape().ndims-1)

                with tf.variable_scope('block0'):
                    h_xz = dropout(h_xz, rate=0.2, training=train)
                    h_xz = conv(h_xz, filters=4096, kernel_size=(1, 1), strides=(1,1), padding='valid', use_bias=True, kernel_initializer=w_init())
                    h_xz = lrelu(h_xz, slope=slope)

                with tf.variable_scope('block1'):
                    h_xz = dropout(h_xz, rate=0.2, training=train)
                    h_xz = conv(h_xz, filters=4096, kernel_size=(1, 1), strides=(1,1), padding='valid', use_bias=True, kernel_initializer=w_init())
                    h_xz = lrelu(h_xz, slope=slope)

                with tf.variable_scope('block2'):
                    h_xz = dropout(h_xz, rate=0.2, training=train)
                    h_xz = conv(h_xz, filters=1, kernel_size=(1, 1), strides=(1,1), padding='valid', use_bias=True, kernel_initializer=w_init())

            logits = h_xz
            output = tf.nn.sigmoid(h_xz)

            return output, logits
