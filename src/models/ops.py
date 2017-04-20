import tensorflow as tf
import numpy as np

import tensorflow as tf

def batch_norm(h, training=True):
    return tf.contrib.layers.batch_norm(h, is_training=training,
                                            updates_collections=None,
                                            decay=0.9,
                                            epsilon=1e-5,
                                            scale=True,
                                            )


def lrelu(x, slope=0.1):
    return tf.maximum(x, slope * x)


def conv_maxout(x, num_pieces=2):

    splited = tf.split(x, num_pieces, axis=3)
    h = tf.stack(splited, axis=-1)
    h = tf.reduce_max(h, axis=4)
    return h

def add_nontied_bias(x, initializer=None):

    with tf.variable_scope('add_nontied_bias'):
        if initializer is not None:
            bias = tf.get_variable('bias', shape=x.get_shape().as_list()[1:], trainable=True, initializer=initializer)
        else:
            bias = tf.get_variable('bias', shape=x.get_shape().as_list()[1:], trainable=True, initializer=tf.zeros_initializer())

        output = x + bias

    return output

def cal_marginal(raw_marginals, eps=1e-7):
    marg = np.clip(raw_marginals.mean(axis=0), eps, 1. - eps)
    # print(marg.shape)
    return np.log(marg / (1. - marg))
