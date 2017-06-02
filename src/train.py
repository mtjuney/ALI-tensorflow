import tensorflow as tf
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from tqdm import tqdm

from pathlib import Path

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', '-e', default='cifar10')
parser.add_argument('--data_dir', '-d', type=str)
parser.add_argument('--log_dir', '-l', default='log')
parser.add_argument('--checkpoint_dir', '-c', default='checkpoint')
parser.add_argument('--sample_dir', '-s', default='sample')
parser.add_argument('--save_dir', default='out/cifar10')
parser.add_argument('--load_model', default=None, type=str)
args = parser.parse_args()

save_dir = Path(args.save_dir)
log_dir = save_dir / args.log_dir
checkpoint_dir = save_dir / args.checkpoint_dir
sample_dir = save_dir / args.sample_dir

log_dir.mkdir(parents=True, exist_ok=True)
checkpoint_dir.mkdir(parents=True, exist_ok=True)
sample_dir.mkdir(parents=True, exist_ok=True)


def main():

    if args.experiment == 'cifar10':
        num_epoch = 7000
        batch_size = 100
        z_dim = 64
        image_shape = (32, 32, 3)

        from datasets.cifar10 import load_cifar10
        data, data_val = load_cifar10(args.data_dir)

        raw_marginal = data[np.random.permutation(len(data))[:500]]
        sample_x = data_val[:batch_size]

        from models.cifar10 import ALI
        ali = ALI(z_dim=z_dim, image_shape=image_shape, raw_marginal=raw_marginal)
        print('Model : cifar10')
    else:
        raise

    sample_z = np.random.normal(size=(batch_size, 1, 1, z_dim)).astype(np.float32)

    saver = tf.train.Saver()

    data_size = len(data)

    # Summaries

    tf.summary.image('input_x', ali.input_x, max_outputs=10)
    tf.summary.image('sample', ali.G_x, max_outputs=20)
    tf.summary.image('resample', ali.resampler, max_outputs=20)
    tf.summary.scalar('Loss/D', ali.d_loss)
    tf.summary.scalar('Loss/G', ali.g_loss)
    tf.summary.scalar('Acc/D', (tf.reduce_mean(1. - ali.D_G_x) + tf.reduce_mean(ali.D_G_z)) * 0.5)
    tf.summary.scalar('Acc/G', (tf.reduce_mean(ali.D_G_x) + tf.reduce_mean(1. - ali.D_G_z)) * 0.5)

    for var in (ali.gx_vars + ali.gz_vars + ali.d_vars):
        tf.summary.histogram(var.name, var)

    summaries = tf.summary.merge_all()

    with tf.Session() as sess:

        writer = tf.summary.FileWriter(str(log_dir), sess.graph)

        tf.global_variables_initializer().run()

        if args.load_model is not None:
            saver.restore(sess=sess, save_path=args.load_model)

        global_step = 0

        for epoch in range(num_epoch):

            # Train
            perm = np.random.permutation(data_size)
            for idx in tqdm(range(0, len(data), batch_size), desc='[Epoch {}/{}]'.format(epoch, num_epoch)):

                perm_ = perm[idx:idx+batch_size]

                batch_x = data[perm_]
                batch_z = np.random.normal(size=(batch_size, 1, 1, z_dim)).astype(np.float32)

                # Update D and G
                feeds = {ali.input_x: batch_x, ali.input_z: batch_z, ali.train_g: True, ali.train_d: True}
                sess.run(ali.optims, feed_dict=feeds)

                global_step += 1


            # Validation
            feeds = {ali.input_x: sample_x, ali.input_z: sample_z, ali.train_g: False, ali.train_d: False}
            summaries_, samples, resamples = sess.run([summaries, ali.G_x, ali.resampler], feed_dict=feeds)
            writer.add_summary(summaries_, global_step)

            if samples.shape[-1] == 1:
                broad = np.zeros(samples.shape[:-1] + (3,)).astype(np.float32)
                broad += samples
                samples = broad

            utils.save_images(str(sample_dir / 'step_{}.png'.format(global_step)), samples)
            utils.save_images(str(sample_dir / 'reconst_step_{}.png'.format(global_step)), resamples)

            print('[Epoch {}] Save Parameters'.format(epoch))
            saver.save(sess, str(checkpoint_dir / 'model'), global_step=global_step)


if __name__ == '__main__':
    main()
