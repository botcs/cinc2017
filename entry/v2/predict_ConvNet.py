# coding: utf-8

#! /usr/bin/env python3.5
import tensorflow as tf
import numpy as np

import model.cnn as cnn
import model.classifier as classifier

import os, sys
from scipy import io


# # Load recording
dir = '../../validation/'
fname = sys.argv[1]
assert os.path.isfile(dir + fname + ".mat"), "Not existing file: " + dir + fname + ".mat"
data = io.loadmat(dir + fname + '.mat')['val'].astype(np.float32).squeeze()
data -= data.mean()
data /= data.std()


# # Set up predictor

print('Building model graph...')
tf.reset_default_graph()
batch_size = tf.placeholder_with_default(1, [], name='batch_size')

input_op = tf.placeholder(tf.float32, [1, None])
seq_len = tf.placeholder(tf.float32, [1])

cnn_params = {
    'out_dims': [128, 256, 256],
    'kernel_sizes': 64,
    'pool_sizes': 1,
    'model_name': 'CNN'
}
c = cnn.model(seq_len=seq_len, input_op=input_op, **cnn_params)

a = tf.reduce_mean(c.output, axis=1)
fc = classifier.model(input_op=a, fc_sizes=[])

pred = fc.pred


# # Run predictor

label_dict = {0: 'N', 1: 'A', 2: 'O', 3: '~'}
saver = tf.train.Saver()
with tf.Session() as sess:
    print('Sess started')
    coord = tf.train.Coordinator()
    saver.restore(
        sess,
        'model/--cnn128x64-256x64-256x64--fc-46000')
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    print('Evaluating')
    output = sess.run(pred, feed_dict={input_op: [data],
                                       seq_len: [len(data)],
                                       batch_size: 1})

    print('Closing threads')
    coord.request_stop()
    coord.join(threads)

    result = label_dict[np.where(tf.equal(output, tf.reduce_max(output, axis=1)[:, None]).eval()[0])[0][0]]


# # Save result

with open("answers.txt", "a") as file:
    file.write(fname + ',' + result + '\n')

