# coding: utf-8

#! /usr/bin/env python3.5
import tensorflow as tf
import numpy as np

import model.cnn as cnn
import model.classifier as classifier

import os, sys
from scipy import io


# ### Load recording

fname = sys.argv[1]
assert os.path.isfile(fname + ".mat"), "Not existing file: " + fname + ".mat"
data = io.loadmat(fname + '.mat')['val'].astype(np.float32).squeeze()
data -= data.mean()
data /= data.std()


# ### Set up predictor

print('Building model graph...')
tf.reset_default_graph()
batch_size = tf.placeholder_with_default(1, [], name='batch_size')

input_op = tf.placeholder(tf.float32, [1, None])
seq_len = tf.placeholder(tf.float32, [1])

cnn_params = {
    'out_dims' : [32, 64, 64],
    'kernel_sizes' : 64,
    'pool_sizes' : 1
}

c = cnn.model(
    seq_len=seq_len, 
    input_op=input_op, 
    model_name='CNN_block',
    **cnn_params)

RESIDUAL_POOL = 4
residual_input = c.output[..., None, :]

for i in range(1, 4):    
    residual_input = tf.contrib.layers.max_pool2d(
        residual_input, 
        kernel_size=[RESIDUAL_POOL, 1], 
        stride=[RESIDUAL_POOL, 1])
    
    c = cnn.model(
        seq_len=seq_len, 
        input_op=residual_input, 
        residual=True, 
        model_name='CNN_block_%d'%i,
        **cnn_params)
    residual_input += c.output

res_out = tf.squeeze(residual_input, axis=2)
a = tf.reduce_mean(res_out, axis=1)
fc = classifier.model(input_op=a, fc_sizes=[16])

pred = fc.pred


# ### Run predictor

label_dict = {0: 'N', 1: 'A', 2: 'O', 3: '~'}
saver = tf.train.Saver()
with tf.Session() as sess:
    print('Sess started')
    coord = tf.train.Coordinator()
    saver.restore(
        sess,
        'model/pool4--cnn32x64-64x64-64x64--fc16-20000')
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    print('Evaluating')
    output = sess.run(pred, feed_dict={input_op: [data],
                                       seq_len: [len(data)],
                                       batch_size: 1})

    print('Closing threads')
    coord.request_stop()
    coord.join(threads)

    result = label_dict[np.where(tf.equal(output, tf.reduce_max(output, axis=1)[:, None]).eval()[0])[0][0]]



# ### Save result

with open("answers.txt", "a") as file:
    file.write(fname + ',' + result)

