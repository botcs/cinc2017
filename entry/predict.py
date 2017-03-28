#!/usr/bin/python3
# coding: utf-8

import tensorflow as tf
import model.cnn as cnn
import model.rnn as rnn
import model.classifier as classifier
import numpy as np
import scipy.io as io
import sys
import os.path


# Load recording

dir = "./validation/"
assert len(sys.argv) == 2, "Wrong parameter list in the call of that script."
fname = sys.argv[1]
assert os.path.isfile(dir + fname + ".mat"), "Not existing file: " + dir + fname + ".mat"
data = io.loadmat(dir + fname + '.mat')['val'].astype(np.float32).squeeze()
data -= data.mean()
data /= data.std()


# Set up predictor

tf.reset_default_graph()
batch_size = tf.placeholder_with_default(1, [])
input_op = tf.placeholder(tf.float32,[1,None])
seq_len = tf.placeholder(tf.float32,[1])

cnn_params = {
    'out_dims' : [10],
    'kernel_sizes' : 32,
    'pool_sizes' : 10
}
rnn_params = {
    'rnn_sizes' : [10],
    'time_steps' : 100
}
fc_params = {
    'fc_sizes' : []
}

c = cnn.get_output(seq_len=seq_len, input_op=input_op, **cnn_params)
r = rnn.get_model(batch_size=batch_size, seq_len=seq_len, input_op=c, **rnn_params)
_, pred = classifier.get_logits_and_pred(input_op=r.last_output, **fc_params)


# Run predictor

label_dict = {0: 'N', 1: 'A', 2: 'O', 3: '~'}
with tf.Session() as sess:
	print('Sess started')
	coord = tf.train.Coordinator()
	tf.global_variables_initializer().run()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)
	
	init_state = sess.run(r.zero_state)
	print('Evaluating')
	output = sess.run(pred, feed_dict={input_op: [data],
										seq_len: [len(data)],
										r.init_state: init_state,
										batch_size: 1})

	print('Closing threads')
	coord.request_stop()
	coord.join(threads)
	
	result = label_dict[np.where(tf.equal(output, tf.reduce_max(output, axis=1)[:, None]).eval()[0])[0][0]]


# Save result

with open('answers.txt', 'a') as file:
    file.write(fname + ',' + result + '\n')

