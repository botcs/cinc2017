#!/usr/bin/env ipython3
# coding: utf-8

import tensorflow as tf
import numpy as np
import scipy.io as io
import sys


import model.cnn as cnn
import model.rnn as rnn
import model.fourier as fourier
import model.time_domain as time_domain
import model.classifier as classifier

import time
import os
import shutil
import json

# In[2]:

flags = tf.app.flags
flags.DEFINE_integer('gpu', 0, 'device to run on [0]')
flags.DEFINE_string('model_def', './hyperparams/test_model.json', 'load hyperparameters from ["model.json"]')
FLAGS = flags.FLAGS
FLAGS._parse_flags()

os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)
# Do not log on stdout W and I logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# # Model definition
# In[4]:

with open(FLAGS.model_def) as f:
    hyper_param = json.load(f)
    cnn_params = hyper_param['cnn_params']
    rnn_params = hyper_param['rnn_params']
    fc_params = hyper_param['fc_params']
    fourier_params = hyper_param['fourier_params']
    time_domain_params = hyper_param['time_domain_params']
    batch_size = hyper_param['batch_size']
    model_name = os.path.split(FLAGS.model_def)[1]
    # remove file ending
    model_name = model_name[:model_name.find('.json')]

# In[5]:
print('Building model graph...')
tf.reset_default_graph()
batch_size = tf.placeholder_with_default(1, [], name='batch_size')
#input_op, seq_len, label = data.ops.get_batch_producer(
#    batch_size=batch_size, path='./data/train.TFRecord')

input_op = tf.placeholder(tf.float32, [1, None])
seq_len = tf.placeholder(tf.float32, [1])


c = cnn.model(seq_len=seq_len, input_op=input_op, **cnn_params)
r = rnn.get_model(batch_size=batch_size, seq_len=seq_len, input_op=c.output, **rnn_params)
f = fourier.get_output(seq_len=seq_len, input_op=input_op, **fourier_params)
td = time_domain.get_output(seq_len=seq_len, input_op=input_op, **time_domain_params)
concatenated_features=tf.concat([r.last_output, f, td], 1)
fc = classifier.model(input_op=concatenated_features, **fc_params)

logits = fc.logits
pred = fc.pred
print('Building model... done!')

# Load recording
print('Loading record...', end=' ')
dir = "./validation/"
assert len(sys.argv) == 2, "Wrong parameter list in the call of that script."
fname = sys.argv[1]
assert os.path.isfile(dir + fname + ".mat"), "Not existing file: " + fname + ".mat"
data = io.loadmat(dir + fname + '.mat')['val'].astype(np.float32).squeeze()
data -= data.mean()
data /= data.std()
print('done!')



# Run predictor
saver = tf.train.Saver()
label_dict = {0: 'N', 1: 'A', 2: 'O', 3: '~'}
print('Initializin session...', end=' ')
with tf.Session() as sess:
	saver.restore(sess, './ckpt/test_model--cnn64x1024-64x512-32x512-16x256--rnn--steps64--sizes128-64-32-32-16--fc32-16-8-16550')
	print('done!')
	print('Evaluating...', end=' ')
	feed_dict = {
            input_op: [data],
            seq_len: [len(data)]
	}

	output = sess.run(pred, feed_dict)
	print('done!')

	result = label_dict[np.argmax(output)]


# Save result
print('Writing output...', end=' ')
with open('answers.txt', 'a') as file:
    file.write(fname + ',' + result + '\n')
print('done!')
