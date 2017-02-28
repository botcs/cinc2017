#!/usr/bin/env python3
# coding: utf-8

import tensorflow as tf
import numpy as np

batch_size = 4
time_steps = 250
input_dims = 1 # Scalar
output_dims = 1 # Scalar

# Model Hyperparameters
flags = tf.app.flags

flags.DEFINE_integer('batch_size', batch_size, 'Number of samples per update cycle [%d]'%batch_size)
flags.DEFINE_integer('time_steps', time_steps, 'Length of unrolled LSTM network')
flags.DEFINE_string('rnn_sizes', '30, 30, 30', 'Stacked RNN state size. Use comma separated integers ["10, 10, 10"]')
flags.DEFINE_string('fc_sizes', '30, 10', 'Size of fully connected layers. Use comma separated integers ["30, 10"]')
flags.DEFINE_float('keep_prob', 0.5, 'Probability of keeping an activation value after the DROPOUT layer, during training [0.5]')
flags.DEFINE_string('log_dir', '/tmp/log', 'Logs will be saved to this directory')

FLAGS = flags.FLAGS
FLAGS._parse_flags()


class generator(object):
    '''Generate continous time series using stacked LSTM network.
    
    generator will return an object, whose main fields are tensorflow graph nodes.
    '''
    
    def get_input(self, batch_size):
        with tf.variable_scope('input'):
            x = tf.placeholder(tf.float32, [batch_size, None, input_dims], name='X')
            seq_len = tf.placeholder(tf.int16, [None], name='length')
        return x, seq_len
    
    def get_rnn(self, x, seq_len, rnn_sizes):
        with tf.variable_scope('LSTM'):
            rnn_tuple_state = []

            for size in rnn_sizes:
                shape = [2, batch_size, size]
                ph = tf.placeholder(tf.float32, shape)
                rnn_tuple_state.append(
                    tf.contrib.rnn.LSTMStateTuple(ph[0], ph[1]))
                
            rnn_tuple_state = tuple(rnn_tuple_state)    

            
            with tf.variable_scope('dynamic_wrapper'):
                cells = [tf.contrib.rnn.BasicLSTMCell(size) for size in rnn_sizes]
                multi_cell = tf.contrib.rnn.MultiRNNCell(cells)
                
                outputs, last_states = tf.nn.dynamic_rnn(
                    initial_state=rnn_tuple_state,
                    inputs=x, cell=multi_cell, 
                    sequence_length=seq_len)
        
        return outputs, last_states

    def get_fc(self, rnn_out, fc_sizes):
        with tf.variable_scope('fully_connected'):
            act_fn = tf.nn.relu
            h = rnn_out
            for size in fc_sizes:
                h = tf.contrib.layers.fully_connected(h, size, act_fn)
            preds = tf.contrib.layers.fully_connected(h, output_dims, None)
            
        return preds
        
    def __init__(self,
            batch_size=FLAGS.batch_size,
            time_steps=FLAGS.time_steps,
            rnn_sizes=[int(s) for s in FLAGS.rnn_sizes.split(',')],
            fc_sizes=[int(s) for s in FLAGS.fc_sizes.split(',')],
            keep_prob=FLAGS.keep_prob,
            log_dir=FLAGS.log_dir):
        '''Initializer default vales use tf.app.flags
        returns an object, whose main fields are tensorflow graph nodes.
        
        batch_size: int, Number of samples to process in parallel. Handled as a fix value
        time_steps: int, Maximum number of time steps which the model use for back propagation
        rnn_sizes: [int, [int...]] Size of corresponding LSTM cell's hidden state
        fc_sizes: [int, [int...]] Size of fc layers connected to the last LSTM cell's output
        keep_prob: float, Probability of keeping a value in DROPOUT layers
        log_dir: str, path/to/log/dir
        '''
        
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.rnn_sizes = rnn_sizes
        self.fc_sizes = fc_sizes
        self.keep_prob = keep_prob
        self.log_dir = log_dir
        
        self.x, self.seq_len = self.get_input(batch_size)
        self.rnn_out, _ = self.get_rnn(self.x, self.seq_len, rnn_sizes)
        self.y_ = self.get_fc(self.rnn_out, fc_sizes)
        
        
