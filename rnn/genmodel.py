#!/usr/bin/env python3
# coding: utf-8

import tensorflow as tf
import numpy as np

def_batch_size = 4
def_time_steps = 250
def_input_dims = 1 # Scalar
def_output_dims = 1 # Scalar
def_keep_prob = 0.5
# Model Hyperparameters
flags = tf.app.flags

flags.DEFINE_integer('batch_size', def_batch_size, 'Number of samples per update cycle [%d]'%def_batch_size)
flags.DEFINE_integer('time_steps', def_time_steps, 'Length of unrolled LSTM network [%d]'%def_time_steps)
flags.DEFINE_string('rnn_sizes', '30, 30, 30', 'Stacked RNN state size. Use comma separated integers ["10, 10, 10"]')
flags.DEFINE_string('fc_sizes', '30, 10', 'Size of fully connected layers. Use comma separated integers ["30, 10"]')
flags.DEFINE_float('keep_prob', def_keep_prob, 'Probability of keeping an activation value after the DROPOUT layer, during training [%f]'%def_keep_prob)
flags.DEFINE_string('log_dir', '/tmp/log', 'Logs will be saved to this directory')

FLAGS = flags.FLAGS
FLAGS._parse_flags()


class generator(object):
    '''Generate continous time series, using stacked LSTM network.
    
    generator will return an object, whose main fields are tensorflow graph nodes.
    '''
    
    def get_input(self, batch_size):
        with tf.variable_scope('input'):
            x = tf.placeholder(tf.float32, [batch_size, None, def_input_dims], name='X')
            seq_len = tf.placeholder(tf.int16, [None], name='length')
        return x, seq_len
    
    def get_rnn(self, x, seq_len, batch_size, rnn_sizes, keep_prob):
        with tf.variable_scope('LSTM'):
            rnn_tuple_state = []

            for size in rnn_sizes:
                shape = [2, batch_size, size]
                ph = tf.placeholder(tf.float32, shape)
                rnn_tuple_state.append(
                    tf.contrib.rnn.LSTMStateTuple(ph[0], ph[1]))
                
            rnn_tuple_state = tuple(rnn_tuple_state)

            cells = [tf.contrib.rnn.BasicLSTMCell(size) for size in rnn_sizes]
            keep_prob = tf.placeholder_with_default(keep_prob, [], 'keep_prob')
            cells = [tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob) 
                     for cell in cells]
            multi_cell = tf.contrib.rnn.MultiRNNCell(cells)
            
            with tf.variable_scope('dynamic_wrapper'):
                outputs, last_states = tf.nn.dynamic_rnn(
                    initial_state=rnn_tuple_state,
                    inputs=x, cell=multi_cell, 
                    sequence_length=seq_len)
        
        return outputs, last_states, keep_prob

    def get_fc(self, rnn_out, fc_sizes, keep_prob):
        with tf.variable_scope('fully_connected'):
            act_fn = tf.nn.relu
            h = rnn_out
            keep_prob = tf.placeholder_with_default(keep_prob, [], 'keep_prob')
            for size in fc_sizes:
                h = tf.contrib.layers.fully_connected(h, size, act_fn)
                tf.nn.dropout(h, keep_prob)
            preds = tf.contrib.layers.fully_connected(h, def_output_dims, None)
        return preds, keep_prob
    
    def rebuild_graph(self):
        # tf.reset_default_graph()
        self.keep_prob = tf.placeholder_with_default(self.def_keep_prob, [], 'keep_prob')
        self.x, self.seq_len = self.get_input(self.batch_size)
        
        self.rnn_outputs, self.rnn_last_states, self.rnn_keep_prob =\
            self.get_rnn(self.x, self.seq_len, self.batch_size, self.rnn_sizes, self.keep_prob)
            
        self.y_, self.fc_keep_prob = self.get_fc(self.rnn_outputs, self.fc_sizes, self.keep_prob)
        
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
        self.def_keep_prob = keep_prob
        self.log_dir = log_dir
        
        self.keep_prob = tf.placeholder_with_default(self.def_keep_prob, [], 'keep_prob')
        self.x, self.seq_len = self.get_input(batch_size)
        
        self.rnn_outputs, self.rnn_last_states, self.rnn_keep_prob =\
            self.get_rnn(self.x, self.seq_len, batch_size, rnn_sizes, self.keep_prob)
            
        self.y_, self.fc_keep_prob = self.get_fc(self.rnn_outputs, fc_sizes, self.keep_prob)
        
        
        
