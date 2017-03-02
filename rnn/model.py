#!/usr/bin/env python3
# coding: utf-8

import tensorflow as tf
import numpy as np
import time
import os

def_batch_size = 4
def_time_steps = 300
def_input_dim = 1 # Scalar
# normal sinus rhythm, atrial fibrillation (AF), an alternative rhythm, or is too noisy
def_label_dim = 4 
def_keep_prob = 0.5
# Model Hyperparameters
flags = tf.app.flags

flags.DEFINE_integer('batch_size', def_batch_size, 'Number of samples per update cycle [%d]'%def_batch_size)
flags.DEFINE_integer('time_steps', def_time_steps, 'Length of unrolled LSTM network [%d]'%def_time_steps)
flags.DEFINE_string('rnn_sizes', '10, 10, 10', 'Stacked RNN state size. Use comma separated integers ["10, 10, 10"]')
flags.DEFINE_string('fc_sizes', '30, 10', 'Size of fully connected layers. Use comma separated integers ["30, 10"]')
flags.DEFINE_float('keep_prob', def_keep_prob, 'Probability of keeping an activation value after the DROPOUT layer, during training [%f]'%def_keep_prob)
flags.DEFINE_string('model_path', '/tmp/model', 'Logs will be saved to this directory')

FLAGS = flags.FLAGS

class stackedLSTM(object):
    '''
    Build unrolled stacked LSTM with dropout, on top of last hidden RNN variable sized
    fully connected layers.
    
    Convenience functions help fast-proto
    Use this class for describing specific usage by implementing `build_graph`
    '''
    
    def get_input(self, batch_size):
        with tf.variable_scope('input'):
            x = tf.placeholder(tf.float32, [batch_size, None, def_input_dim], name='X')
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
                zero_state = multi_cell.zero_state(batch_size, tf.float32)
        
        return outputs, last_states, keep_prob, rnn_tuple_state, zero_state

    def get_fc(self, in_node, fc_sizes, out_dim, keep_prob):
        with tf.variable_scope('fully_connected'):
            act_fn = tf.nn.relu
            h = in_node
            keep_prob = tf.placeholder_with_default(keep_prob, [], 'keep_prob')
            for size in fc_sizes:
                h = tf.contrib.layers.fully_connected(h, size, act_fn)
                tf.nn.dropout(h, keep_prob)
            preds = tf.contrib.layers.fully_connected(h, out_dim, None)
        return preds, keep_prob
    
    def get_name(self):
        rnn_sizes = [str(s) for s in self.rnn_sizes]
        fc_sizes = [str(s) for s in self.fc_sizes]
        name = str(self.time_steps) 
        name += 'rnn' + '-'.join(rnn_sizes) 
        name += '--fc' + '-'.join(fc_sizes)
        name += '---' + time.strftime("%Y-%m-%d--%H-%M-%S")
        return name
    
    def build_graph(self, model_name=None):
        pass
    
    def get_checkpoint_path(self):
        return os.path.join(self.model_path, self.name)
        
    def __init__(self,
            batch_size=FLAGS.batch_size,
            time_steps=FLAGS.time_steps,
            rnn_sizes=[int(s) for s in FLAGS.rnn_sizes.split(',')],
            fc_sizes=[int(s) for s in FLAGS.fc_sizes.split(',')],
            keep_prob=FLAGS.keep_prob,
            model_path=FLAGS.model_path,
            model_name=None):
        '''
        Initializer default vales use tf.app.flags
        returns an object, whose main fields are tensorflow graph nodes.
        
        batch_size: int, Number of samples to process in parallel. 
        time_steps: int, Maximum number of time steps which the model use for backprop
        rnn_sizes: [int, [int...]] Size of corresponding LSTM cell's hidden state
        fc_sizes: [int, [int...]] Size of fc layers connected to the last LSTM cell's output
        keep_prob: float, Probability of keeping a value in DROPOUT layers
        model_path: str, path/to/model/dir
        '''
        
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.rnn_sizes = rnn_sizes
        self.fc_sizes = fc_sizes
        self.def_keep_prob = keep_prob
        self.model_path = model_path
        
        self.build_graph()

class classifier(stackedLSTM):
    '''
    Classify continous time series, using stacked LSTM network.
    classifier will return an object, whose main fields are tensorflow graph nodes.
    
    '''
    
    def build_graph(self, model_name=None):
        # tf.reset_default_graph()
        if not model_name:
            model_name = self.get_name()
        self.name = model_name
        
        
        self.keep_prob = tf.placeholder_with_default(self.def_keep_prob, [], 'keep_prob')
        
        self.x, self.seq_len = self.get_input(self.batch_size)
        
        #with tf.variable_scope('classifier'):
        
        rnn = self.get_rnn(
            self.x, self.seq_len, self.batch_size, self.rnn_sizes, self.keep_prob)
        self.rnn_outputs, self.rnn_last_states, self.rnn_keep_prob = rnn[:3]
        self.init_state, self.zero_state = rnn[3:]
        
        
        # Last layer's tuple, second element: (c=, h=)
        self.rnn_last_outputs = self.rnn_last_states[-1][1]
        
        self.logits, self.fc_keep_prob = self.get_fc(
            self.rnn_last_outputs, self.fc_sizes, def_label_dim, self.keep_prob)

        self.preds = tf.nn.softmax(logits=self.logits)
        
        
class generator(stackedLSTM):
    '''
    Generate continous time series, using stacked LSTM network.
    generator will return an object, whose main fields are tensorflow graph nodes.
    
    '''
    
    def build_graph(self, model_name=None):
        # tf.reset_default_graph()
        if not model_name:
            model_name = self.get_name()
        self.name = model_name
        
        
        self.keep_prob = tf.placeholder_with_default(self.def_keep_prob, [], 'keep_prob')
        
        self.x, self.seq_len = self.get_input(self.batch_size)
        
        rnn = self.get_rnn(
            self.x, self.seq_len, self.batch_size, self.rnn_sizes, self.keep_prob)
        self.rnn_outputs, self.rnn_last_states, self.rnn_keep_prob = rnn[:3]
        self.init_state, self.zero_state = rnn[3:]

        self.outputs, self.fc_keep_prob = self.get_fc(
            self.rnn_outputs, self.fc_sizes, def_input_dim, self.keep_prob)
  