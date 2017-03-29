#!/usr/bin/env python3
# coding: utf-8

import tensorflow as tf

class foo(object):
    pass
FLAGS = foo()
FLAGS.input_dim = 1
FLAGS.time_steps = 300
FLAGS.keep_prob = 0.5
FLAGS.rnn_sizes = '10, 10, 10'

'''
# Model Hyperparameters
# FOR SCRIPTING
flags = tf.app.flags
flags.DEFINE_integer('time_steps', FLAGS.time_steps, 'Length of unrolled LSTM network [%d]'%FLAGS.time_steps)
flags.DEFINE_string('rnn_sizes', FLAGS.rnn_sizes, 'Stacked RNN state size. Use comma separated integers [%s]'%FLAGS.rnn_sizes)
flags.DEFINE_float('keep_prob', FLAGS.keep_prob, 'Probability of keeping an activation value after the DROPOUT layer, during training [%f]'%FLAGS.keep_prob)

FLAGS = flags.FLAGS
'''

class stackedLSTM(object):
    '''
    Build unrolled stacked LSTM with dropout, on top of last hidden RNN variable sized
    fully connected layers.
    
    Convenience functions help fast-proto
    Use this class for describing specific usage by implementing `build_graph`
    '''
    
    def get_layers(self, batch_size, seq_len, x, rnn_sizes, keep_prob):
        with tf.variable_scope('LSTM'):
            rnn_tuple_state = []

            for size in rnn_sizes:
                shape = [2, None, size]
                ph = tf.placeholder(tf.float32, shape, name='RNN_init_state_placeholder')
                rnn_tuple_state.append(
                    tf.contrib.rnn.LSTMStateTuple(ph[0], ph[1]))
                
            rnn_tuple_state = tuple(rnn_tuple_state)

            cells = [tf.contrib.rnn.BasicLSTMCell(size) for size in rnn_sizes]
            keep_prob = tf.placeholder_with_default(keep_prob, [], 'keep_prob')
            cells = [tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob) 
                     for cell in cells]
            
            multi_cell = tf.contrib.rnn.MultiRNNCell(cells)
            
            with tf.variable_scope('dynamic_wrapper'):
                output, last_state = tf.nn.dynamic_rnn(
                    initial_state=rnn_tuple_state,
                    inputs=x, cell=multi_cell, 
                    sequence_length=seq_len)
                zero_state = multi_cell.zero_state(batch_size, tf.float32)
            print(*last_state, sep='\n')
            
        return output, last_state, keep_prob, rnn_tuple_state, zero_state

    def get_name(self):
        rnn_sizes = [str(s) for s in self.rnn_sizes]
        name = '--rnn--steps'
        name += str(self.time_steps) 
        name += '--sizes' + '-'.join(rnn_sizes) 
        return name
    
    def build_graph(self, model_name=None):
        if not model_name:
            model_name = self.get_name()
        self.name = model_name
        
        self.keep_prob = tf.placeholder_with_default(self.def_keep_prob, [], 'keep_prob')
        
        rnn = self.get_layers(
            self.batch_size, self.seq_len, self.input, self.rnn_sizes, self.keep_prob)
        self.output, self.last_state, self.keep_prob = rnn[:3]
        self.init_state, self.zero_state = rnn[3:]
        
        # Last layer's tuple, second element: (c=, h=)
        self.last_output = self.last_state[-1][1]
        
    
    def __init__(self,
            batch_size=None,
            seq_len=None,
            input_op=None,
            time_steps=FLAGS.time_steps,
            rnn_sizes=[int(s) for s in FLAGS.rnn_sizes.split(',')],
            keep_prob=FLAGS.keep_prob,
            model_name=None):
        '''
        Initializer default vales use tf.app.flags
        returns an object, whose main fields are tensorflow graph nodes.
        
        time_steps: int, Maximum number of time steps which the model use for backprop
        rnn_sizes: [int, [int...]] Size of corresponding LSTM cell's hidden state
        keep_prob: float, Probability of keeping a value in DROPOUT layers
        '''
        self.batch_size = batch_size
        self.input = input_op
        self.seq_len = seq_len
        
        self.time_steps = time_steps
        self.rnn_sizes = rnn_sizes
        self.def_keep_prob = keep_prob
        self.name = self.get_name()
        
        with tf.variable_scope('RNN'):
            print('\nRNN' + self.name)
            self.build_graph()

def get_model(batch_size, seq_len, input_op, **kwargs):
    r = stackedLSTM(batch_size, seq_len, input_op, **kwargs)
    return r