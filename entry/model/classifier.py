#!/usr/bin/env python3
# coding: utf-8

import tensorflow as tf

<<<<<<< HEAD
class foo(object):
    pass
FLAGS = foo()
FLAGS.input_dim = 1 # Scalar
FLAGS.label_dim = 4 
=======

class foo(object):
    pass


FLAGS = foo()
FLAGS.input_dim = 1  # Scalar
FLAGS.label_dim = 4
>>>>>>> 89b6a43314e18ded2ed7ab8f7e2938583d71c218
FLAGS.keep_prob = 0.5
FLAGS.fc_sizes = '30, 10'
'''
# Model Hyperparameters
# FOR SCRIPTING
flags = tf.app.flags
flags.DEFINE_string('fc_sizes', FLAGS.fc_sizes, 'Size of fully connected layers. Use comma separated integers [%s]'%FLAGS.fc_sizes)
flags.DEFINE_float('keep_prob', FLAGS.keep_prob, 'Probability of keeping an activation value after the DROPOUT layer, during training [%f]'%FLAGS.keep_prob)
FLAGS = flags.FLAGS
'''
<<<<<<< HEAD
=======


>>>>>>> 89b6a43314e18ded2ed7ab8f7e2938583d71c218
class model(object):
    '''
    Classify fixed length features, with weighted loss
    classifier will return an object, whose main fields are tensorflow graph nodes.
<<<<<<< HEAD
    
=======

>>>>>>> 89b6a43314e18ded2ed7ab8f7e2938583d71c218
    '''

    def get_layers(self, in_node, fc_sizes, out_dim, keep_prob):

        act_fn = tf.nn.relu
        h = in_node
        keep_prob = tf.placeholder_with_default(keep_prob, [], 'keep_prob')
        for i, size in enumerate(fc_sizes):
            with tf.variable_scope('hidden_layer%d' % i):
                h = tf.contrib.layers.fully_connected(h, size, act_fn)
                print(h)
                h = tf.nn.dropout(h, keep_prob)
<<<<<<< HEAD
        logits = tf.contrib.layers.fully_connected(h, out_dim, None, scope='logits')
        print(logits)
        return logits, keep_prob
    
=======
        logits = tf.contrib.layers.fully_connected(
            h, out_dim, None, scope='logits')
        print(logits)
        return logits, keep_prob

>>>>>>> 89b6a43314e18ded2ed7ab8f7e2938583d71c218
    def get_name(self):
        fc_sizes = [str(s) for s in self.fc_sizes]
        name = '--fc' + '-'.join(fc_sizes)
        return name
<<<<<<< HEAD
    
=======

>>>>>>> 89b6a43314e18ded2ed7ab8f7e2938583d71c218
    def build_graph(self, model_name=None):
        if not model_name:
            model_name = self.get_name()
        self.name = model_name
<<<<<<< HEAD
        
        self.keep_prob = tf.placeholder_with_default(self.def_keep_prob, [], 'keep_prob')
        
=======

        self.keep_prob = tf.placeholder_with_default(
            self.def_keep_prob, [], 'keep_prob')

>>>>>>> 89b6a43314e18ded2ed7ab8f7e2938583d71c218
        self.logits, self.fc_keep_prob = self.get_layers(
            self.input, self.fc_sizes, FLAGS.label_dim, self.keep_prob)

        self.pred = tf.nn.softmax(logits=self.logits, name='predictions')
        print(self.pred)
<<<<<<< HEAD
        
    def __init__(self,
            input_op,
            fc_sizes=[int(s) for s in FLAGS.fc_sizes.split(',')],
            keep_prob=FLAGS.keep_prob,
            model_name=None):
        '''
        Initializer default vales use tf.app.flags
        returns an object, whose main fields are tensorflow graph nodes.
        
=======

    def __init__(self,
                 input_op,
                 fc_sizes=[int(s) for s in FLAGS.fc_sizes.split(',')],
                 keep_prob=FLAGS.keep_prob,
                 model_name=None):
        '''
        Initializer default vales use tf.app.flags
        returns an object, whose main fields are tensorflow graph nodes.

>>>>>>> 89b6a43314e18ded2ed7ab8f7e2938583d71c218
        fc_sizes: [int, [int...]] Size of fc layers connected to the last LSTM cell's output
        keep_prob: float, Probability of keeping a value in DROPOUT layers
        '''
        self.input = input_op
        self.fc_sizes = fc_sizes
        self.def_keep_prob = keep_prob
        self.name = self.get_name()
        with tf.variable_scope('classifier'):
            print('\nFC' + self.name)
            self.build_graph()

<<<<<<< HEAD
def get_logits_and_pred(input_op, return_name=True, **kwargs):
    '''Convenience function for retrieveng 
=======

def get_logits_and_pred(input_op, return_name=True, **kwargs):
    '''Convenience function for retrieveng
>>>>>>> 89b6a43314e18ded2ed7ab8f7e2938583d71c218
    calssifier model graph definition's output'''
    c = model(input_op, **kwargs)
    if return_name:
        return c.logits, c.pred, c.name
    return c.logits, c.pred
<<<<<<< HEAD
    
=======
>>>>>>> 89b6a43314e18ded2ed7ab8f7e2938583d71c218
