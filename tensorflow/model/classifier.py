#!/usr/bin/env python3
# coding: utf-8

import tensorflow as tf
import numpy as np
import time
import os

def_input_dim = 1 # Scalar
# normal sinus rhythm, atrial fibrillation (AF), an alternative rhythm, or is too noisy
def_label_dim = 4 
def_keep_prob = 0.5
# Model Hyperparameters

# FOR SCRIPTING
flags = tf.app.flags

flags.DEFINE_string('fc_sizes', '30, 10', 'Size of fully connected layers. Use comma separated integers ["30, 10"]')
flags.DEFINE_float('keep_prob', def_keep_prob, 'Probability of keeping an activation value after the DROPOUT layer, during training [%f]'%def_keep_prob)
flags.DEFINE_string('model_path', '/tmp/model', 'Logs will be saved to this directory')

FLAGS = flags.FLAGS

class model(object):
    '''
    Classify fixed length features, with weighted loss
    classifier will return an object, whose main fields are tensorflow graph nodes.
    
    '''
    def get_input(self):
        with tf.variable_scope('input'):
            x = tf.placeholder(tf.float32, [None, def_input_dim], name='input')
        return x

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
        fc_sizes = [str(s) for s in self.fc_sizes]
        name = '--fc' + '-'.join(fc_sizes)
        name += '---' + time.strftime("%Y-%m-%d")
        return name
    
    def build_graph(self, model_name=None):
        if not model_name:
            model_name = self.get_name()
        self.name = model_name
        
        self.keep_prob = tf.placeholder_with_default(self.def_keep_prob, [], 'keep_prob')
        
        if self.input is None:
            self.input = self.get_input()
        
        self.logits, self.fc_keep_prob = self.get_fc(
            self.input, self.fc_sizes, def_label_dim, self.keep_prob)

        self.pred = tf.nn.softmax(logits=self.logits)
    
    def get_checkpoint_path(self):
        return os.path.join(self.model_path, self.name)
        
    def __init__(self,
            input=None,
            fc_sizes=[int(s) for s in FLAGS.fc_sizes.split(',')],
            keep_prob=FLAGS.keep_prob,
            model_path=FLAGS.model_path,
            model_name=None):
        '''
        Initializer default vales use tf.app.flags
        returns an object, whose main fields are tensorflow graph nodes.
        
        fc_sizes: [int, [int...]] Size of fc layers connected to the last LSTM cell's output
        keep_prob: float, Probability of keeping a value in DROPOUT layers
        model_path: str, path/to/model/dir
        '''
        self.input = input
        self.fc_sizes = fc_sizes
        self.def_keep_prob = keep_prob
        self.model_path = model_path
        with tf.variable_scope('classifier'):
            self.build_graph()

def get_logits_and_pred(input, **kwargs):
    '''Convenience function for retrieveng 
    calssifier model graph definition's output'''
    c = model(input, **kwargs)
    return c.logits, c.pred
    