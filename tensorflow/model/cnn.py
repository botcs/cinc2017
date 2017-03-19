#!/usr/bin/env python3
# coding: utf-8

import tensorflow as tf
import time
import os

def_use_bnorm = True

def_kernel_size = 5

# normal sinus rhythm, atrial fibrillation (AF), an alternative rhythm, or is too noisy
def_keep_prob = 0.5
# Model Hyperparameters

# FOR SCRIPTING
flags = tf.app.flags
flags.DEFINE_string('out_dims', '512, 1024', 'Size of feature map dimensions. Use comma separated integers ["30, 10"]')
flags.DEFINE_string('kernel_sizes', '128, 64', 'Size of convolution kernels. Use comma separated integers ["128, 64"]')
flags.DEFINE_float('keep_prob', def_keep_prob, 'Probability of keeping an activation value after the DROPOUT layer, during training [%f]'%def_keep_prob)
flags.DEFINE_bool('use_bnorm', def_use_bnorm, 'Use batch normalization if True, else use simply biases')
flags.DEFINE_string('model_path', '/tmp/model', 'Logs will be saved to this directory')
FLAGS = flags.FLAGS


# In[18]:

class model(object):
    '''
    Classify fixed length features, with weighted loss
    classifier will return an object, whose main fields are tensorflow graph nodes.
    
    '''
    def get_input(self):
        # [batch_size, seq_len]
        x = tf.placeholder(tf.float32, [None, None], name='input')
        return x

    def get_cnn(self, in_node, out_dims, kernel_sizes, keep_prob, use_bnorm=True):
        '''
        `out_dims`: a list of integers for the featuremap [out_dims1, out_dims2, ...]
        `kernels_sizes`: a single integer or 
            a list of integers [kernel_size1, kernel_size2, ...] which must be the 
            same length as out_dims
        '''
        
        if type(kernel_sizes) is not list:
            kernel_sizes = [kernel_sizes] * len(out_dims)
        
        with tf.variable_scope('conv_module'):
            h = in_node[..., None]
            if use_bnorm:
                biases_initializer = None
                normalizer_fn = tf.contrib.layers.batch_norm
            else:
                biases_initializer = tf.zeros_initializer
                normalizer_fn = None
                
            keep_prob = tf.placeholder_with_default(keep_prob, [], 'keep_prob')
            
            for dim, ker in zip(out_dims, kernel_sizes):
                # does the same as 1d, but with convenience function
                print('\n', h)
                h = tf.contrib.layers.conv2d(h, dim, ker, 
                                             normalizer_fn=normalizer_fn,
                                             biases_initializer=biases_initializer)
                
                h = tf.nn.dropout(h, keep_prob)
                print(h)
        return h
    
    def get_name(self):
        cnn_sizes = ['%dx%d'%(d, k) for d, k in 
            zip(self.out_dims, self.kernel_sizes)]
        
        name = '--cnn' + '-'.join(cnn_sizes)
        name += '---' + time.strftime("%Y-%m-%d")
        return name
    
    def build_graph(self, model_name=None):
        if not model_name:
            model_name = self.get_name()
        self.name = model_name
        
        self.keep_prob = tf.placeholder_with_default(self.def_keep_prob, [], 'keep_prob')
        
        if self.input is None:
            self.input = self.get_input()
        
        self.output = self.get_cnn(
            self.input, self.out_dims, self.kernel_sizes, self.keep_prob)
    
    def get_checkpoint_path(self):
        return os.path.join(self.model_path, self.name)
        
    def __init__(self,
            input=None,
            out_dims=[int(s) for s in FLAGS.out_dims.split(',')],
            kernel_sizes=[int(s) for s in FLAGS.kernel_sizes.split(',')],
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
        self.out_dims = out_dims
        self.kernel_sizes = kernel_sizes
        
        if len(kernel_sizes) == 1:
            kernel_sizes = [kernel_sizes] * len(out_dims)
        
        self.def_keep_prob = keep_prob
        self.model_path = model_path
        with tf.variable_scope('CNN'):
            self.build_graph()

def get_cnn_output(input, **kwargs):
    cnn = model(input, **kwargs)
    return model.output