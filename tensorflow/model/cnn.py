#!/usr/bin/env python3
# coding: utf-8

import tensorflow as tf

class foo(object):
    pass
FLAGS = foo()
FLAGS.use_bnorm = True
FLAGS.kernel_size = 5
FLAGS.keep_prob = 0.5
FLAGS.out_dims = '512, 1024, 1024'
FLAGS.kernel_sizes = '128, 64, 32'
FLAGS.pool_sizes = '8, 4, 2'
# normal sinus rhythm, atrial fibrillation (AF), an alternative rhythm, or is too noisy
'''
# Model Hyperparameters
# FOR SCRIPTING
flags = tf.app.flags
flags.DEFINE_string('out_dims', FLAGS.out_dims, 'Size of feature map dimensions. Use comma separated integers [%s]'%FLAGS.out_dims)
flags.DEFINE_string('kernel_sizes', FLAGS.out_dims, 'Size of convolution kernels. Use comma separated integers [%s]'%FLAGS.out_dims)
flags.DEFINE_float('keep_prob', FLAGS.keep_prob, 'Probability of keeping an activation value after the DROPOUT layer, during training [%f]'%FLAGS.keep_prob)
flags.DEFINE_bool('use_bnorm', def_use_bnorm, 'Use batch normalization if True, else use simply biases [%b]'%FLAGS.use_bnorm)
FLAGS = flags.FLAGS
'''

class model(object):
    '''
    Classify fixed length features, with weighted loss
    classifier will return an object, whose main fields are tensorflow graph nodes.
    
    '''

    def get_layers(self, 
        seq_len, in_node, 
        out_dims, kernel_sizes,
        pool_sizes, keep_prob, use_bnorm=True):
        '''
        `out_dims`: a list of integers for the featuremap [out_dims1, out_dims2, ...]
        `kernels_sizes`: a single integer or 
            a list of integers [kernel_size1, kernel_size2, ...] which must be the 
            same length as out_dims
        `pool_sizes`: a single integer or 
            a list of integers [pool_size1, pool_size2, ...] which must be the 
            same length as out_dims
        '''
        
        if type(kernel_sizes) is int:
            kernel_sizes = [kernel_sizes] * len(out_dims)
        if type(pool_sizes) is int:
            pool_sizes = [pool_sizes] * len(out_dims)
        assert len(out_dims) == len(kernel_sizes) == len(pool_sizes)
        
        with tf.variable_scope('conv_module'):
            # Converting to NHWC where N is batch and H will be seq_len
            h = in_node[..., None, None]
            if use_bnorm:
                biases_initializer = None
                normalizer_fn = tf.contrib.layers.batch_norm
            else:
                biases_initializer = tf.zeros_initializer
                normalizer_fn = None
            
            
            keep_prob = tf.placeholder_with_default(keep_prob, [], 'keep_prob')
            
            for dim, ker, pool in zip(out_dims, kernel_sizes, pool_sizes):
                # does the same as 1d, but with convenience function
                h = tf.contrib.layers.conv2d(h, dim, [ker, 1], 
                                             normalizer_fn=normalizer_fn,
                                             biases_initializer=biases_initializer)
                
                h = tf.contrib.layers.max_pool2d(
                    h, kernel_size=[pool, 1], stride=[pool, 1])
                seq_len /= 2
                print(h)
                h = tf.nn.dropout(h, keep_prob)
        return tf.squeeze(h, axis=2), seq_len
    
    def get_name(self):
        cnn_sizes = ['%dx%d'%(d, k) for d, k in 
            zip(self.out_dims, self.kernel_sizes)]
        
        name = '--cnn' + '-'.join(cnn_sizes)
        return name
    
    def build_graph(self, model_name=None):
        if not model_name:
            model_name = self.get_name()
        self.name = model_name
        
        self.keep_prob = tf.placeholder_with_default(self.def_keep_prob, [], 'keep_prob')
        
        self.output, self.seq_len = self.get_layers(
            self.seq_len, self.input, 
            self.out_dims, self.kernel_sizes, 
            self.pool_sizes, self.keep_prob)
    
        
    def __init__(self,
            seq_len,
            input_op,
            out_dims=[int(s) for s in FLAGS.out_dims.split(',')],
            kernel_sizes=[int(s) for s in FLAGS.kernel_sizes.split(',')],
            pool_sizes=[int(s) for s in FLAGS.pool_sizes.split(',')],
            keep_prob=FLAGS.keep_prob,
            model_name=None):
        '''
        Initializer default vales use tf.app.flags
        returns an object, whose main fields are tensorflow graph nodes.
        '''
        
        self.seq_len = seq_len
        self.input = input_op
        self.out_dims = out_dims
        self.kernel_sizes = kernel_sizes
        self.pool_sizes = pool_sizes
        
        if len(kernel_sizes) == 1:
            kernel_sizes = [kernel_sizes] * len(out_dims)
        self.def_keep_prob = keep_prob
        self.name = self.get_name()
        with tf.variable_scope('CNN'):
            print('\nCNN' + self.name)
            self.build_graph()

def get_output(seq_len, input_op, **kwargs):
    cnn = model(seq_len, input_op, **kwargs)
    return cnn.output