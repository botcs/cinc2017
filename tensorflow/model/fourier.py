#!/usr/bin/env python3
# coding: utf-8

import tensorflow as tf
import numpy as np
from tensorflow.python.client import device_lib

class foo(object):
    pass
FLAGS = foo()
FLAGS.use_magnitude = True #determines whether we use the magnitude or complex valued numbers
FLAGS.featureindices = '33,  27, 17,  38, 134' #selected frequencies


def cpu_fft(input_array):
        return np.fft.fft(input_array)

class model(object):
    '''
    Calculates fourier descriptors from input features and returns selected frequency magnitudes
    '''

    def get_layers(self, seq_len, in_node, featureindices, use_magnitude=True):
        '''
        generate layers of the network- there is only one layer creating the fft descriptor
        '''    
        
        with tf.variable_scope('fft_module'):
            # Converting to NHWC where N is batch and H will be seq_len
            h = in_node[..., None, None]

            
            local_device_protos = device_lib.list_local_devices()
            GpuAvailable=False
            
            for x in local_device_protos:
                if x.device_type == 'GPU':
                    GpuAvailable=True
            if GpuAvailable:
                #if gpu is available use tf.fft- executes on gpu
                comp=tf.cast(h,dtype=tf.complex64)
                f = tf.fft(comp)
            else:
                f=tf.py_func(cpu_fft, [h], tf.complex128)
            if use_magnitude:
                f=tf.abs(f)
            f=tf.cast(f, tf.float32)
            out=f[...,featureindices[0],:,:]
            for ind in range(1,len(featureindices)):
                out=tf.concat( [out,f[...,featureindices[ind],:,:]],1)
            out=tf.reshape(out,[-1,len(featureindices)])
            print(out.get_shape())

        return out, seq_len


    def get_name(self):
        return '--fourier1x'+str(len(self.featureindices))
    
    def build_graph(self, model_name=None):
        if not model_name:
            model_name = self.get_name()
        self.name = model_name
        
        self.output, self.seq_len = self.get_layers(self.seq_len, self.input, self.featureindices, self.use_magnitude)
    
        
    def __init__(self,  seq_len, input_op,
            out_dims=[len(FLAGS.featureindices.split(','))],
            featureindices=[int(s) for s in FLAGS.featureindices.split(',')],
            use_magnitude=FLAGS.use_magnitude,
            model_name=None):
        '''
        Initializer default vales use tf.app.flags
        returns an object, whose main fields are tensorflow graph nodes.
        '''
        
        self.seq_len = seq_len
        self.input = input_op
        self.use_magnitude=use_magnitude
        self.featureindices=featureindices

        self.out_dims = out_dims
        self.name = self.get_name()
        with tf.variable_scope('Fourier'):
            print('\nFourier' + self.name)
            self.build_graph()

def get_output(seq_len, input_op, **kwargs):
    ff = model(seq_len, input_op, **kwargs)
    return ff.output
