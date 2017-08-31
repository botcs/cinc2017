import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
import pickle 

def get_logits(input, num_classes, param_path, res_blocks=3, init_channel=32, in_channel=1):
    #sd = th.load(pytorch_statedict_path)
    #for k, v in sd.items():
    #    print(k, v.size())
    params = pickle.load(open(param_path, 'rb'))
    paramgen = iter(params)
    def init(*args, do_assert=True):
        p = next(paramgen)
        if do_assert:
            assert p.shape == args, (p.shape, args)
        return p

    def selu(x):
        with ops.name_scope('elu') as scope:
            alpha = 1.6732632423543772848170429916717
            scale = 1.0507009873554804934193349852946
            return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))
    def BatchNorm(input, channel=8):
        with tf.variable_scope('BatchNorm'):
            weight = tf.Variable(init(channel), name='weight')
            bias = tf.Variable(init(channel), name='bias')
            mean = tf.Variable(init(channel), name='running_mean')
            var = tf.Variable(init(channel), name='running_var')
            return tf.nn.batch_normalization(input, mean, var, bias, weight, 1e-05)

    def Conv1d(input, in_channel, out_channel, kernel_size, dilation=1, bias=False):
        with tf.variable_scope('Conv1d'):
            w = tf.Variable(init(kernel_size, in_channel, out_channel), name='weight')
            if dilation > 1:
                w = tf.expand_dims(w, 0)
                x = tf.expand_dims(input, 1)
                x = tf.nn.atrous_conv2d(x, w, dilation, 'SAME')
                x = tf.squeeze(x, 1)
            else:
                x = tf.nn.conv1d(input, w, 1, 'SAME')
            if bias:
                b = tf.Variable(init(out_channel), name='bias')
                x = x + b
        return x
    def MaxPool1d(input):
        with tf.variable_scope('MaxPool1d'):
            x = tf.expand_dims(input, 1)
            x = tf.nn.max_pool(x, [1, 1, 2, 1], [1, 1, 2, 1], 'SAME')
            x = tf.squeeze(x, 1)
        return x

    def Encoder(input, init_channel, in_channel=in_channel):
        def DownSampleBlock(input, in_channel, out_channel):
            with tf.variable_scope('DownSampleBlock'):
                x = Conv1d(input, in_channel, out_channel, 7, bias=True)
                x = BatchNorm(x, out_channel)
                x = selu(x)
                x = MaxPool1d(x)
            return x    
        with tf.variable_scope('Encoder'):
            x = DownSampleBlock(input, in_channel, init_channel)
            x = DownSampleBlock(x, init_channel, init_channel*2)
            x = DownSampleBlock(x, init_channel*2, init_channel*4)
            x = DownSampleBlock(x, init_channel*4, init_channel*8)
        return x
    def DilatedBlock(input, channel=8, kernel_size=9, dilation=2):
        # No change in # of channels -> identity mapping
        with tf.variable_scope('DilatedBlock'):
            x = BatchNorm(input, channel)
            x = Conv1d(x, channel, channel, kernel_size)
            x = selu(x)
            x = BatchNorm(x, channel)
            x = Conv1d(x, channel, channel, kernel_size, dilation)
            x = selu(x)
        return x + input    
    def ResNet(input, channel, res_blocks=res_blocks):
        with tf.variable_scope('ResNet'):
            x = DilatedBlock(input, channel)
            for _ in range(res_blocks-1):
                x = DilatedBlock(x, channel)
        return x

    def Features(input, init_channel):
        x = Encoder(input, init_channel)
        x = ResNet(x, init_channel*8)
        return x

    def NET(input, init_channel=init_channel, num_classes=num_classes):
        x = Features(input, init_channel)
        lens = tf.shape(input, 'lens')[1]
        lens = tf.cast(lens, tf.float32)
        with tf.variable_scope('Logit'):
	        logit = Conv1d(x, init_channel*8, num_classes, 1)
	        logit = tf.reduce_sum(logit, 1) / lens
	
        return logit
    return NET(input, init_channel=init_channel)
