import tensorflow as tf
import numpy as np
import torch as th
from tensorflow.python.framework import ops

def get_logits(input, num_classes, pytorch_statedict_path, res_blocks=3):
    sd = th.load(pytorch_statedict_path)
    #for k, v in sd.items():
    #    print(k, v.size())
    def gener():
        for p in sd.values():
            yield p.cpu().transpose(0, -1).numpy()    
    paramgen = gener()
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
        
    def AvgPool1d(input, factor):
        with tf.variable_scope('AvgPool1d'):
            x = tf.expand_dims(input, 1)
            x = tf.nn.max_pool(x, [1, 1, factor, 1], [1, 1, factor, 1], 'SAME')
            x = tf.squeeze(x, 1)
        return x    

    def Encoder(input, init_channel):
        def DownSampleBlock(input, in_channel, out_channel):
            with tf.variable_scope('DownSampleBlock'):
                x = Conv1d(input, in_channel, out_channel, 7, bias=True)
                x = BatchNorm(x, out_channel)
                x = selu(x)
                x = MaxPool1d(x)
            return x    
        with tf.variable_scope('Encoder'):
            x = DownSampleBlock(input, 1, init_channel)
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

    def FreqFeatures(input):
        with tf.variable_scope('SkipFCN'):
            out = SELU(BatchNorm(Conv1d(x, 1, 16, 17, 1)))
            out = SELU(BatchNorm(Conv1d(x, 16, 16, 9, 2)))
            x = out

            out = SELU(BatchNorm(Conv1d(x, 16, 32, 9, 1)))
            out = SELU(BatchNorm(Conv1d(x, 32, 32, 9, 2)))
            out = MaxPool1d(out)
            x = MaxPool1d(x)
            
            out = tf.concat([x, out], axis=2)
            out = SELU(BatchNorm(Conv1d(x, 32, 64, 9, 1)))
            out = SELU(BatchNorm(Conv1d(x, 64, 64, 9, 2)))
            out = SELU(BatchNorm(Conv1d(x, 64, 64, 9, 4)))
            out = MaxPool1d(out)
            x = MaxPool1d(x)

            out = tf.concat([x, out], axis=2)
            out = SELU(BatchNorm(Conv1d(x, 64, 128, 9, 1)))
            out = SELU(BatchNorm(Conv1d(x, 128, 128, 9, 2)))
            out = SELU(BatchNorm(Conv1d(x, 128, 128, 9, 4)))
            out = MaxPool1d(out)
            x = MaxPool1d(x)

            out = tf.concat([x, out], axis=2)
            out = SELU(BatchNorm(Conv1d(x, 128, 128, 9, 1)))
            out = SELU(BatchNorm(Conv1d(x, 128, 128, 9, 2)))
            out = SELU(BatchNorm(Conv1d(x, 128, 128, 9, 2)))
            length = tf.shape(out)[1]
            out = AvgPool(out, length//20)


    def TimeFeatures(input, init_channel):
        x = Encoder(input, init_channel)
        x = ResNet(x, init_channel*8)
        length = tf.shape(out)[1]
        out = AvgPool(out, length//20)
        return x

    def NET(timeInput, freqInput):
        TF = TimeFeatures(timeInput, 16)
        FF = FreqFeatures(freqInput)
        with tf.variable_scope('Logit'):
            logit = BatchNorm1d(logit, 256)
            logit = SELU(logit)
            logit = Conv1d(x, 256, 3, 1)
            logit = tf.reduce_mean(logit, 1)
            
        return logit
    return NET(timeInput, freqInput)
