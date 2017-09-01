import tensorflow as tf
import numpy as np
import torch as th
from tensorflow.python.framework import ops

def get_logits(timeInput, freqInput, pytorch_statedict_path, res_blocks=3, 
               testlogit=False, testfeatures=False, noGlobalAvg=False, num_classes=3):
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

    def SELU(x):
        with ops.name_scope('elu') as scope:
            alpha = 1.6732632423543772848170429916717
            scale = 1.0507009873554804934193349852946
            return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))
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
            x = tf.nn.max_pool(x, [1, 1, 2, 1], [1, 1, 2, 1], 'VALID')
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
        
    def variable_size_window(input, N):
        '''
        Returns fix number `N` of equal sized windows and slices the variable
        length input
        Use `SYMMETRIC` padding if necessary.

        Returns `N` equal slices
        '''
        with tf.name_scope('sample_division'):
            x = input
            if len(x.get_shape()) == 3:
                x = x[..., None, :]
            else:
                raise ValueError('`input_op` has incorrect number of dimensions. \
            required shape: [batch_size, sequence_length, num_features]')
            with tf.name_scope('windowing'):
                x_shape = tf.shape(x)
                batch_size, max_seq_len = x_shape[0], x_shape[1]
                # Make sure sequence can be divided to equal parts
                padding = [[0, 0], [0, N-max_seq_len % N], [0, 0], [0, 0]]
                x_pad = tf.pad(x, padding, 'CONSTANT')

                # Don't pad if not necessary, i.e. max_seq_len%N == 0
                new_x = tf.cond(tf.equal(max_seq_len % N, 0),
                                lambda: x, lambda: x_pad)
                max_seq_len = tf.shape(new_x)[1]
                new_shape = [batch_size, N, max_seq_len//N, x.get_shape()[-1].value]
                div_x = tf.reshape(new_x, new_shape)

                # Convenience variable
                return div_x
    
    def GlobalAvg(input, N):
        with tf.name_scope('GlobalAvg'):
            x = variable_size_window(input, N)    
            x = tf.reduce_mean(x, 2)
            return x
    
    def GlobalMax(input, N):
        with tf.name_scope('GlobalAvg'):
            x = variable_size_window(input, N)    
            x = tf.reduce_max(x, 2)
            return x
            
    def FreqFeatures(input):
        with tf.variable_scope('SkipFCN'):
            out = SELU(BatchNorm(Conv1d(input, 16, 16, 17, 1), 16))
            out = SELU(BatchNorm(Conv1d(out, 16, 16, 9, 2), 16))
            out = MaxPool1d(out)
            x = out

            out = SELU(BatchNorm(Conv1d(out, 16, 32, 9, 1), 32))
            out = SELU(BatchNorm(Conv1d(out, 32, 32, 9, 2), 32))
            out = MaxPool1d(out)
            x = MaxPool1d(x)

            out = tf.concat([x, out], axis=2)
            out = SELU(BatchNorm(Conv1d(out, 32+16, 64, 9, 1), 64))
            out = SELU(BatchNorm(Conv1d(out, 64, 64, 9, 2), 64))
            out = SELU(BatchNorm(Conv1d(out, 64, 64, 9, 4), 64))
            out = MaxPool1d(out)
            x = MaxPool1d(x)

            out = tf.concat([x, out], axis=2)
            out = SELU(BatchNorm(Conv1d(out, 64+16, 128, 9, 1), 128))
            out = SELU(BatchNorm(Conv1d(out, 128, 128, 9, 2), 128))
            out = SELU(BatchNorm(Conv1d(out, 128, 128, 9, 4), 128))
            out = MaxPool1d(out)
            x = MaxPool1d(x)

            out = tf.concat([x, out], axis=2)
            out = SELU(BatchNorm(Conv1d(out, 128+16, 128, 9, 1), 128))
            out = SELU(BatchNorm(Conv1d(out, 128, 128, 9, 2), 128))
            out = SELU(BatchNorm(Conv1d(out, 128, 128, 9, 2), 128))
            #if not noGlobalAvg:
            #    out = GlobalMax(out, 20)
            # THIS IS ONLY FOR PARAMETER LEFTOVERS
            # models.0.logit.weight [3, 128, 1]
            # models.0.logit.bias [3]
            Conv1d(out, 128, 3, 1, bias=True)
            
            return out
            

    def TimeFeatures(input, init_channel):
        x = Encoder(input, init_channel)
        x = ResNet(x, init_channel*8)
        #if not noGlobalAvg:
        #    x = GlobalMax(x, 20)
        # THIS IS ONLY FOR PARAMETER LEFTOVERS
        # models.1.logit.weight [3, 128, 1]
        Conv1d(x, 128, 3, 1)
        return x

    def NET(timeInput, freqInput):
        FF = FreqFeatures(freqInput)
        TF = TimeFeatures(timeInput, 16)
        if testfeatures:
            return FF, TF
        features = tf.concat([FF, TF], axis=2)
        with tf.variable_scope('Logit'):
            logit = BatchNorm(features, 256)
            logit = SELU(logit)
            logit = Conv1d(logit, 256, num_classes, 1, bias=True)
            if testlogit:
                return logit
            logit = tf.reduce_mean(logit, 1)
        return logit
    return NET(timeInput, freqInput)
