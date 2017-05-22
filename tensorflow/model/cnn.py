#!/usr/bin/env python3
# coding: utf-8
import tensorflow as tf


class model(object):
    '''
    Classify fixed length features, with weighted loss
    classifier will return an object,
    whose main fields are tensorflow graph nodes.

    '''

    def get_layers(self, seq_len, in_node, out_dims, kernel_sizes, pool_sizes,
                   residual, use_bnorm=True):
        '''
        `out_dims`: a list of integers for the featuremap [out_dims1,
         out_dims2, ...]
        `kernels_sizes`: a single integer or
          a list of integers [kernel_size1, kernel_size2, ...] which must be the
          same length as out_dims
        `pool_sizes`: a single integer or
          a list of integers [pool_size1, pool_size2, ...] which must be the
          same length as out_dims
        '''

        # Converting to NHWC where N is batch and H will be seq_len

        if not residual:
            h = in_node[..., None, None]
        else:
            h = in_node

        if use_bnorm:
            biases_initializer = None
            normalizer_fn = tf.contrib.layers.batch_norm
        else:
            biases_initializer = tf.zeros_initializer
            normalizer_fn = None

        # keep_prob = tf.placeholder_with_default(keep_prob, [], 'keep_prob')

        for i, (dim, ker, pool) in enumerate(
                zip(out_dims, kernel_sizes, pool_sizes)):
            with tf.variable_scope('conv%d' % (i + 1)):
                scope = 'c%dk%dw%d' % (dim, ker, pool)
                # does the same as 1d, but with convenience function
                h = tf.contrib.layers.conv2d(
                    h, dim, [ker, 1],
                    normalizer_fn=normalizer_fn,
                    biases_initializer=biases_initializer,
                    scope=scope)

                if pool > 1:
                    h = tf.contrib.layers.max_pool2d(
                        h, kernel_size=[pool, 1], stride=[pool, 1])
                    seq_len /= pool

                # h = tf.nn.dropout(h, keep_prob)
                print(h)
        if residual:
            return h, seq_len
        return tf.squeeze(h, axis=2), seq_len

    def get_name(self):
        cnn_sizes = ['%dx%d' % (d, k) for d, k in
                     zip(self.out_dims, self.kernel_sizes)]

        name = '--cnn' + '-'.join(cnn_sizes)
        return name

    def build_graph(self, model_name=None):
        if not model_name:
            model_name = self.get_name()
        self.name = model_name

        # self.keep_prob = tf.placeholder_with_default(
        #     self.def_keep_prob, [], 'keep_prob')

        self.output, self.seq_len = self.get_layers(
            seq_len=self.seq_len,
            in_node=self.input,
            out_dims=self.out_dims,
            kernel_sizes=self.kernel_sizes,
            pool_sizes=self.pool_sizes,
            residual=self.residual)

    def __init__(self, seq_len, input_op, out_dims, kernel_sizes, pool_sizes,
                 residual, model_name=None, **kwargs):
        '''
        Initializer default vales use tf.app.flags
        returns an object, whose main fields are tensorflow graph nodes.
        '''

        self.seq_len = seq_len
        self.input = input_op

        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * len(out_dims)
        if isinstance(pool_sizes, int):
            pool_sizes = [pool_sizes] * len(out_dims)
        assert len(out_dims) == len(kernel_sizes) == len(pool_sizes)

        self.out_dims = out_dims
        self.kernel_sizes = kernel_sizes
        self.pool_sizes = pool_sizes

        # self.def_keep_prob = keep_prob
        self.residual = residual
        self.name = self.get_name()
        with tf.variable_scope(model_name):
            print(model_name + self.name)
            self.build_graph()


def get_output(seq_len, input_op, return_name=True, **kwargs):
    cnn = model(seq_len, input_op, **kwargs)
    if return_name:
        return seq_len, cnn.output, cnn.name
    return seq_len, cnn.output
