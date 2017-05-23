#!/usr/bin/env python3
# coding: utf-8

import tensorflow as tf


class model(object):
    '''
    Classify fixed length features, with weighted loss
    classifier will return an object, whose main fields are tensorflow graph
    nodes.
    '''

    def get_layers(self, in_node, fc_sizes, out_dim):

        act_fn = tf.nn.relu
        h = in_node
        # keep_prob = tf.placeholder_with_default(keep_prob, [], 'keep_prob')
        for i, size in enumerate(fc_sizes):
            with tf.variable_scope('hidden_layer%d' % i):
                h = tf.contrib.layers.fully_connected(
                    h, size, act_fn, tf.contrib.layers.batch_norm)
                tf.add_to_collection('activations', h)
                print(h)
        logits = tf.contrib.layers.fully_connected(
            h, out_dim, None, tf.contrib.layers.batch_norm, scope='logits')
        tf.add_to_collection('activations', logits)
        print(logits)
        return logits

    def get_name(self):
        fc_sizes = [str(s) for s in self.fc_sizes]
        name = '--fc' + '-'.join(fc_sizes)
        return name

    def build_graph(self, model_name=None):
        if not model_name:
            model_name = self.get_name()
        self.name = model_name

        # self.keep_prob = tf.placeholder_with_default(
        #     self.def_keep_prob, [], 'keep_prob')

        self.logits = self.get_layers(
            self.input, self.fc_sizes, self.num_classes)

        self.pred = tf.nn.softmax(logits=self.logits, name='predictions')
        print(self.pred)

    def __init__(self, input_op, fc_sizes, num_classes=4, model_name=None,
                 **kwargs):
        '''
        Initializer default vales use tf.app.flags
        returns an object, whose main fields are tensorflow graph nodes.

        fc_sizes: [int, [int...]] Size of fc layers connected to the last LSTM
        cell's output
        keep_prob: float, Probability of keeping a value in DROPOUT layers
        '''
        self.input = input_op
        self.fc_sizes = fc_sizes
        # self.def_keep_prob = keep_prob
        self.num_classes = num_classes
        self.name = self.get_name()
        with tf.variable_scope('classifier'):
            print('\nFC' + self.name)
            self.build_graph()


def get_logits_and_pred(input_op, return_name=True, **kwargs):
    '''Convenience function for retrieveng
    calssifier model graph definition's output'''
    c = model(input_op, **kwargs)
    if return_name:
        return c.logits, c.pred, c.name
    return c.logits, c.pred
