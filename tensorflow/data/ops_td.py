#!/usr/bin/env ipython
# coding: utf-8

import tensorflow as tf
import numpy as np


class foo(object):
    pass


FLAGS = foo()
FLAGS.path = 'train.TFRecord'
FLAGS.batch_size = tf.constant(16, name='batch_size')
FLAGS.capacity = 512
FLAGS.threads = 8


def parse_example(filename_queue):
    # Define how to parse the example

    reader = tf.TFRecordReader()
    _, example = reader.read(filename_queue)

    context_features = {
        'length': tf.FixedLenFeature([], dtype=tf.int64),
        'label': tf.FixedLenFeature([], dtype=tf.int64)
    }
    sequence_features = {
        'data': tf.FixedLenSequenceFeature([], dtype=tf.float32),
        'tdfeature': tf.FixedLenSequenceFeature([], dtype=tf.float32)
    }
    con_parsed, seq_parsed = tf.parse_single_sequence_example(
        serialized=example,
        context_features=context_features,
        sequence_features=sequence_features
    )
    res = (seq_parsed['data'],
           con_parsed['length'],
           con_parsed['label'],
           seq_parsed['tdfeature'])
    return res


def get_batch_producer(
        path=FLAGS.path,
        batch_size=FLAGS.batch_size,
        prefetch_size=FLAGS.capacity,
        num_of_threads=FLAGS.threads,
        scope='batch_producer'):
    with tf.name_scope(scope):
        filename_queue = tf.train.string_input_producer(
            [path], name='filename_producer')
        with tf.name_scope('example_producer'):
            data, seq_len, label, td_feature = parse_example(filename_queue)
            data = tf.placeholder_with_default(data, [None], name='data')
            label = tf.cast(label, tf.int32, name='label')
            seq_len = tf.cast(seq_len, tf.int32, name='seq_length')
            td_feature = tf.placeholder_with_default(
                td_feature, [None], name='td_feature')
        with tf.name_scope('padded_batch_producer'):
            q = tf.PaddingFIFOQueue(
                capacity=prefetch_size,
                dtypes=[tf.float32, tf.int32, tf.int32, tf.float32],
                shapes=[[None], [], [], [None]], name='padding_queue')

            enqueue_op = q.enqueue(
                [data, seq_len, label, td_feature], name='push_single_example')
            qr = tf.train.QueueRunner(q, [enqueue_op] * num_of_threads)
            tf.train.add_queue_runner(qr)
            batch_op = q.dequeue_many(n=batch_size, name='pop_batch')
    return batch_op
