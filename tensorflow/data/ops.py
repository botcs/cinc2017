#!/usr/bin/env ipython
# coding: utf-8

import tensorflow as tf
import numpy as np


class foo(object):
    pass


FLAGS = foo()
FLAGS.path = 'train.TFRecord'
FLAGS.batch_size = 16
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
        'data': tf.FixedLenSequenceFeature([], dtype=tf.float32)
    }
    con_parsed, seq_parsed = tf.parse_single_sequence_example(
        serialized=example,
        context_features=context_features,
        sequence_features=sequence_features
    )
    res = (seq_parsed['data'],
           con_parsed['length'],
           con_parsed['label'])
    return res


def get_batch_producer(
        path=FLAGS.path,
        batch_size=FLAGS.batch_size,
        prefetch_size=FLAGS.capacity,
        num_of_threads=FLAGS.threads,
        scope='batch_producer'):

    if isinstance(path, str):
        path = [path]
    with tf.name_scope(scope):
        filename_queue = tf.train.string_input_producer(
            path, name='filename_producer', shuffle=True)
        with tf.name_scope('example_producer'):
            data, seq_len, label = parse_example(filename_queue)
            data = tf.placeholder_with_default(data, [None], name='data')
            label = tf.cast(label, tf.int32, name='label')
            seq_len = tf.cast(seq_len, tf.int32, name='seq_length')

        # THIS IS STILL NOT AVAILABLE IN TENSORFLOW
        # https://github.com/tensorflow/tensorflow/issues/5147
        # shuffle batch with dynamic padding... duh... no workaround
        # ValueError: All shapes must be fully defined:
        # [TensorShape([Dimension(None)]), TensorShape([]), TensorShape([])]
        """with tf.name_scope('shuffle_batch_producer'):
      q = tf.RandomShuffleQueue(
        capacity=2*prefetch_size,
        min_after_dequeue=prefetch_size,
        dtypes=[tf.float32, tf.int32, tf.int32],
        shapes=[[None], [], []], name='shuffle_queue')

      enqueue_op = q.enqueue([data, seq_len, label], name='push_single_example')
      qr = tf.train.QueueRunner(q, [enqueue_op]*num_of_threads)
      tf.train.add_queue_runner(qr)
      batch_op = q.dequeue_many(n=batch_size, name='pop_batch')
    """

        batch_op = tf.train.batch(
            [data, seq_len, label],
            batch_size,
            num_of_threads,
            prefetch_size,
            shapes=[[None], [], []],
            dynamic_pad=True,
            name='padded_batch_queue')

        """if shuffle:
      batch_op = tf.train.shuffle_batch(
        [data],
        batch_size,
        capacity=5*prefetch_size,
        min_after_dequeue=3*prefetch_size,
        num_threads=num_of_threads,
        enqueue_many=False,
        #shapes=[[None], [], []],
        name='shuffle_batch_producer')
    """

        """with tf.name_scope('padded_batch_producer'):
      q = tf.PaddingFIFOQueue(
        capacity=prefetch_size,
        dtypes=[tf.float32, tf.int32, tf.int32],
        shapes=[[None], [], []], name='padding_queue')

      enqueue_op = q.enqueue([data, seq_len, label], name='push_single_example')
      qr = tf.train.QueueRunner(q, [enqueue_op]*num_of_threads)
      tf.train.add_queue_runner(qr)
      batch_op = q.dequeue_many(n=batch_size, name='pop_batch')
    """
    return batch_op


def get_even_batch_producer(paths,
                            batch_size=FLAGS.batch_size,
                            prefetch_size=FLAGS.capacity,
                            num_of_threads=FLAGS.threads):

    tf.assert_greater_equal(
        batch_size, len(paths), data=[batch_size, len(paths)],
        message='batch_size must be greater than the number of classes')

    sub_batch_size = batch_size // len(paths)
    input_prods = []
    for path in paths:
        input_prods.append(get_batch_producer(
            path,
            batch_size=sub_batch_size,
            prefetch_size=FLAGS.capacity,
            num_of_threads=FLAGS.threads,
            scope='producer_%s' % path
        ))

    '''batch_op = tf.train.batch_join(
      list(zip(*input_prods)),
      batch_size,
      #num_of_threads,
      capacity=prefetch_size,
      shapes=[[None], [], []],
      dynamic_pad=True,
      enqueue_many=True,
      name='even_batch_producer')'''

    with tf.name_scope('even_batch_producer'):
        q = tf.PaddingFIFOQueue(
            capacity=prefetch_size,
            dtypes=[tf.float32, tf.int32, tf.int32],
            shapes=[[None], [], []], name='padding_even_queue')

        for data, seq_len, label in input_prods:
            enqueue_op = q.enqueue_many(
                [data, seq_len, label], name='push_many_example_of_class')
            qr = tf.train.QueueRunner(q, [enqueue_op] * num_of_threads)
            tf.train.add_queue_runner(qr)

        batch_op = q.dequeue_many(n=batch_size, name='pop_batch')

    return batch_op, input_prods
