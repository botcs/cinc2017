#!/usr/bin/env ipython
# coding: utf-8

import tensorflow as tf
import numpy as np

<<<<<<< HEAD
class foo(object):
    pass
=======

class foo(object):
    pass


>>>>>>> 89b6a43314e18ded2ed7ab8f7e2938583d71c218
FLAGS = foo()
FLAGS.path = 'train.TFRecord'
FLAGS.batch_size = tf.constant(16, name='batch_size')
FLAGS.capacity = 512
FLAGS.threads = 8

<<<<<<< HEAD
def parse_example(filename_queue):
    # Define how to parse the example
    
    reader = tf.TFRecordReader()
    _, example = reader.read(filename_queue)
    
=======

def parse_example(filename_queue):
    # Define how to parse the example

    reader = tf.TFRecordReader()
    _, example = reader.read(filename_queue)

>>>>>>> 89b6a43314e18ded2ed7ab8f7e2938583d71c218
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
<<<<<<< HEAD
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
        filename_queue = tf.train.string_input_producer([path], name='filename_producer')
=======
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
>>>>>>> 89b6a43314e18ded2ed7ab8f7e2938583d71c218
        with tf.name_scope('example_producer'):
            data, seq_len, label, td_feature = parse_example(filename_queue)
            data = tf.placeholder_with_default(data, [None], name='data')
            label = tf.cast(label, tf.int32, name='label')
            seq_len = tf.cast(seq_len, tf.int32, name='seq_length')
<<<<<<< HEAD
            td_feature = tf.placeholder_with_default(td_feature, [None], name='td_feature')
=======
            td_feature = tf.placeholder_with_default(
                td_feature, [None], name='td_feature')
>>>>>>> 89b6a43314e18ded2ed7ab8f7e2938583d71c218
        with tf.name_scope('padded_batch_producer'):
            q = tf.PaddingFIFOQueue(
                capacity=prefetch_size,
                dtypes=[tf.float32, tf.int32, tf.int32, tf.float32],
                shapes=[[None], [], [], [None]], name='padding_queue')

<<<<<<< HEAD
            enqueue_op = q.enqueue([data, seq_len, label, td_feature], name='push_single_example')
            qr = tf.train.QueueRunner(q, [enqueue_op]*num_of_threads)
=======
            enqueue_op = q.enqueue(
                [data, seq_len, label, td_feature], name='push_single_example')
            qr = tf.train.QueueRunner(q, [enqueue_op] * num_of_threads)
>>>>>>> 89b6a43314e18ded2ed7ab8f7e2938583d71c218
            tf.train.add_queue_runner(qr)
            batch_op = q.dequeue_many(n=batch_size, name='pop_batch')
    return batch_op
