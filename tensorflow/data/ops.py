#!/usr/bin/env ipython
# coding: utf-8

import tensorflow as tf
import numpy as np

class foo(object):
    pass
FLAGS = foo()
FLAGS.path = 'train.TFRecord'
FLAGS.batch_size = tf.constant(16)
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
    num_of_threads=FLAGS.threads):
    
    filename_queue = tf.train.string_input_producer([path])
    data, seq_len, label = parse_example(filename_queue)
    seq_len = tf.cast(seq_len, tf.int32)
    label = tf.cast(label, tf.int32)
    q = tf.PaddingFIFOQueue(
        capacity=prefetch_size,
        dtypes=[tf.float32, tf.int32, tf.int32],
        shapes=[[None], [], []])
    
    enqueue_op = q.enqueue([data, seq_len, label])
    qr = tf.train.QueueRunner(q, [enqueue_op]*num_of_threads)
    tf.train.add_queue_runner(qr)
    batch_op = q.dequeue_many(n=batch_size)
    return batch_op