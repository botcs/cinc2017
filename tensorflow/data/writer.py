#!/usr/bin/env python3.5
# coding: utf-8

import numpy as np
import tensorflow as tf
import scipy.io as io
import sys
import os
import argparse


def read_raw(refname, dir):
    label_dict = {
        'N': 0,
        'A': 1,
        'O': 2,
        '~': 3
    }

    data = []
    label = []
    lens = []
    annotations = open(refname, 'r').read().splitlines()
    for i, line in enumerate(annotations, 1):
        fname, label_str = line.split(',')
        location = os.path.normpath(dir + '/' + fname + '.mat')
        x = io.loadmat(location)['val'].astype(np.float32).squeeze()
        data.append(x)
        y = label_dict[label_str]
        label.append(y)
        lens.append(len(x))
        if i % 50 == 0:
            print('\rReading files: %5d   ' % i, end='', flush=True)

    print('\rReading files: %5d   ' % i, end='', flush=True)
    assert(len(label) == len(data) == len(lens))
    print('\nReading successful!')
    data_size = len(data)
    data = np.array(data)
    label = np.array(label)
    lens = np.array(lens)
    class_hist = np.histogram(label, bins=len(label_dict))[0]

    return data, label, class_hist, refname


def make_example(sequence, label):
    # The object we return
    ex = tf.train.SequenceExample()
    # A non-sequential feature of our example
    sequence_length = len(sequence)
    ex.context.feature['length'].int64_list.value.append(sequence_length)
    ex.context.feature['label'].int64_list.value.append(label)

    fl_val = ex.feature_lists.feature_list['data']
    for token in sequence:
        fl_val.feature.add().float_list.value.append(token)

    return ex


def write_TFRecord(data, label, fname='train', threads=8):
    with open(fname + '.TFRecord', 'w') as fp:
        writer = tf.python_io.TFRecordWriter(fp.name)

        print('Sampling %s...' % fname)

        for i, (x, y) in enumerate(zip(data, label), 1):
            ex = make_example(x, y)
            writer.write(ex.SerializeToString())
            print('\r%5d' % i, end=' ', flush=True)
        writer.close()
        print("\nWrote to {}".format(fp.name))


def main(args):
    data, label, class_hist, fname = read_raw(args.ref, args.from_dir)
    if not os.path.exists(os.path.dirname(args.to)):
        print('make directory:', os.path.dirname(args.to))
        os.mkdir(os.path.dirname(args.to))
    to_path = os.path.normpath(args.to)
    write_TFRecord(data, label, to_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--from_dir', help='location of original data',
        default='./raw/training2017/')
    parser.add_argument(
        '--to', help='location of destination',
        default='./TFRecords/train')
    parser.add_argument(
        '--ref', help='location of reference file',
        default='./raw/training2017/REFERENCE.csv')
    args = parser.parse_args()
    main(args)
