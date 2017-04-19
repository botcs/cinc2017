#!/usr/bin/env ipython
# coding: utf-8

import numpy as np
import tensorflow as tf
import scipy.io as io
import sys

def read_raw(refname='REFERENCE', dir='./raw/'):
    label_dict = {
        'N':0,
        'A':1,
        'O':2,
        '~':3
    }

    data = []
    label = []
    lens = []
    annotations = open(dir+refname+'.csv', 'r').read().splitlines()
    for i, line in enumerate(annotations):
        fname, label_str = line.split(',')

        x = io.loadmat(dir+fname+'.mat')['val']\
            .astype(np.float32).squeeze()
        # Normal
        # x -= x.mean()
        # x /= x.std()

        data.append(x)

        y = label_dict[label_str]
        label.append(y)

        lens.append(len(x))
        if i%50==0: 
            print('\rReading files: %05d   ' % i, end='', flush=True)

    print('\rReading files: %05d   ' % i, end='', flush=True)
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
        print('Sampling %s...'%fname)
         
        for i, (x, y) in enumerate(zip(data, label)):
            ex = make_example(x, y)
            writer.write(ex.SerializeToString())
            print('\r%05d'%i, end=' ', flush=True)
        writer.close()
        print("\nWrote to {}".format(fp.name))
        
if __name__ == '__main__':
    if len(sys.argv) > 1:
        fnames = sys.argv[1:]
    else:
        fnames = ['TRAIN', 'VALIDATION', 'TEST']
    datasets = map(read_raw, fnames)
    for dataset in datasets:
        data, label, class_hist, fname = dataset
        np.save('class_histogram'+fname, class_hist)
        write_TFRecord(data, label, fname)
    