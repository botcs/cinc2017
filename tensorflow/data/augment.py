#!/usr/bin/env python3.5
# coding: utf-8

import scipy.io as io
import os
import numpy as np
import argparse


def read_mat(refname, dir):

    data = []
    label = []
    lens = []
    annotations = open(refname, 'r').read().splitlines()
    for i, line in enumerate(annotations, 1):
        fname, label_str = line.split(',')
        location = os.path.normpath(dir + '/' + fname + '.mat')
        x = io.loadmat(location)['val']\
            .astype(np.float32).squeeze()
        data.append(x)
        label.append(label_str)
        lens.append(len(x))
        if i % 50 == 0:
            print('\rReading files: %05d' % i, end='', flush=True)

    print('\rReading files: %05d\t' % i, end='', flush=True)
    assert(len(label) == len(data) == len(lens))
    print('Reading successful!')

    # for fancy indexing
    data = np.array(data)
    label = np.array(label)
    lens = np.array(lens)
    return data, label, lens, refname


def write_mat(data, label, refname, dir):

    refname = os.path.normpath(dir + '/' + os.path.basename(refname))
    annotations = open(refname, 'w')
    for i, (d, l) in enumerate(zip(data, label), 1):
        record_name = 'AUG%05d' % i
        fname = os.path.normpath(dir + '/' + record_name + '.mat')
        annotations.write('%s,%s\n' % (record_name, l))
        io.savemat(fname, {'val': d})
        if i % 50 == 0:
            print('\rWriting files: %05d' % i, end='', flush=True)

        print('\rWriting files: %05d\t' % i, end='', flush=True)
        print('Writing successful!')


def augment(data, num_samples, labels=None, alfa=.01, beta=.01):

    res = []
    res_labels = []
    for i in range(num_samples):
        sample = data[i % len(data)]
        # uni noise
        multiplicative_noise = np.random.randn(len(sample)) * alfa + 1
        # normal noise
        additive_noise = np.random.randn(len(sample)) * sample.std() * beta
        #additive_noise = 0
        res.append(sample * multiplicative_noise + additive_noise)
        if labels:
            l = labels[i % len(labels)]
            res_labels.append(l)

    return res, res_labels


def main(args):

    orig = read_mat(refname=args.ref, dir=args.from_dir)
    data, label = orig[:2]
    classes = 'NAO~'

    if not os.path.exists(args.to_dir):
        print('make directory:', args.to_dir)
        os.mkdir(args.to_dir)

    data_sets = [data[label == c] for c in classes]
    aug_data = []
    aug_label = []
    print('Augmenting...\t', end='', flush=True)
    for data_set, label in zip(data_sets, classes):
        res = augment(data_set, args.N, [label])
        aug_data.extend(res[0])
        aug_label.extend(res[1])
        print('%s\t' % label, end='', flush=True)
    print('Done!')
    write_mat(aug_data, aug_label, args.ref, dir=args.to_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--from_dir', help='location of original data',
        default='./raw/training2017/')
    parser.add_argument(
        '--to_dir', help='location of destination',
        default='./raw/augment/')
    parser.add_argument(
        '--ref', help='location of reference file',
        default='./raw/training2017/REFERENCE.csv')
    parser.add_argument(
        'N', help='number to sample for every class',
        type=int)
    args = parser.parse_args()
    main(args)
