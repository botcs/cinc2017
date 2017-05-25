#!/usr/bin/env ipython
# coding: utf-8

import numpy as np
import tensorflow as tf
import scipy.io as io

all_label_dict = {
    'N': 0,
    'A': 1,
    'O': 2,
    '~': 3
}

all_data = []
all_label = []
all_lens = []
label_count = [0, 0, 0, 0]
annotations = open('./training2017/REFERENCE.csv', 'r').read().splitlines()
for i, line in enumerate(annotations):
    fname, all_label_str = line.split(',')

    x = io.loadmat(
        './training2017/' +
        fname +
        '.mat')['val'].astype(
        np.float32).squeeze()

    # [0, 1]
    # x -= x.min()
    # x /= x.max()

    # [-1, 1]
    # x -= x.min()
    # x /= x.max()
    # x *= 2
    # x -= 0

    # Normal
    x -= x.mean()
    x /= x.std()

    all_data.append(x)

    y = all_label_dict[all_label_str]
    all_label.append(y)

    all_lens.append(len(x))
    if i % 50 == 0:
        print('\rReading files: %05d   ' % i, end='', flush=True)

print('\rReading files: %05d   ' % i, end='', flush=True)

assert(len(all_label) == len(all_data) == len(all_lens))
print('\nReading successful!')
all_data_size = len(all_data)

# No problem with different lengths
# Using np.array because slice indexing does not copy the all_data
# While native python slicing does
all_data = np.array(all_data)
all_label = np.array(all_label)
all_lens = np.array(all_lens)


def shuffle():
    global all_data
    global all_label
    global all_lens
    p = np.random.permutation(all_data_size)
    # Using fancy indexing for Unison Shuffle
    all_data = all_data[p]
    all_label = all_label[p]
    all_lens = all_lens[p]


def join_samples(sample_list, sample_all_lens):
    res = np.zeros((len(sample_list), sample_all_lens.max(), 1))
    for idx, (sample, l) in enumerate(zip(sample_list, sample_all_lens)):
        res[idx, :l] = sample[None, :, None]
    return res


def random_batch(batch_size=4, num_epochs=10):
    n = batch_size
    shuffle()
    for _ in range(num_epochs):
        for i in range(0, all_data_size, batch_size):
            all_data_window = join_samples(
                all_data[i:i + n], all_lens[i:i + n])
            yield all_data_window, all_label[i:i + n], all_lens[i:i + n]


def equal_batch(batch_size=4):
    n = batch_size
