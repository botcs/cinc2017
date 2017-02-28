#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import tensorflow as tf
import scipy.io as io

label_dict = {
    'N':0,
    'A':1,
    'O':2,
    '~':3
}

data = []
label = []
lens = []

annotations = open('./training2017/REFERENCE.csv', 'r').read().splitlines()
for i, line in enumerate(annotations):
    fname, label_str = line.split(',')
    
    x = io.loadmat('./training2017/'+fname+'.mat')['val'].astype(np.float32).squeeze()
    
    # [0, 1]
    # x -= x.min()
    # x /= x.max()
    
    # [-1, 1]
    # x -= x.min()
    # x /= x.max()
    # x *= 2
    # x -= 1
    
    # Normal
    x -= x.mean()
    x /= x.std()
    
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

# No problem with different lengths
# Using np.array because slice indexing does not copy the data
# While native python slicing does
data = np.array(data)
label = np.array(label)
lens = np.array(lens)


def shuffle():
    global data
    global label
    global lens
    p = np.random.permutation(data_size)
    # Using fancy indexing for Unison Shuffle
    data = data[p]
    label = label[p]
    lens = lens[p]
    

def batch_pool(batch_size=64, num_epochs=10, random=True):
    n = batch_size
    if random:
        shuffle()
    for _ in range(num_epochs):    
        for i in range(0, data_size, batch_size):
            yield data[i:i+n], label[i:i+n], lens[i:i+n]



