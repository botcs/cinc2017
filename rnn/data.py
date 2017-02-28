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
for line in annotations:
    fname, label_str = line.split(',')
    
    x = io.loadmat('./training2017/'+fname+'.mat')['val'].squeeze()
    data.append(x)
    
    y = label_dict[label_str]
    label.append(y)
    
    lens.append(len(x))
    
assert(len(label) == len(data) == len(lens))
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



