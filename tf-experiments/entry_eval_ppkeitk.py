#! /usr/bin/env python3


import json
import os
import numpy as np
from scipy.io import loadmat
import tensorflow as tf
from time import time, sleep

from model.assembler import get_model_logits


class data_handler:

    def __init__(self, basename, fname_reference='RECORDS',
                 answers='answers.txt'):
        self.basename = basename
        self.fname_reference = fname_reference
        self.answers = answers
        self.processed = set()
        self.processing = None
        self.waiting = set()
        self.last_update = time()
        self.start_time = time()

    def map_label(self, label):

        str2idx = {'N': 0, 'A': 1, 'O': 2, '~': 3}
        idx2str = {v: k for k, v in str2idx.items()}
        if type(label) == str:
            return str2idx[label]
        elif type(label) == int:
            return idx2str[label]

    def t_since_update(self):
        return time() - self.last_update

    def next(self):
        if self.processing is not None: raise ValueError(
            'asked for next sample, while previous was not processed')

        if len(self.waiting) > 0:
            record_name = self.waiting.pop()
            fname = os.path.join(self.basename, record_name)
            res = {'name': record_name}
            res.update(loadmat(fname))
            self.processing = record_name
            return res
        return None

    def map_logits(self, logits):
        return self.map_label(int(np.argmax(logits)))

    def write(self, fname, logits):
        assert fname not in self.waiting
        assert fname not in self.processed
        assert fname == self.processing

        print(sample['name'], 'evaluated to',
              dh.map_logits(logits_val),
              'Wall time: %4.3f' % (time() - self.start_time))
        with open(self.answers, 'a') as f:
            f.write(fname + ',' + self.map_logits(logits) + '\n')
        self.processing = None
        self.processed.add(fname)
        self.last_update = time()

    def check(self):
        with open('RECORDS') as f:
            fnames = {fn for fn in f.read().split('\n') if len(fn) > 0}

        new_names = fnames - self.processed - self.waiting
        if len(new_names) > 0:
            print('New files (%d) are waiting' % len(new_names))
            self.last_update = time()
            self.waiting.update(new_names)
            return True
        elif len(self.waiting) > 0:
            print('No new file, but (%d) are waiting' %
                  len(self.waiting))
            return True
        return False

dh = data_handler('data/raw/training2017/')
proto_path = 'test_scripts/wide_resnet.json'
proto_name = os.path.splitext(proto_path)[0]
proto_name = os.path.basename(proto_name)

tf.reset_default_graph()
print('Building computational graph... ')
print('-'*80)
network_params = json.load(open(proto_path))
input_op = tf.placeholder(1, [1, None], 'INPUT')
seq_len = tf.shape(input_op, name='seq_len')[-1]
logits = get_model_logits(seq_len, input_op, **network_params)[0]
print('-'*80)
print('Computational graph building done!')

print('Loading pretrained parameters...')
sv = tf.train.Supervisor(
    logdir='ckpt/%s/' % proto_name,
)

WAIT_TIME = .1
MAX_TIME = 10.

#'''
if tf.gfile.Exists('answers.txt'):
    print('!!!REMOVING previous answers.txt')
    tf.gfile.Remove('answers.txt')
#'''

with sv.managed_session() as sess:
    start_time = time()
    t = time() - start_time
    while not sv.should_stop() and dh.t_since_update() < MAX_TIME:
        print('Checking RECORDS:', end='')
        if dh.check():
            while len(dh.waiting) > 0:
                sample = dh.next()
                logits_val = sess.run(logits, {input_op: sample['val']})
                dh.write(sample['name'], logits_val)
        else:
            print('No file to process, waiting %1.1f sec(s)...' %
                  WAIT_TIME)
            sleep(WAIT_TIME)

    print('Max idle time (%d secs) expired, quitting' % MAX_TIME)
