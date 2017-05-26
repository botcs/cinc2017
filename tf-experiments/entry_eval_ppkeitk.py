#! /usr/bin/env python3.5

# from tensorflow.python import debug as tf_debug
import json
import os
import sys
import numpy as np
from scipy.io import loadmat
import tensorflow as tf
from time import time, sleep

from model.assembler import get_model_logits
tf.set_random_seed(42)


def pprint(*text, **kwargs):
    print('---TFLOW:', *text, flush=True, **kwargs)


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

        pprint(fname, 'evaluated to',
               self.map_logits(logits),
               (np.exp(logits)/np.exp(logits).sum() * 100).astype(int),
               'Wall time: %4.3f' % (time() - self.start_time))
        with open(self.answers, 'a') as f:
            f.write(fname + ',' + self.map_logits(logits) + '\n')

        self.processing = None
        self.processed.add(fname)
        self.last_update = time()

    def check(self):
        try:
            with open('RECORDS') as f:
                fnames = {fn for fn in f.read().split('\n') if len(fn) > 0}
        except FileNotFoundError:
            pass

        new_names = fnames - self.processed - self.waiting
        if len(new_names) > 0:
            pprint('New files (%d) read' % len(new_names))
            self.last_update = time()
            self.waiting.update(new_names)
            return True
        elif len(self.waiting) > 0:
            pprint('No new file, but (%d) are waiting' %
                   len(self.waiting))
            return True
        return False


def main(argv):

    dh = data_handler('data/raw/training2017/')
    proto_path = 'proto/deep_resnet_no_pool.json'
    proto_name = os.path.splitext(proto_path)[0]
    proto_name = os.path.basename(proto_name)

    tf.reset_default_graph()
    pprint('Building computational graph... ')
    pprint('-'*80)
    network_params = json.load(open(proto_path))
    input_op = tf.placeholder(1, [1, None], 'INPUT')
    seq_len = tf.shape(input_op, name='seq_len')[-1]
    logits = get_model_logits(seq_len, input_op, **network_params)[0]
    pprint('-'*80)
    pprint('Computational graph building done!')

    pprint('Loading pretrained parameters...')

    WAIT_TIME = .01
    MAX_TIME = 10.

    '''
    if tf.gfile.Exists('answers.txt'):
        pprint('!!!REMOVING previous answers.txt')
        tf.gfile.Remove('answers.txt')
    '''

    with tf.Session() as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        ckpt = tf.train.get_checkpoint_state('ckpt/%s/' % proto_name)
        saver = tf.train.Saver()
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            global_step =\
                ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            pprint('Checkpoint loaded at train step:', global_step)
        else:
            pprint('No checkpoint file found')
            return

        start_time = time()
        t = time() - start_time
        while True:
            if dh.check():
                while len(dh.waiting) > 0:
                    sample = dh.next()
                    feed = {'is_training:0': False, input_op: sample['val']}
                    logits_val = sess.run(logits, feed)
                    dh.write(sample['name'], logits_val)
            else:
                # pprint('No file to process, waiting... ')
                sleep(WAIT_TIME)
            if dh.t_since_update() > MAX_TIME:
                break

        pprint('Max idle time (%d secs) expired, quitting' % MAX_TIME)


if __name__ == '__main__':
    main(sys.argv)
