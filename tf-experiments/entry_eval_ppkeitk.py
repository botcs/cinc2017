#! /usr/bin/env python3

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


def write_answer(fname, logits):
    labels = ['N', 'A', 'O', '~']
    with open('answers.txt', 'a') as f:
        f.write(fname + ',' + labels[int(np.argmax(logits))] + '\n')
    

def main(argv):

    proto_path = 'proto/deep_resnet_no_pool.json'
    proto_name = os.path.splitext(proto_path)[0]
    proto_name = os.path.basename(proto_name)

    tf.reset_default_graph()
    pprint('Building computational graph... ')
    pprint('-'*80)
    network_params = json.load(open(proto_path))
    input_op = tf.placeholder(1, [1, None], 'INPUT')
    seq_len = tf.shape(input_op, name='seq_len')[-1]
    with tf.device('cpu:0'):
        logits = get_model_logits(seq_len, input_op, **network_params)[0]
    pprint('-'*80)
    pprint('Computational graph building done!')

    pprint('Loading pretrained parameters...')


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

        
        sample = loadmat(argv[1])
        feed = {'is_training:0': False, input_op: sample['val']}
        logits_val = sess.run(logits, feed)
        write_answer(argv[1], logits_val)
        

if __name__ == '__main__':
    main(sys.argv)
