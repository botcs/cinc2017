#! /usr/bin/env python3

import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow_wrapper as tw
import data_handler

def pprint(*text, **kwargs):
    print(*text, flush=True, **kwargs)

def process_logits(logits):
    noise, classes = logits
    if noise.argmax() == 0:
        return('~')
    if classes.argmax() == 1:
        return('A')
    if classes.argmax() == 0:
        return('N')
    return('O')


def write_answer(fname, answer):
    with open('answers.txt', 'a') as f:
         f.write(fname + ',' + str(answer) + '\n')

def main(argv):
    print('-'*80)
    print('*'*80)
    print('-'*80)
    noise_model_path = 'noise.pkl'
    class_model_path = 'class.pkl'

    tf.reset_default_graph()
    pprint('Building computational graph... ')

    with tf.device('cpu'):
        input_op = tf.placeholder(1, [1, None, 1], name='INPUT')
        noise_model = tw.get_logits(input_op, 3, noise_model_path)[0]
        class_model = tw.get_logits(input_op, 3, class_model_path, res_blocks=9, init_channel=16)[0]
        logits = [noise_model, class_model]
    pprint('Computational graph building done!')
    pprint('Loading data')
    transformations = [
        #data_handler.Crop(6000),
        data_handler.Threshold(sigma=2.2),
        #data_handler.RandomMultiplier(-1),
    ]
    data = data_handler.load_composed(
        sys.argv[1], transformations=transformations)[:, :, None]
    print(data.shape)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        logits_val = sess.run(logits, {input_op: data})
        print('Noise:', logits_val[0])
        print('Class:', logits_val[1])
    print(process_logits(logits_val))
    write_answer(argv[1], process_logits(logits_val))
    return

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
    print('-'*80)
    print('*'*80)
    print('-'*80)
 
if __name__ == '__main__':
    main(sys.argv)
