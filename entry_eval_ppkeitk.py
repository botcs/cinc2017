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
    class_model_path = 'freq-params.pkl'

    tf.reset_default_graph()
    pprint('Building computational graph... ')

    with tf.device('cpu'):
        input_op1 = tf.placeholder(1, [1, None, 1])
        input_op2 = tf.placeholder(1, [1, None, 8])
        noise_model = tw.get_logits(input_op1, 3, noise_model_path)[0]
        class_model = tw.get_logits(
                input=input_op2, 
                num_classes=3, 
                param_path=class_model_path, 
                res_blocks=9, 
                init_channel=8,
                in_channel=8)[0]
        logits = [noise_model, class_model]
    pprint('Computational graph building done!')
    pprint('Loading data')
    timeTransformations = [
        data_handler.Threshold(sigma=2.2),
    ]
    freqTransformations = [
        data_handler.Spectogram(15),
        data_handler.Logarithm()
    ]
    timeData = data_handler.load_composed(
        sys.argv[1], transformations=timeTransformations)[:, :, None]
    freqData = data_handler.load_composed(
            sys.argv[1], transformations=freqTransformations).T[None, :, :]
    print(timeData.shape, freqData.shape)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        logits_val = sess.run(logits, {input_op1: timeData, input_op2:freqData})
        print('Noise:', logits_val[0])
        print('Class:', logits_val[1])
        print(process_logits(logits_val))
    write_answer(argv[1], process_logits(logits_val))
    
    print('-'*80)
    print('*'*80)
    print('-'*80)
 
if __name__ == '__main__':
    main(sys.argv)
