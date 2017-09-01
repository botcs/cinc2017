#! /usr/bin/env python3

import os
import sys
import numpy as np
import tensorflow as tf
import combined_wrapper as tw
import data_handler

def pprint(*text, **kwargs):
    print(*text, flush=True, **kwargs)

def process_logits(logits):
    noise, classes = logits
    if noise.argmax() == 0:
        return('~')
    if classes.argmax() == 0:
        return('N')
    if classes.argmax() == 1:
        return('A')
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
        xtime = tf.placeholder(1, [1, None, 1])
        xfreq = tf.placeholder(1, [1, None, 16])
        noise_model = tw.get_logits(timeInput=xtime, freqInput=xfreq, pytorch_statedict_path=noise_model_path, res_blocks=3, num_classes=2)[0]
        class_model = tw.get_logits(timeInput=xtime, freqInput=xfreq, pytorch_statedict_path=class_model_path, res_blocks=3, num_classes=3)[0]
        logits = [noise_model, class_model]
    pprint('Computational graph building done!')
    pprint('Loading data')
    global_transforms = [
        #data_handler.Crop(6000),
    ]

    transTime = [
        data_handler.Threshold(sigma=2.2),
        data_handler.RandomMultiplier(-1),
    ]

    transFreq = [
        data_handler.RandomMultiplier(-1),
        data_handler.Spectogram(31),
        #data_handler.Logarithm()
    ]
    
    data = data_handler.load_forked(
        sys.argv[1],
        global_transforms=global_transforms,
        fork_transforms={'time':transTime, 'freq':transFreq})
    timeData = data['time'][:, :, None]
    freqData = data['freq'][:, :, None].transpose(2, 1, 0)
    print('Evaluation')
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        logits_val = sess.run(logits, {xtime: timeData, xfreq:freqData})
        print('Noise:', logits_val[0])
        print('Class:', logits_val[1])
        print(process_logits(logits_val))
    write_answer(argv[1], process_logits(logits_val))

    print('-'*80)
    print('*'*80)
    print('-'*80)
 
if __name__ == '__main__':
    main(sys.argv)
