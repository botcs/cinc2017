#!/usr/bin/env ipython
# coding: utf-8

import numpy as np
import tensorflow as tf
import scipy.io as io
import random

<<<<<<< HEAD
def read_raw(UsedForTraining,dir='../../training2017/'):
    UsedForTesting=1.0-UsedForTraining
    label_dict = {
        'N':0,
        'A':1,
        'O':2,
        '~':3
    }


    #load presaved features - time domain features based on pantompkins
    features=np.load('td_features.npy') #this file contains the previously saved time domain features
=======

def read_raw(UsedForTraining, dir='../../training2017/'):
    UsedForTesting = 1.0 - UsedForTraining
    label_dict = {
        'N': 0,
        'A': 1,
        'O': 2,
        '~': 3
    }

    # load presaved features - time domain features based on pantompkins
    # this file contains the previously saved time domain features
    features = np.load('td_features.npy')
>>>>>>> 89b6a43314e18ded2ed7ab8f7e2938583d71c218
    train_data = []
    train_features = []
    train_label = []
    test_data = []
    test_label = []
    test_features = []
    lens = []
<<<<<<< HEAD
    annotations = open(dir+'REFERENCE.csv', 'r').read().splitlines()
    NumRecords=len(annotations)
    Trainindices=set(random.sample(range(NumRecords),int(NumRecords*UsedForTraining)))
    for i, line in enumerate(annotations):
        fname, label_str = line.split(',')

        x = io.loadmat(dir+fname+'.mat')['val']\
=======
    annotations = open(dir + 'REFERENCE.csv', 'r').read().splitlines()
    NumRecords = len(annotations)
    Trainindices = set(
        random.sample(
            range(NumRecords), int(
                NumRecords * UsedForTraining)))
    for i, line in enumerate(annotations):
        fname, label_str = line.split(',')

        x = io.loadmat(dir + fname + '.mat')['val']\
>>>>>>> 89b6a43314e18ded2ed7ab8f7e2938583d71c218
            .astype(np.float32).squeeze()
        # Normal
        x -= x.mean()
        x /= x.std()

<<<<<<< HEAD
        

=======
>>>>>>> 89b6a43314e18ded2ed7ab8f7e2938583d71c218
        y = label_dict[label_str]
        if i in Trainindices:
            train_label.append(y)
            train_data.append(x)
<<<<<<< HEAD
            train_features.append(features[i,:])
        else:
            test_label.append(y)
            test_data.append(x)
            test_features.append(features[i,:])
        lens.append(len(x))
        if i%50==0: 
=======
            train_features.append(features[i, :])
        else:
            test_label.append(y)
            test_data.append(x)
            test_features.append(features[i, :])
        lens.append(len(x))
        if i % 50 == 0:
>>>>>>> 89b6a43314e18ded2ed7ab8f7e2938583d71c218
            print('\rReading files: %05d   ' % i, end='', flush=True)

    print('\rReading files: %05d   ' % i, end='', flush=True)
    assert(len(train_label) == len(train_data))
    assert(len(test_label) == len(test_data))
    print('\nReading successful!')
<<<<<<< HEAD
    
    train_data = np.array(train_data)
    train_label = np.array(train_label)
    train_features =np.array(train_features)
    test_data = np.array(test_data)
    test_label = np.array(test_label)
    test_features =np.array(test_features)
    #lens = np.array(lens)  # it seems we do not use this at all???
    #data_size = len(data)
    train_class_hist = np.histogram(train_label, bins=len(label_dict))[0]
    test_class_hist = np.histogram(test_label, bins=len(label_dict))[0]
    
    return train_data, train_label, train_class_hist,train_features, test_data, test_label, test_class_hist, test_features
=======

    train_data = np.array(train_data)
    train_label = np.array(train_label)
    train_features = np.array(train_features)
    test_data = np.array(test_data)
    test_label = np.array(test_label)
    test_features = np.array(test_features)
    # lens = np.array(lens)  # it seems we do not use this at all???
    #data_size = len(data)
    train_class_hist = np.histogram(train_label, bins=len(label_dict))[0]
    test_class_hist = np.histogram(test_label, bins=len(label_dict))[0]

    return train_data, train_label, train_class_hist, train_features, test_data, test_label, test_class_hist, test_features

>>>>>>> 89b6a43314e18ded2ed7ab8f7e2938583d71c218

def make_example(sequence, label, features):
    # The object we return
    ex = tf.train.SequenceExample()
    # A non-sequential feature of our example
    sequence_length = len(sequence)
    ex.context.feature['length'].int64_list.value.append(sequence_length)
    ex.context.feature['label'].int64_list.value.append(label)

<<<<<<< HEAD

=======
>>>>>>> 89b6a43314e18ded2ed7ab8f7e2938583d71c218
    fl_val = ex.feature_lists.feature_list['tdfeature']
    for token in features:
        fl_val.feature.add().float_list.value.append(token)

<<<<<<< HEAD

=======
>>>>>>> 89b6a43314e18ded2ed7ab8f7e2938583d71c218
    fl_val = ex.feature_lists.feature_list['data']
    for token in sequence:
        fl_val.feature.add().float_list.value.append(token)

    return ex


def write_TFRecord(data, label, features, fname='train', threads=8):
    with open(fname + '.TFRecord', 'w') as fp:
        writer = tf.python_io.TFRecordWriter(fp.name)
        print('Sampling...')
<<<<<<< HEAD
         
            
        for i, (x, y,z) in enumerate(zip(data, label,features)):
            ex = make_example(x, y,z)
            writer.write(ex.SerializeToString())
            print('\r%05d'%i, end=' ', flush=True)
        writer.close()
        print("\nWrote to {}".format(fp.name))
        
if __name__ == '__main__':
    UsedForTraining=0.7
    train_data, train_label, train_class_hist,train_features, test_data, test_label, test_class_hist, test_features = read_raw(UsedForTraining)
    np.save('train_class_hist_with_td_features', train_class_hist)
    write_TFRecord(train_data, train_label,train_features,fname='train_with_td_features')
    print('train sampling finished')
    np.save('test_class_hist_with_td_features', test_class_hist)
    write_TFRecord(test_data, test_label,test_features,fname='test_with_td_features')
=======

        for i, (x, y, z) in enumerate(zip(data, label, features)):
            ex = make_example(x, y, z)
            writer.write(ex.SerializeToString())
            print('\r%05d' % i, end=' ', flush=True)
        writer.close()
        print("\nWrote to {}".format(fp.name))


if __name__ == '__main__':
    UsedForTraining = 0.7
    train_data, train_label, train_class_hist, train_features, test_data, test_label, test_class_hist, test_features = read_raw(
        UsedForTraining)
    np.save('train_class_hist_with_td_features', train_class_hist)
    write_TFRecord(
        train_data,
        train_label,
        train_features,
        fname='train_with_td_features')
    print('train sampling finished')
    np.save('test_class_hist_with_td_features', test_class_hist)
    write_TFRecord(
        test_data,
        test_label,
        test_features,
        fname='test_with_td_features')
>>>>>>> 89b6a43314e18ded2ed7ab8f7e2938583d71c218
