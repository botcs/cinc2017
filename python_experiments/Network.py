import random
import numpy as np
import tensorflow as tf


def conv2d(x, W, b, strides=1):
    # Conv2D with bias and ReLU
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    # ReLU - definitions of the regular ReLU -> use leaky instead, avoid dead ReLU problem
    #conv1 = tf.nn.relu(conv1_in)
    # Leaky ReLU
    alpha = 0.01
    x = tf.maximum(alpha * x, x)
    return x


def maxpool2d(x, k=3):
    # MaxPool2D
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, Size):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, Size[0], Size[1], Size[2]])
    # Conv1
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Pooling
    pool1 = maxpool2d(conv1, k=2)
    # Conv2
    conv2 = conv2d(pool1, weights['wc2'], biases['bc2'])
    # Pooling
    pool2 = maxpool2d(conv2, k=2)
    # Conv3
    conv3 = conv2d(pool2, weights['wc3'], biases['bc3'])
    # Pooling
    pool3 = maxpool2d(conv3, k=2)
    # Conv4
    conv4 = conv2d(pool3, weights['wc4'], biases['bc4'])
    # Pooling
    pool4 = maxpool2d(conv4, k=2)
    # Conv5
    conv5 = conv2d(pool4, weights['wc5'], biases['bc5'])
    # Pooling
    pool5 = maxpool2d(conv5, k=2)
    # print pool5.get_shape()
    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(pool5, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout - uncomment this to use dropout ->training will become much slower but more accurate
    #fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


def SetWeightsAndBiases(var, weights, biases):
    weights['wc1'] = var[0]
    weights['wc2'] = var[1]
    weights['wc3'] = var[2]
    weights['wc4'] = var[3]
    weights['wc5'] = var[4]
    weights['wd1'] = var[5]
    weights['out'] = var[6]
    biases['bc1'] = var[7]
    biases['bc2'] = var[8]
    biases['bc3'] = var[9]
    biases['bc4'] = var[10]
    biases['bc5'] = var[11]
    biases['bd1'] = var[12]
    biases['out'] = var[13]
    return weights, biases


def GetRandomWeightsAndBiases(n_classes):
    # Define random parameters for training in the network
    # Store layers weight & bias
    weights = {
        # 5x5 conv, 3 input, 64 outputs
        'wc1': tf.Variable(tf.random_normal([5, 1, 1, 512])),
        # 5x5 conv, 64 inputs, 32 outputs
        'wc2': tf.Variable(tf.random_normal([5, 1, 512, 512])),
        'wc3': tf.Variable(tf.random_normal([3, 1, 512, 512])),
        'wc4': tf.Variable(tf.random_normal([3, 1, 512, 256])),
        'wc5': tf.Variable(tf.random_normal([3, 1, 256, 256])),
        # fully connected, 6*6*16 inputs, 1024 outputs
        'wd1': tf.Variable(tf.random_normal([9 * 1 * 256, 20])),
        # 1024 inputs, 10 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([20, n_classes]))
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([512])),
        'bc2': tf.Variable(tf.random_normal([512])),
        'bc3': tf.Variable(tf.random_normal([512])),
        'bc4': tf.Variable(tf.random_normal([256])),
        'bc5': tf.Variable(tf.random_normal([256])),
        'bd1': tf.Variable(tf.random_normal([20])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
    return weights, biases
