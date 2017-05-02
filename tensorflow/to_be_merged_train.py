import tensorflow as tf
import numpy as np

import model.cnn as cnn
import model.rnn as rnn
import model.fourier as fourier
import model.classifier as classifier

import data.ops

import time
import matplotlib.pyplot as plt


<<<<<<< HEAD
#Model definition
=======
# Model definition
>>>>>>> 89b6a43314e18ded2ed7ab8f7e2938583d71c218

tf.reset_default_graph()
batch_size = tf.placeholder_with_default(32, [])
input_op, seq_len, label = data.ops.get_batch_producer(
    batch_size=batch_size, path='./train.TFRecord')

cnn_params = {
<<<<<<< HEAD
    'out_dims' : [10],
    'kernel_sizes' : 32,
    'pool_sizes' : 10
}
rnn_params = {
    'rnn_sizes' : [10],
    'time_steps' : 100
}
fourier_params = {
    'use_magnitude' : True,
    'featureindices' : [33,  27, 17,  38, 134]
}
fc_params = {
    'fc_sizes' : [5]
}

c = cnn.get_output(seq_len=seq_len, input_op=input_op, **cnn_params)
r = rnn.get_model(batch_size=batch_size, seq_len=seq_len, input_op=c, **rnn_params)
f = fourier.get_output(seq_len=seq_len, input_op=input_op,**fourier_params)
concatenated_features=tf.concat([r.last_output,f], 1)
logits, pred = classifier.get_logits_and_pred(input_op=concatenated_features, **fc_params)

#Time measure
#convenience function
=======
    'out_dims': [10],
    'kernel_sizes': 32,
    'pool_sizes': 10
}
rnn_params = {
    'rnn_sizes': [10],
    'time_steps': 100
}
fourier_params = {
    'use_magnitude': True,
    'featureindices': [33, 27, 17, 38, 134]
}
fc_params = {
    'fc_sizes': [5]
}

c = cnn.get_output(seq_len=seq_len, input_op=input_op, **cnn_params)
r = rnn.get_model(
    batch_size=batch_size,
    seq_len=seq_len,
    input_op=c,
    **rnn_params)
f = fourier.get_output(seq_len=seq_len, input_op=input_op, **fourier_params)
concatenated_features = tf.concat([r.last_output, f], 1)
logits, pred = classifier.get_logits_and_pred(
    input_op=concatenated_features, **fc_params)

# Time measure
# convenience function

>>>>>>> 89b6a43314e18ded2ed7ab8f7e2938583d71c218

def measure_time(op, feed_dict={}, n_times=10):
    with tf.Session() as sess:
        print('Sess started')
        coord = tf.train.Coordinator()
        tf.global_variables_initializer().run()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
<<<<<<< HEAD
        
        init_state = sess.run(r.zero_state)
        rnn_feed = {r.init_state : init_state}
=======

        init_state = sess.run(r.zero_state)
        rnn_feed = {r.init_state: init_state}
>>>>>>> 89b6a43314e18ded2ed7ab8f7e2938583d71c218
        feed_dict.update(rnn_feed)
        print('Evaluating')
        for _ in range(n_times):
            t = time.time()
            test_output = sess.run(op, feed_dict)
            print(test_output, 'Eval time:', time.time() - t)
<<<<<<< HEAD
            
=======

>>>>>>> 89b6a43314e18ded2ed7ab8f7e2938583d71c218
        print('Closing threads')
        coord.request_stop()
        coord.join(threads)

        return test_output


<<<<<<< HEAD
#Evaluation
#Confusion matrix
#Accuracy operator

x = tf.placeholder_with_default(input=logits, shape=[None, 4])
x_oh = tf.cast(tf.equal(x, tf.reduce_max(x, axis=1)[:, None]), tf.float32)[..., None]
=======
# Evaluation
# Confusion matrix
# Accuracy operator

x = tf.placeholder_with_default(input=logits, shape=[None, 4])
x_oh = tf.cast(tf.equal(x, tf.reduce_max(x, axis=1)
                        [:, None]), tf.float32)[..., None]
>>>>>>> 89b6a43314e18ded2ed7ab8f7e2938583d71c218
#x_oh = tf.equal(x, tf.reduce_max(x, axis=1)[:, None])[..., None]
y = tf.placeholder_with_default(input=label, shape=[None])
y_oh = tf.one_hot(y, depth=4)[..., None]

conf_op = tf.reduce_sum(tf.transpose(x_oh, perm=[0, 2, 1]) * y_oh,
<<<<<<< HEAD
    axis=0, name='confusion_matrix')
=======
                        axis=0, name='confusion_matrix')
>>>>>>> 89b6a43314e18ded2ed7ab8f7e2938583d71c218

print(conf_op)
x_tot = tf.reduce_sum(conf_op, axis=0)
y_tot = tf.reduce_sum(conf_op, axis=1)
correct_op = tf.diag_part(conf_op)
eps = [1e-10] * 4
<<<<<<< HEAD
acc_op = tf.reduce_mean(2*correct_op / (x_tot + y_tot + eps), name='accuracy')
print(acc_op)
measure_time(acc_op, n_times=3)

#Sparse, weighted softmax loss

class_hist = np.load('./data/class_hist.npy')
weight = tf.gather(tf.constant(1 - np.sqrt(class_hist/class_hist.sum())), label, name='weight')
=======
acc_op = tf.reduce_mean(
    2 * correct_op / (x_tot + y_tot + eps), name='accuracy')
print(acc_op)
measure_time(acc_op, n_times=3)

# Sparse, weighted softmax loss

class_hist = np.load('./data/class_hist.npy')
weight = tf.gather(
    tf.constant(
        1 -
        np.sqrt(
            class_hist /
            class_hist.sum())),
    label,
    name='weight')
>>>>>>> 89b6a43314e18ded2ed7ab8f7e2938583d71c218

class_hist, weight

loss = tf.losses.sparse_softmax_cross_entropy(label, logits, weight)
unweighted_loss = tf.losses.sparse_softmax_cross_entropy(label, logits)

measure_time([loss, unweighted_loss])

<<<<<<< HEAD
#Train operator

learning_rate = tf.Variable(initial_value=.05, trainable=False, name='learning_rate')
=======
# Train operator

learning_rate = tf.Variable(
    initial_value=.05,
    trainable=False,
    name='learning_rate')
>>>>>>> 89b6a43314e18ded2ed7ab8f7e2938583d71c218
global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
grad_clip = tf.Variable(initial_value=3., trainable=False, name='grad_clip')

with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    gvs = optimizer.compute_gradients(loss)
    with tf.name_scope('gradient_clipping'):
<<<<<<< HEAD
        capped_gvs = [(tf.clip_by_value(grad, -grad_clip, grad_clip), var) 
                      for grad, var in gvs]
        
=======
        capped_gvs = [(tf.clip_by_value(grad, -grad_clip, grad_clip), var)
                      for grad, var in gvs]

>>>>>>> 89b6a43314e18ded2ed7ab8f7e2938583d71c218
    opt = optimizer.apply_gradients(capped_gvs, global_step)


measure_time([loss, unweighted_loss, acc_op, opt], n_times=10)
