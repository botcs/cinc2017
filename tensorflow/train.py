#!/usr/bin/env python3.5
import tensorflow as tf
import numpy as np

import model.cnn as cnn
import model.rnn as rnn
import model.classifier as classifier
import time
import data.ops

# # Model definition

tf.reset_default_graph()
batch_size = tf.placeholder_with_default(32, [], name='batch_size')    
cnn_params = {
    'out_dims' : [10, 10, 10],
    'kernel_sizes' : 32,
    'pool_sizes' : 2
}
rnn_params = {
    'rnn_sizes' : [10],
    'time_steps' : 100
}
fc_params = {
    'fc_sizes' : [30, 20, 10]
}

input_op, seq_len, label = data.ops.get_batch_producer(
    batch_size=batch_size, path='./data/train.TFRecord')

c = cnn.get_output(seq_len=seq_len, input_op=input_op, **cnn_params)
r = rnn.get_model(batch_size=batch_size, seq_len=seq_len, input_op=c, **rnn_params)
logits, pred = classifier.get_logits_and_pred(input_op=r.last_output, **fc_params)


# # Evaluation
# 
# ## **Confusion matrix**
# 
# ## **Accuracy operator**

with tf.name_scope('evaluation'):
    with tf.name_scope('one_hot_encoding'):
        y_oh = tf.cast(tf.equal(
            logits, tf.reduce_max(logits, axis=1)[:, None]), tf.float32)[..., None]

        label_oh = tf.one_hot(label, depth=4)[..., None]
    with tf.name_scope('confusion_matrix'):
        conf_op = tf.reduce_sum(tf.transpose(y_oh, perm=[0, 2, 1]) * label_oh,
            axis=0, name='result')

    with tf.name_scope('accuracy'):
        y_tot = tf.reduce_sum(conf_op, axis=0, name='label_class_sum')
        label_tot = tf.reduce_sum(conf_op, axis=1, name='pred_class_sum')
        correct_op = tf.diag_part(conf_op, name='correct_class_sum')
        eps = tf.constant([1e-10] * 4, name='eps')
        acc_op = tf.reduce_mean(2*correct_op / (y_tot + label_tot + eps), name='result')





# Sparse, weighted softmax loss

class_hist = np.load('./data/class_hist.npy')
with tf.name_scope('loss'):
    weight = tf.constant(1 - np.sqrt(class_hist/class_hist.sum()), name='weights')
    weight = tf.gather(weight, label, name='weight_selector')
    loss = tf.losses.sparse_softmax_cross_entropy(label, logits, weight, scope='weighted_loss')
    unweighted_loss = tf.losses.sparse_softmax_cross_entropy(label, logits, scope='unweighted_loss')
class_hist, weight


# Train operator

with tf.name_scope('train'):
    learning_rate = tf.Variable(initial_value=.05, trainable=False, name='learning_rate')
    global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
    grad_clip = tf.Variable(initial_value=3., trainable=False, name='grad_clip')
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    gvs = optimizer.compute_gradients(loss)
    with tf.name_scope('gradient_clipping'):
        capped_gvs = [(tf.clip_by_value(grad, -grad_clip, grad_clip), var) 
                      for grad, var in gvs]
        
    opt = optimizer.apply_gradients(capped_gvs, global_step)

# Summaries

train_writer = tf.summary.FileWriter('/tmp/model/hist/', graph=tf.get_default_graph())
for v in tf.trainable_variables():
    tf.summary.histogram(v.name, v)
    tf.summary.histogram(v.name + '/gradients', tf.gradients(loss, v))

tf.summary.scalar('weighted_loss', loss)
tf.summary.scalar('unweighted_loss', unweighted_loss)
tf.summary.scalar('accuracy', acc_op)
tf.summary.image('confusion_matrix', conf_op[None, ..., None])
summaries = tf.summary.merge_all()

with tf.Session() as sess:
    print('Sess started')
    coord = tf.train.Coordinator()
    tf.global_variables_initializer().run()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    feed_dict={}
    init_state = sess.run(r.zero_state)
    rnn_feed = {r.init_state : init_state}
    feed_dict.update(rnn_feed)
    print('Evaluating')
    for i in range(10):
        t = time.time()
        test_output = sess.run([opt, loss, summaries], feed_dict)
        train_writer.add_summary(test_output[2], i)
        print('%d/10'%i, 'time: %f'%(time.time()-t), 'loss: %f'%test_output[1])

    print('Closing threads')
    coord.request_stop()
    coord.join(threads)

