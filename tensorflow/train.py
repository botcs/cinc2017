#!/usr/bin/env python3.5

# coding: utf-8

# In[1]:

import tensorflow as tf
import numpy as np

import data.ops
import model.cnn as cnn
import model.rnn as rnn
import model.fourier as fourier
import model.time_domain as time_domain
import model.classifier as classifier

import time
import os
import shutil
import json


# In[2]:

flags = tf.app.flags
flags.DEFINE_integer('gpu', 0, 'device to train on [0]')
flags.DEFINE_string('model_def', './hyperparams/test_model.json', 'load hyperparameters from ["model.json"]')
FLAGS = flags.FLAGS
FLAGS._parse_flags()

os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)


# # Model definition

# In[3]:

''' OLD DEFINITION, now model hyperparameter files are used
    
    cnn_params = {
        'out_dims' : [10, 10],
        'kernel_sizes' : 64,
        'pool_sizes' : 4
    }
    rnn_params = {
        'rnn_sizes' : [10],
        'time_steps' : 100
    }
    fc_params = {
        'fc_sizes' : [10]
    }
    batch_size = 32

    data = {
        'cnn_params' : cnn_params,
        'rnn_params' : rnn_params,
        'fc_params' : fc_params,
        'batch_size' : batch_size
    }
    with open(FLAGS.model_def, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4, sort_keys=True)
    #print(json.dumps(data, ensure_ascii=False, indent=4, sort_keys=True))
'''


# In[4]:

with open(FLAGS.model_def) as f:
    hyper_param = json.load(f)
    cnn_params = hyper_param['cnn_params']
    rnn_params = hyper_param['rnn_params']
    fc_params = hyper_param['fc_params']
    fourier_params = hyper_param['fourier_params']
    time_domain_params = hyper_param['time_domain_params']
    batch_size = hyper_param['batch_size']
    model_name = os.path.split(FLAGS.model_def)[1]
    # remove file ending
    model_name = model_name[:model_name.find('.json')]


# In[5]:

tf.reset_default_graph()
batch_size = tf.placeholder_with_default(64, [], name='batch_size')
input_op, seq_len, label = data.ops.get_batch_producer(
    batch_size=batch_size, path='./data/train.TFRecord')

c = cnn.model(seq_len=seq_len, input_op=input_op, **cnn_params)
r = rnn.get_model(batch_size=batch_size, seq_len=seq_len, input_op=c.output, **rnn_params)
f = fourier.get_output(seq_len=seq_len, input_op=input_op,**fourier_params)
td = time_domain.get_output(seq_len=seq_len, input_op=input_op,**time_domain_params)
concatenated_features=tf.concat([r.last_output,f,td], 1)
fc = classifier.model(input_op=concatenated_features, **fc_params)

logits = fc.logits
pred = fc.pred

MODEL_PATH = '/tmp/model/' + model_name + c.name + r.name + fc.name
MODEL_EXISTS = os.path.exists(MODEL_PATH)
if MODEL_EXISTS:
    print('Model directory is not empty, removing old files')
    shutil.rmtree(MODEL_PATH)


# # Time measure
# 
# convenience function

# In[3]:

def measure_time(op, feed_dict={}, n_times=10):
    with tf.Session() as sess:
        print('Sess started')
        coord = tf.train.Coordinator()
        tf.global_variables_initializer().run()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        print('Evaluating')
        for _ in range(n_times):
            t = time.time()
            fetch = sess.run(op, feed_dict)
            print(fetch, 'Eval time:', time.time() - t)
            
        print('Closing threads')
        coord.request_stop()
        coord.join(threads)

        return fetch


# # Evaluation
# 
# ## **Confusion matrix**
# 
# ## **Accuracy operator**

# In[4]:

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
        acc = tf.reduce_mean(2*correct_op / (y_tot + label_tot + eps), name='result')


# # Sparse, weighted softmax loss

# In[5]:

class_hist = np.load('./data/class_hist.npy')
with tf.name_scope('loss'):
    #weight = tf.constant([.1, 1, .2, 3])
    weight = tf.constant(1 - np.sqrt(class_hist/class_hist.sum()), name='weights')
    weight = tf.gather(weight, label, name='weight_selector')
    loss = tf.losses.sparse_softmax_cross_entropy(label, logits, weight, scope='weighted_loss')
    unweighted_loss = tf.losses.sparse_softmax_cross_entropy(label, logits, scope='unweighted_loss')
class_hist, weight


# # Train operator

# In[6]:

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


# # Summaries

# In[7]:

train_writer = tf.summary.FileWriter(MODEL_PATH, graph=tf.get_default_graph())
sum_ops = []
for v in tf.trainable_variables():
    sum_ops.append(tf.summary.histogram(v.name[:-2], v))
    sum_ops.append(tf.summary.histogram('gradients/'+v.name[:-2], tf.gradients(loss, v)))

sum_ops.append(tf.summary.scalar('weighted_loss', loss))
sum_ops.append(tf.summary.scalar('unweighted_loss', unweighted_loss))
sum_ops.append(tf.summary.scalar('accuracy', acc))
sum_ops.append(tf.summary.image('confusion_matrix', conf_op[None, ..., None], max_outputs=10))
summaries = tf.summary.merge(sum_ops)

saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
TRAIN_STEPS = 5000
with tf.Session() as sess:
    print('Sess started')
    
    print('Initializing model')
    tf.global_variables_initializer().run()
        
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    print('Training')
    for i in range(TRAIN_STEPS):
        t = time.time()
        fetch = sess.run([opt, loss, acc, global_step])
        step = fetch[-1]
        print('%d/%d'%(step, TRAIN_STEPS), 
              'time:%f'%(time.time()-t), 
              'loss:%f'%fetch[1],
              'acc:%f'%fetch[2]
              )
        if step % 10 == 0:
            print('Evaluating summaries...')
            train_writer.add_summary(summaries.eval(), global_step=fetch[-1])
        if step % 50 == 0:
            print('Saving model...')
            print(saver.save(sess, MODEL_PATH, global_step=fetch[-1]))
    
    print('Ending, closing producer threads')
    coord.request_stop()
    coord.join(threads)
