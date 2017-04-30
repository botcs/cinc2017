#! /usr/bin/env ipython
import tensorflow as tf
import numpy as np

import data.ops
import model.cnn as cnn
import model.rnn as rnn
import model.classifier as classifier

import time
import os
import shutil


# In[2]:

flags = tf.app.flags
flags.DEFINE_integer('gpu', 0, 'device to train on [0]')
FLAGS = flags.FLAGS
FLAGS._parse_flags()

os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)


# In[3]:

paths = [
  './data/NORMAL_CLASS_REF.TFRecord',
  './data/OTHER_CLASS_REF.TFRecord',
  './data/ATRIUM_CLASS_REF.TFRecord',
  './data/NOISE_CLASS_REF.TFRecord'
]


# In[4]:

tf.reset_default_graph()
batch_size = tf.placeholder_with_default(16, [], name='batch_size')
(input_op, seq_len, label), input_prods = data.ops.get_even_batch_producer(
  paths=paths, batch_size=batch_size)


# In[5]:

cnn_params = {
  'out_dims': [256, 512, 512],
  'kernel_sizes': 64,
  'pool_sizes': 1
}
c = cnn.model(seq_len=seq_len, input_op=input_op, **cnn_params)

#a = tf.transpose(c.output, perm=[0, 2, 1])
#a = tf.nn.top_k(a, k=8, sorted=False, name='MAX_POOL').values
#a = tf.transpose(a, perm=[0, 2, 1])
a = tf.reduce_mean(c.output, axis=1)
fc = classifier.model(input_op=a, fc_sizes=[])

logits = fc.logits
pred = fc.pred

MODEL_PATH = '/tmp/balanced/' + c.name + fc.name
MODEL_EXISTS = os.path.exists(MODEL_PATH)
if MODEL_EXISTS:
  print('Model directory is not empty, removing old files')
  shutil.rmtree(MODEL_PATH)


# In[6]:

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
      print('Eval time:', time.time() - t)

    print('Closing threads')
    coord.request_stop()
    coord.join(threads)

    return fetch


# In[7]:

measure_time(label)


# # Evaluation
#
# ## **Confusion matrix**
#
# ## **Accuracy operator**

# In[8]:

with tf.name_scope('evaluation'):
  with tf.name_scope('one_hot_encoding'):
    y_oh = tf.cast(tf.equal(
      logits, tf.reduce_max(logits, axis=1)[:, None]), tf.float32)

    label_oh = tf.one_hot(label, depth=4)
  with tf.name_scope('confusion_matrix'):
    conf_op = tf.reduce_sum(tf.transpose(
      y_oh[..., None], perm=[0, 2, 1]) * label_oh[..., None],
      axis=0, name='result')

  with tf.name_scope('accuracy'):
    y_tot = tf.reduce_sum(conf_op, axis=0, name='label_class_sum')
    label_tot = tf.reduce_sum(conf_op, axis=1, name='pred_class_sum')
    correct_op = tf.diag_part(conf_op, name='correct_class_sum')
    eps = tf.constant([1e-10] * 4, name='eps')
    acc = tf.reduce_mean(
      2 * correct_op / (y_tot + label_tot + eps), name='result')


# In[9]:

class_hist = np.load('./data/class_histogramTRAIN.npy')
with tf.name_scope('loss'):
  #weight = tf.constant([.1, 1, .2, 3])
  weight = tf.constant(
    1 -
    np.sqrt(
      class_hist /
      class_hist.sum()),
    name='weights')
  weight = tf.gather(weight, label, name='weight_selector')
  train_loss = tf.losses.softmax_cross_entropy(
    label_oh, logits, weight, scope='weighted_loss')
  unweighted_loss = tf.losses.softmax_cross_entropy(
    label_oh, logits, scope='unweighted_loss')

  l2_loss = tf.reduce_sum([tf.nn.l2_loss(v, name='L2_reg_loss')
               for v in tf.trainable_variables()])
  beta = 0.0001
  loss = unweighted_loss + beta * l2_loss
#class_hist, weight


# In[10]:

with tf.name_scope('train'):
  learning_rate = tf.Variable(
    initial_value=.001,
    trainable=False,
    name='learning_rate')
  global_step = tf.Variable(
    initial_value=0,
    trainable=False,
    name='global_step')
  grad_clip = tf.Variable(
    initial_value=3.,
    trainable=False,
    name='grad_clip')
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
  gvs = optimizer.compute_gradients(loss)
  with tf.name_scope('gradient_clipping'):
    capped_gvs = [(tf.clip_by_value(grad, -grad_clip, grad_clip), var)
            for grad, var in gvs]

  opt = optimizer.apply_gradients(capped_gvs, global_step)

  #opt = optimizer.minimize(1-max)


# In[11]:

train_writer = tf.summary.FileWriter(MODEL_PATH, graph=tf.get_default_graph())
sum_ops = []
for v in tf.trainable_variables():
  sum_ops.append(tf.summary.histogram(v.name[:-2], v))
  sum_ops.append(tf.summary.histogram(
    'gradients/' + v.name[:-2], tf.gradients(loss, v)))

sum_ops.append(tf.summary.scalar('weighted_loss', loss))
sum_ops.append(tf.summary.scalar('unweighted_loss', unweighted_loss))
sum_ops.append(tf.summary.scalar('accuracy', acc))
sum_ops.append(tf.summary.image('confusion_matrix',
                conf_op[None, ..., None], max_outputs=10))
summaries = tf.summary.merge(sum_ops)
eval_summaries = tf.summary.merge([tf.summary.scalar('eval_accuracy', acc), tf.summary.image(
  'confusion_matrix', conf_op[None, ..., None], max_outputs=10)])


# In[12]:

saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
# with open('test.txt', 'w') as f:
#metagraph = saver.export_meta_graph(as_text=True)
# f.write(str(metagraph.ListFields()))

TRAIN_STEPS = 200000
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
    print('%d/%d' % (step, TRAIN_STEPS),
        'time:%f' % (time.time() - t),
        'loss:%f' % fetch[1],
        'acc:%f' % fetch[2]
        )
    if step % 20 == 0:
      print('Evaluating TRAIN summaries...')
      train_writer.add_summary(summaries.eval(), global_step=fetch[-1])
    if step % 1000 == 0:
      print('Saving model...')
      print(saver.save(sess, MODEL_PATH, global_step=fetch[-1]))

  print('Ending, closing producer threads')
  coord.request_stop()
  coord.join(threads)
