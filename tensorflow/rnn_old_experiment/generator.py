
# coding: utf-8

# In[1]:

import tensorflow as tf
import numpy as np
import data
import model
import time
import shutil
import io
import os
import matplotlib.pyplot as plt
# import seaborn as sns

def_learning_rate = .001
def_grad_clip = 5.
def_eval_freq = 1
def_save_freq = 20
def_gpu = 0

flags = tf.app.flags
flags.DEFINE_float(
  'rate',
  def_learning_rate,
  'Learning rate for Adam optimizer [%f]' %
  def_learning_rate)
flags.DEFINE_float(
  'clip',
  def_grad_clip,
  'Clipping gradients during backpropagation [%f]' %
  def_grad_clip)
flags.DEFINE_integer(
  'eval_freq',
  def_eval_freq,
  'Number of samples after model is evaluated on its own input [%d]' %
  def_eval_freq)
flags.DEFINE_integer(
  'save_freq',
  def_save_freq,
  'Number of samples after model checkpoint is saved [%d]' %
  def_save_freq)
flags.DEFINE_integer('gpu', def_gpu, 'GPU ID to use [%d]' % def_gpu)
FLAGS = flags.FLAGS
FLAGS._parse_flags()

os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)


# In[2]:

tf.reset_default_graph()
g = model.generator()
feeder = data.random_batch(g.batch_size)


# # Loss, images and optimizer

# In[3]:

'''
  Name scope is good for graph definition for debugging in TensorBoard
'''
global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
target = tf.placeholder(tf.float32, [g.batch_size, None, model.def_input_dim])

with tf.name_scope('linear_regression'):
  loss = tf.reduce_sum((g.outputs - target)**2)
  with tf.name_scope('total'):
    loss = tf.reduce_mean(loss)

with tf.name_scope('visualizer'):
  plot_buf_placeholder = tf.placeholder(
    tf.string, [], 'plot_buf_placeholder')
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(plot_buf_placeholder, channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)

with tf.name_scope('optimizer'):
  optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.rate)
  gvs = optimizer.compute_gradients(loss)
  with tf.name_scope('gradient_clipping'):
    capped_gvs = [(tf.clip_by_value(grad, -FLAGS.clip, FLAGS.clip), var)
            for grad, var in gvs]

  opt = optimizer.apply_gradients(capped_gvs, global_step)


# # Evaluation

# In[4]:

def eval(run_length, sess):
  res = np.random.randn(g.batch_size, run_length)

  state = sess.run(g.zero_state)
  x = np.zeros((g.batch_size, 1, model.def_input_dim))
  for i in range(run_length):
    feed_dict = {
      g.keep_prob: 1,
      g.x: x,
      g.init_state: state,
      g.seq_len: np.ones((g.batch_size))
    }
    x, state = sess.run([g.outputs, g.rnn_last_states], feed_dict)
    res[:, i] = x.squeeze()

  return res

# http://stackoverflow.com/questions/38543850/


def gen_plot(value_to_plot, num_subplots, name=None):
  """Create a pyplot plot and save to buffer."""
  x = value_to_plot.squeeze()
  fig = plt.figure(1)
  plt.clf()
  for i in range(num_subplots):
    plt.subplot(num_subplots, 1, i + 1)
    plt.plot(x[i])
  buf = io.BytesIO()
  fig.savefig(buf, dpi=150, format='png')
  if name:
    fig.savefig(name, dpi=150, format='png')
    print('\nImage saved to:%s\n' % name)

  plt.close(fig)
  buf.seek(0)

  return buf.getvalue()


# # Summaries

# In[5]:

summaries = tf.summary.merge([
  [(tf.summary.histogram(grad.name, grad),
    tf.summary.histogram(var.name, var))
   for grad, var in gvs],
  tf.summary.scalar('loss', loss)
])
im_sum = tf.summary.image('generated', image, max_outputs=10)


# # Path check

# In[6]:

sess = tf.InteractiveSession()
path = g.get_checkpoint_path()
if os.path.exists(path):
  print('  Found existing path, removing its content...')
  shutil.rmtree(path)
writer = tf.summary.FileWriter(path, graph=sess.graph)
im_path = os.path.join(path, 'demo_img')
os.mkdir(im_path)
print(writer.get_logdir())
saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
tf.global_variables_initializer().run()


# # Train test

# In[7]:

for i, batch in enumerate(feeder):
  state = sess.run(g.zero_state)
  x_feed, label_feed, lens_feed = batch
  target_feed = np.roll(x_feed, 1)

  start_time = time.time()
  for idx in range(0, lens_feed.max(), g.time_steps):
    data_window = x_feed[:, idx:idx + g.time_steps]
    target_window = target_feed[:, idx:idx + g.time_steps]
    lens_window = lens_feed - idx

    feed_dict = {
      g.x: data_window,
      g.init_state: state,
      g.seq_len: lens_window,
      target: target_window
    }
    fetch_dict = {
      'opt': opt,
      'step': global_step,
      'loss': loss,
      'state': g.rnn_last_states
    }

    start_window_time = time.time()

    # Training happens here
    fetch = sess.run(fetch_dict, feed_dict)
    state = fetch['state']
    #######################

    window_time = time.time() - start_window_time
    valpsec = g.time_steps / window_time

    if idx % (2 * g.time_steps) == 0:
      sum_eval = sess.run(summaries, feed_dict)
      writer.add_summary(summary=sum_eval, global_step=fetch['step'])
      print('\r%05d val/sec: %d' % (idx, valpsec), end='', flush=True)

  sample_time = time.time() - start_time
  print('\t it: %05d sample_time: %03.1fs loss: %f' %
      (i, sample_time, fetch['loss']))

  if i % FLAGS.eval_freq == 0:
    fname = os.path.join(im_path, '%05d.png' % fetch['step'])
    plot = gen_plot(eval(3 * g.time_steps, sess), 3, fname)
    feed_dict[plot_buf_placeholder] = plot
    writer.add_summary(im_sum.eval(feed_dict), global_step=fetch['step'])

  if i % FLAGS.save_freq == 0:
    path = saver.save(
      sess,
      g.get_checkpoint_path() +
      '/saver',
      fetch['step'])
    print('\nModel checkpoint saved to: %s\n' % path)
