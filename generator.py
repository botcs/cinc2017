import tensorflow as tf
import numpy as np
import data
import model
import time

import io
import matplotlib.pyplot as plt

def_rate = .001
def_grad_clip = 5.

flags = tf.app.flags
flags.DEFINE_float('rate', def_rate, 'Training learning rate [%f]'%def_rate)
flags.DEFINE_float('grad_clip', def_grad_clip, 'Training gradient squeeze between +/-[%f]'%def_grad_clip)
FLAGS = flags.FLAGS

with tf.variable_scope('generator'):
    # for variable reusability
    g = model.generator()
    
feeder = data.batch_pool(g.batch_size)

# Loss, images and optimizer

global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
target = tf.placeholder(tf.float32, [batch_size, None, model.def_input_dim])

with tf.name_scope('linear_regression'):    
    loss = tf.reduce_sum((g.outputs-target)**2)    
    with tf.name_scope('total'):
        loss = tf.reduce_mean(loss)

with tf.name_scope('visualizer'):
    # http://stackoverflow.com/questions/38543850/
    def gen_plot(value_to_plot, subplots):
        """Create a pyplot plot and save to buffer."""
        x = value_to_plot.squeeze()
        fig = plt.figure(1)
        plt.clf()
        for i in range(subplots):
            plt.subplot(subplots, 1, i+1)
            plt.plot(x[i])
        buf = io.BytesIO()
        fig.savefig(buf, dpi=600, format='png')
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()
    
    plot_buf_placeholder = tf.placeholder(tf.string, [], 'plot_buf_placeholder')
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(plot_buf_placeholder, channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
        
with tf.name_scope('optimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.rate)
    gvs = optimizer.compute_gradients(loss)
    with tf.name_scope('gradient_clipping'):
        capped_gvs = [(tf.clip_by_value(grad, -FLAGS.grad_clip, FLAGS.grad_clip)
            , var) for grad, var in gvs]
        
    opt = optimizer.apply_gradients(capped_gvs, global_step)
    
# Summaries  

summaries = tf.summary.merge([
    [(tf.summary.histogram(grad.name, grad), 
      tf.summary.histogram(var.name, var)) 
     for grad, var in gvs],
    tf.summary.scalar('loss', loss)
])
im_sum = tf.summary.image('generated', image, max_outputs=10)


# Evaluation

def eval():
    with tf.variable_scope('generator', reuse=True):
        
    

# Training
  
sess = tf.InteractiveSession()
writer = tf.summary.FileWriter(
    g.get_checkpoint_path(), graph=sess.graph)
print('++++LOGDIR:', writer.get_logdir())
saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
tf.global_variables_initializer().run()

for i, batch in enumerate(feeder):
    state = sess.run(g.zero_state)
    x_feed, label_feed, lens_feed = batch
    target_feed = np.roll(x_feed, 1) 
    
    start_time = time.time()
    for idx in range(0, lens_feed.max(), g.time_steps):
        data_window = x_feed[:, idx:idx+g.time_steps]
        target_window = target_feed[:, idx:idx+g.time_steps]
        lens_window = lens_feed-idx
        
        feed_dict = {
            g.x:data_window,
            g.init_state:state,
            g.seq_len:lens_window,
            target:target_window
        }
        
        # For 
        if idx == 3: image_feed_dict = feed_dict
        
        fetch_dict = {
            'opt' : opt,
            'step' : global_step,
            'loss' : loss,
            'state' : g.rnn_last_states
        }
        start_window_time = time.time()
        fetch = sess.run(fetch_dict, feed_dict)
        window_time = time.time() - start_window_time
        valpsec = g.time_steps / window_time
        state = fetch['state']
        
        if idx % (2 * g.time_steps) == 0:
            sum_eval = sess.run(summaries, feed_dict)
            writer.add_summary(summary=sum_eval, global_step=fetch['step'])
            print('\r%05d val/sec: %d'%(idx, valpsec), end='', flush=True)
            
    sample_time = time.time() - start_time
    print('\t it: %05d sample_time: %03.3fsec loss: %f'%(i, sample_time, fetch['loss']))
    outputs = g.outputs.eval(init_feed_dict)
    plot = gen_plot(outputs, 3)
    feed_dict[plot_buf_placeholder] = plot
    writer.add_summary(im_sum.eval(feed_dict), global_step=fetch['step'])
    
    
    if i%20 == 0: 
        path = saver.save(sess, g.get_checkpoint_path()+'/saver', fetch['step'])
        print('\nModel checkpoint saved to:\n', path)
    
        

