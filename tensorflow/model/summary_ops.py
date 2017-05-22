import tensorflow as tf


def get_variables_histogram(loss, merge=False):
    vargrad_hist = []
    for v in tf.trainable_variables():
        vargrad_hist.append(tf.summary.histogram(
            'variable ' + v.op.name[:-2], v), )
        vargrad_hist.append(tf.summary.histogram(
            'gradients ' + v.op.name[:-2], tf.get_collection('gradients')))
    if merge:
        return tf.summary.merge(vargrad_hist, name='summary_variables_and_gradients')
    return vargrad_hist

def get_label_hist(label):
    return tf.summary.histogram('label', label)

def add_loss_summaries(loss_averages=None):
    '''
    This function is a high level callable, it only adds summaries 
    to the graph, and does not return anything.
    
    The summaries can be merged with `tf.summary.merge` function
    '''
    
    for loss in losses:
        tf.summary.scalar(l.op.name + ' (raw)', l)
        if loss_averages:
            tf.summary.scalar(l.op.name, loss_averages.average(l))
    
def add_eval_summaries():
    
    
def get_general_summaries(loss, acc, lrate, conf=None, merge=False):    
    sum_ops = []
    sum_ops.append(tf.summary.scalar('loss', loss))
    sum_ops.append(tf.summary.scalar('accuracy', acc))
    sum_ops.append(tf.summary.scalar('learning rate', lrate))
    sum_ops.append(tf.summary.image('confusion matrix',
                                    conf[None, ..., None], max_outputs=30))

    if merge:
        return tf.summary.merge(sum_ops, name='summary_general')
    return sum_ops