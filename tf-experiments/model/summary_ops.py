import tensorflow as tf


def get_all_summaries(step, train, loss, conf, acc, lrate, **kwargs):
    '''High level function, adds currently defined summaries to the graph
    merges the operations, and returns the summary operator'''
    add_variables_histogram()
    add_loss_summaries()
    add_activation_summaries()
    add_eval_summaries(acc, lrate, conf)
    add_smooth_summaries()

    return tf.summary.merge_all()


def add_smooth_summaries(avgs=tf.get_collection('stat_avgs')):
    for a in avgs:
        tf.summary.scalar('smoothed/'+a.op.name, a)


def add_variables_histogram():
    grads_and_vars = zip(
        tf.get_collection('gradients'),
        tf.trainable_variables())

    for g, v in grads_and_vars:
        tf.summary.histogram(
            v.op.name + '/gradient', g)
        tf.summary.histogram(
            v.op.name + '/variable', v)


def add_label_hist(label):
    tf.summary.histogram('label', label)


def add_loss_summaries(losses=tf.get_collection('losses')):
    '''
    This function is a high level callable, it only adds summaries
    to the graph, and does not return anything.

    The summaries can be merged with `tf.summary.merge` function
    '''

    # train_loss = [l for l in losses if l.name.find('train_loss') > -1][0]
    # l2_loss = [l for l in losses if l.name.find('l2_loss_all') > -1][0]
    # total_loss = [l for l in losses if l.name.find('total_loss') > -1][0]
    # losses = [train_loss, l2_loss, total_loss]
    for l in losses:
        tf.summary.scalar('raw_losses/' + l.op.name, l)


def add_activation_summaries(activations=tf.get_collection('activations')):

    for ac in activations:
        tf.summary.histogram('activations/' + ac.op.name, ac)
        tf.summary.scalar('sparsity/' + ac.op.name,
                          tf.nn.zero_fraction(ac))


def add_eval_summaries(acc, lrate, conf):
    sum_ops = []
    sum_ops.append(tf.summary.scalar('accuracy', acc))
    sum_ops.append(tf.summary.scalar('decreasing_learning_rate', lrate))
    sum_ops.append(tf.summary.image('confusion_matrix',
                                    conf[None, ..., None], max_outputs=30))
    for s in sum_ops:
        tf.add_to_collection('summary/general', s)
