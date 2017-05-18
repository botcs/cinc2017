import tensorflow as tf


def get_update_op(loss, accuracy, global_step=None):
    if global_step is None:
        global_step = tf.Variable(
            initial_value=0,
            trainable=False,
            name='global_step')

    boundaries = [4000, 6000, 8000, 10000]
    values = [0.001, 0.0005, 0.0001, 0.00005, 0.0001]
    learning_rate = tf.train.piecewise_constant(
        global_step, boundaries, values)
    optimizer = tf.contrib.opt.LazyAdamOptimizer(learning_rate=learning_rate)
    update = optimizer.minimize(loss)
    return tf.tuple([global_step, loss, accuracy], 'update_weights', update)


def get_loss(labels, logits, use_confusion='linear', return_aux=True,
             lamb=1e-4, **kwargs):

    '''Returns:
      loss: 1-accuracy if `use_confusion` is "linear" (default),
      -log(accuracy) if `use_confusion` is "log",
      if None returns sparse cross-entropy between labels and logits.

    Return also if `return_aux` is True (default):
      confusion_matrix: using smooth predictions (differentiable)
      accuracy_operator: using confusion matrix
    '''
    confusion_matrix = get_confusion(labels=labels, logits=logits, **kwargs)
    accuracy_operator = get_accuracy(confusion_matrix)
    with tf.name_scope('loss'):
        if use_confusion.lower() == 'linear':
            train_loss = 1 - accuracy_operator
        elif use_confusion.lower() == 'log':
            train_loss = -tf.log(accuracy_operator)
        elif use_confusion is None:
            train_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels, logits=logits)
        else:
            raise ValueError('Invalid value for parameter `use_confusion`:%s'
                             % use_confusion)

        l2_loss = tf.reduce_sum([tf.nn.l2_loss(v, name='L2_reg_loss')
                                 for v in tf.trainable_variables()])

        loss = unweighted_loss + lamb * l2_loss

    if return_aux:
        return loss, confusion_matrix, accuracy_operator
    return loss


def get_confusion(labels, logits, num_classes=4, differentiable=False,
                  use_softmax=True):
    y = logits
    if use_softmax:
        y = tf.nn.softmax(logits)
    if not differentiable:
        y = tf.one_hot(tf.arg_max(y, 1), depth=num_classes)

    label_oh = tf.one_hot(labels, depth=num_classes)
    with tf.name_scope('confusion'):
        conf_op = tf.reduce_sum(tf.transpose(
            y[..., None], perm=[0, 2, 1]) * label_oh[..., None],
            axis=0, name='confusion_matrix')

    return conf_op


def get_accuracy(conf_op, eps=1e-10):
    with tf.name_scope('accuracy_eval'):
        y_tot = tf.reduce_sum(conf_op, axis=0, name='label_class_sum')
        label_tot = tf.reduce_sum(conf_op, axis=1, name='pred_class_sum')
        correct_op = tf.diag_part(conf_op, name='correct_class_sum')
        eps = tf.constant([1e-10] * 4, name='eps')
        acc = tf.reduce_mean(
            2 * correct_op / (y_tot + label_tot + eps), name='accuracy')

    return acc


''' TEST SCRIPT'''
tf.reset_default_graph()
preds = tf.placeholder(1, [None, 4])
labels = tf.placeholder(tf.int32, [None])
num_classes = 4

conf = get_confusion(labels, preds, differentiable=False, use_softmax=False)
acc = get_accuracy(conf)

diff_conf = get_confusion(labels, preds,
                          differentiable=True, use_softmax=False)
diff_acc = get_accuracy(diff_conf)

with tf.Session() as sess:
    feed_dict = {
            preds: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            labels: [0, 1, 2, 3]}
    print(acc.eval(feed_dict))
    print(diff_acc.eval(feed_dict))
''''''
