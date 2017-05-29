import tensorflow as tf

INITIAL_LEARNING_RATE = 1e-4
LEARNING_RATE_DECAY = 0.5
EPOCH_SIZE = 8525
EPOCH_PER_DECAY = 100
MOVING_AVERAGE_DECAY = .999


def get_trainer(labels, logits, use_dict=False, **kwargs):
    loss, conf, acc = get_loss(labels, logits, **kwargs)

    step, train, lrate = get_update_op(loss, **kwargs)

    if use_dict:
        ret = {
            'step': step,
            'update': update,
            'loss': loss,
            'confusion': conf,
            'acc': acc,
            'learning_rate': lrate}
        return ret
    return step, train, loss, conf, acc, lrate


def get_update_op(loss, global_step=None, group=False, batch_size=8, **kwargs):
    if global_step is None:
        global_step = tf.Variable(
            initial_value=0,
            trainable=False,
            name='global_step',
            dtype=tf.int32
        )
    decay_steps = EPOCH_SIZE * EPOCH_PER_DECAY / batch_size
    learning_rate = tf.train.exponential_decay(
            learning_rate=INITIAL_LEARNING_RATE,
            global_step=global_step,
            decay_steps=decay_steps,
            decay_rate=LEARNING_RATE_DECAY,
            staircase=True,
            name='learning_rate')
    '''
    boundaries = [4000, 6000, 8000, 10000]
    values = [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    learning_rate = tf.train.piecewise_constant(
        global_step, boundaries, values, name='learning_rate')
    '''
    optimizer = tf.contrib.opt.LazyAdamOptimizer(learning_rate=learning_rate)
    grads_and_vars = optimizer.compute_gradients(loss)
    update = optimizer.apply_gradients(
        grads_and_vars=grads_and_vars, global_step=global_step)
    for grad, var in grads_and_vars:
        if grad is not None:
            tf.add_to_collection('gradients', grad)
    with tf.variable_scope('var_avgs'):
        variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step, name='variable_avg')
        variables_averages_op = variable_averages.apply(
            tf.trainable_variables())
    tf.add_to_collection('update_ops', variables_averages_op)
    with tf.control_dependencies(tf.get_collection('update_ops') + [update]):
        train_op = tf.no_op(name='train_step')

    return global_step, train_op, learning_rate


def get_loss(labels, logits, use_confusion='log', return_aux=True,
             beta=1e-4, **kwargs):

    '''Returns:
      loss: 1-accuracy if `use_confusion` is "linear" (default),
      -log(accuracy) if `use_confusion` is "log",
      if None returns sparse cross-entropy between labels and logits.

    Return also if `return_aux` is True (default):
      confusion_matrix: using smooth predictions (differentiable)
      accuracy_operator: using confusion matrix
    '''
    with tf.name_scope('accuracy_eval'):
        confusion_matrix = get_confusion(labels=labels, logits=logits, **kwargs)
        accuracy_operator = get_accuracy(confusion_matrix, **kwargs)
    with tf.name_scope('loss_eval'):
        if use_confusion is None:
            train_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels, logits=logits)
            train_loss = tf.reduce_mean(train_loss)
        elif use_confusion.lower() == 'linear':
            train_loss = 1 - accuracy_operator
        elif use_confusion.lower() == 'log':
            train_loss = -tf.log(accuracy_operator)
        else:
            raise ValueError('Invalid value for parameter `use_confusion`:%s'
                             % use_confusion)
    train_loss = tf.identity(train_loss, name='train_loss')
    tf.add_to_collection('losses', train_loss)
    with tf.name_scope('l2_losses'):
        weights = [v for v in tf.trainable_variables()
                   if v.name.find('weights') != -1]
        l2_losses = [tf.nn.l2_loss(w, name=w.name[
            :w.name.find('weights')]) for w in weights]
        for w, l in zip(weights, l2_losses):
            tf.add_to_collection('weights', w)
            tf.add_to_collection('losses', l)

    l2_loss = tf.reduce_sum(l2_losses, name='l2_loss_all')
    tf.add_to_collection('losses', l2_loss)

    total_loss = tf.identity(train_loss + beta * l2_loss, name='total_loss')
    tf.add_to_collection('losses', total_loss)
    with tf.variable_scope('stat_avg'):
        loss_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, name='ema')
        loss_averages_op = loss_averages.apply(tf.get_collection('losses'))

        for l in [total_loss, train_loss, l2_loss]:
            tf.add_to_collection('stat_avg', loss_averages.average(l))

    tf.add_to_collection('update_ops', loss_averages_op)
    with tf.control_dependencies([loss_averages_op]):
        loss = tf.identity(total_loss, name='loss_with_update')

    if return_aux:
        return loss, confusion_matrix, accuracy_operator
    return loss


def get_confusion(labels, logits, num_classes=4, differentiable=True,
                  use_softmax=True, **kwargs):
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


def get_accuracy(conf_op, eps=1e-10, **kwargs):
    y_tot = tf.reduce_sum(conf_op, axis=0, name='label_class_sum')
    label_tot = tf.reduce_sum(conf_op, axis=1, name='pred_class_sum')
    correct_op = tf.diag_part(conf_op, name='correct_class_sum')
    eps = tf.constant([1e-10] * 4, name='eps')
    acc = tf.reduce_mean(
        2 * correct_op / (y_tot + label_tot + eps), name='accuracy')

    with tf.variable_scope('stat_avg'):
        acc_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, name='ema')
        acc_averages_op = acc_averages.apply([acc])

        tf.add_to_collection('stat_avg', acc_averages.average(acc))

    tf.add_to_collection('update_ops', acc_averages_op)
    with tf.control_dependencies([acc_averages_op]):
        acc = tf.identity(acc, name='acc_with_update')
        return acc


if __name__ == '__main__':
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
