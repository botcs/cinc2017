import tensorflow as tf

# # Evaluation
#
# ## **Confusion matrix**
#
# ## **Accuracy operator**


def get_update_op(logits, loss):
    pass


def get_loss(logits, labels):
    pass


def get_accuracy(logits, labels):
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
