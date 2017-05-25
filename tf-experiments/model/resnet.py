import tensorflow as tf
from . import cnn


def _get_FCN_block(conv_in, kernel_sizes, out_dims, is_training,
                   avg_pool=False, RESIDUAL_POOL=1, **block_params):

    for it, (k_size, out_c) in enumerate(zip(kernel_sizes, out_dims)):
        in_c = conv_in.get_shape()[-1].value
        with tf.variable_scope('Conv%d' % it):
            bn = tf.layers.batch_normalization(
                conv_in, scale=False, training=is_training)
            relu = tf.nn.relu(bn)
            n = 2 * (k_size * in_c * out_c) ** .5
            W = tf.Variable(
                tf.truncated_normal([k_size, in_c, out_c])/n,
                collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'weights'],
                name='weights')
            conv_out = tf.nn.conv1d(relu, W, 1, 'SAME')
            print(conv_out)
            tf.add_to_collection('activations', conv_out)
            conv_in = conv_out

    if RESIDUAL_POOL > 1:
        if avg_pool:
            pool_fn = tf.layers.average_pooling1d
        else:
            pool_fn = tf.layers.max_pooling1d
        name = '%s_POOL_%d--factor' % (
            'AVG' if avg_pool else 'MAX', RESIDUAL_POOL)
        conv_out = pool_fn(
            conv_out,
            pool_size=RESIDUAL_POOL,
            strides=RESIDUAL_POOL)

    return conv_out


def get_resnet_output(
    seq_len, input_op, avg_pool=False,
    block_num=None, **resnet_params):

    is_training = tf.get_collection('inference_vars')[0]
    if block_num is None:
        raise ValueError('`block_num` must be defined for ResNet parameters')

    if len(input_op.get_shape()) == 2:
        input_op = input_op[..., None]
    else:
        raise ValueError('`input_op` has incorrect number of dimensions. \
required shape: [batch_size, sequence_length]')

    res_in = input_op
    # Model assembly
    for i in range(block_num):
        block_params = resnet_params['block%d' % i]
        pool_factor = block_params['RESIDUAL_POOL']
        conv_in = res_in
        # if dimensions are not compatible, tf will catch the error
        with tf.variable_scope('FCN_block_%d' % i):
            block_params['is_training'] = is_training
            conv_out = _get_FCN_block(conv_in, **block_params)
            conv_in = conv_out
            seq_len /= pool_factor

        with tf.variable_scope('shortcut_connection_%d' % i):
            # https://arxiv.org/pdf/1512.03385.pdf
            # using modified B) shortcut
            # since model is small
            # ...
            in_c = res_in.get_shape()[-1].value
            out_c = conv_out.get_shape()[-1].value
            n = 2 * (pool_factor * in_c * out_c) ** .5
            W = tf.Variable(
                tf.random_normal([pool_factor, in_c, out_c])/n,
                collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'weights'],
                name='weights')
            shortcut = tf.nn.conv1d(res_in, W, pool_factor, 'VALID')

        res_in = tf.nn.relu(shortcut + conv_out)

        print('')
    res_out = res_in

    return seq_len, res_out
