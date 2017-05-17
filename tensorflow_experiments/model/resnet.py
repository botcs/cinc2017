import tensorflow as tf
import cnn


def get_resnet_output(seq_len, input_op, squeeze=True, avg_pool=False,
                      block_num=None, **resnet_params):

    if block_num is None:
        raise ValueError('`block_num` must be defined for ResNet parameters')

    if avg_pool:
        pool_fn = tf.contrib.layers.avg_pool2d
    else:
        pool_fn = tf.contrib.layers.max_pool2d

    if len(input_op.get_shape()) == 2:
        residual_input = input_op[..., None, None]
    else:
        raise ValueError('`input_op` has incorrect number of dimensions. \
required shape: [batch_size, sequence_length]')

    # Model assembly
    for i in range(block_num):
        cnn_block_params = resnet_params['block%d' % i]
        try:
            RESIDUAL_POOL = cnn_block_params['RESIDUAL_POOL']
        except KeyError:
            RESIDUAL_POOL = 1
        if cnn_block_params.get('pool_sizes') is None:
            cnn_block_params['pool_sizes'] = 1
        elif cnn_block_params.get('pool_sizes') == 1:
            pass
        else:
            raise ValueError('Residual blocks should not use internal pooling')

        c = cnn.model(
            seq_len=seq_len,
            input_op=residual_input,
            residual=True,
            model_name='CNN_block_%d' % i,
            **cnn_block_params)

        residual_input += c.output

        if RESIDUAL_POOL > 1:
            print('%s_POOL_%d--factor-%d' % (
                'AVG' if avg_pool else 'MAX', i, RESIDUAL_POOL))
            residual_input = pool_fn(
                c.output,
                kernel_size=[RESIDUAL_POOL, 1],
                stride=[RESIDUAL_POOL, 1])
            seq_len /= RESIDUAL_POOL
        print('')
    res_out = residual_input
    res_out_squeeze = tf.squeeze(residual_input, axis=2)

    return seq_len, res_out_squeeze if squeeze else res_out
