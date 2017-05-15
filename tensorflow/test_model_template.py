import tensorflow as tf
import model.cnn as cnn
import model.rnn as rnn
import model.classifier as classifier


def get_output(X):
    cnn_block_params = {
        'out_dims': [128, 256, 256],
        'kernel_sizes': 64,
        'pool_sizes': 1
    }

    RESIDUAL_POOL = 3

    c = cnn.model(
        seq_len=seq_len,
        input_op=input_op,
        model_name='CNN_block',
        **cnn_block_params)

    residual_input = c.output[..., None, :]

    for i in range(1, 4):
        residual_input = tf.contrib.layers.max_pool2d(
            residual_input,
            kernel_size=[RESIDUAL_POOL, 1],
            stride=[RESIDUAL_POOL, 1])

        c = cnn.model(
            seq_len=seq_len,
            input_op=residual_input,
            residual=True,
            model_name='CNN_block_%d' % i,
            **cnn_params)
        residual_input += c.output
