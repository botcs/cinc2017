import tensorflow as tf


def variable_size_window(seq_len, x, N):
    '''
    Returns fix number `N` of equal sized windows and slices the variable
    length input
    Use `SYMMETRIC` padding if necessary.

    Returns `N` equal slices
    '''
    print('Variable sized windowing -- number of windows: %d' % N)
    print('  Dimensions: [batch_size, num_windows, window_size, num_features]')
    with tf.name_scope('sample_division'):
        if len(x.get_shape()) == 3:
            x = x[:, :, None, :]
        else:
            raise ValueError('`input_op` has incorrect number of dimensions. \
        required shape: [batch_size, sequence_length, num_features]')

        x_shape = tf.shape(x)
        batch_size, max_seq_len = x_shape[0], x_shape[1]
        # Make sure sequence can be divided to equal parts
        padding = [[0, 0], [0, N-max_seq_len % N], [0, 0], [0, 0]]
        x_pad = tf.pad(x, padding, 'SYMMETRIC')

        # Don't pad if not necessary, i.e. max_seq_len%N == 0
        new_x = tf.cond(tf.equal(max_seq_len % N, 0), lambda: x, lambda: x_pad)
        max_seq_len = tf.shape(new_x)[1]
        new_shape = [x.get_shape()[0].value,
                     N,
                     max_seq_len//N,
                     x.get_shape()[-1].value]

        div_x = tf.reshape(new_x, new_shape)

        # Convenience variable
        seq_len = tf.ones([batch_size]) * N
        print(div_x)
        return seq_len, div_x
