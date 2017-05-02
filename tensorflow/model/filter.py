import tensorflow as tf


def lfilt(b, a, x, batch_samples=True):

    if len(b.shape) == 1:
        b = b[:, None]
    if len(a.shape) == 1:
        a = a[:, None]
    if len(x.shape) == 1:
        # single sample with single channel
        batch_samples = False
        x = x[:, None]
    if len(x.shape) == 2:
        if batch_samples:
            # [batch_size, seq_len, 1]
            x = x[:, :, None]
        else:
            # [1, seq_len, channels]
            x = x[None, :, :]

    if len(a.shape) != 2 or len(b.shape) != 2:
        raise ValueError(
            'Filter dimension(s) must be the following: [filter_length, num_filters] or [filter_length]')

    if len(x.shape) != 3:
        raise ValueError(
            'Sample dimension(s) must be the following: [batch_size, sequnce_length, channels] or [batch_size, sequnce_length] if `batch_samples` is True or [sequnce_length, channels] otherwise')

    b /= a[0]
    a /= a[0]

    a = a[:0:-1]
    b = b[::-1]

    batch_size = tf.shape(x)[0]
    sequence_length = tf.shape(x)[1]
    N = tf.shape(b)[0]
    M = tf.shape(a)[0]
    num_filters = tf.shape(a)[1]
    edge = tf.maximum(N, M)

    y = tf.zeros(dtype=tf.float32,
                 shape=[batch_size, edge, num_filters],
                 name='y_zeros_init')

    #x_padded = tf.pad(x, [[0, 0], [edge, edge], [0, 0]], mode='SYMMETRIC')
    x_padded = tf.pad(x, [[0, 0], [edge, edge], [0, 0]], mode='CONSTANT')

    def prod(a, b, **kwargs):
        return tf.reduce_sum(a * b, axis=1, **kwargs)

    def body(y, t):
        y_curr = prod(x_padded[:, t:t + N:], [b]) - prod(y[:, -M:], [a])
        y = tf.concat([y, y_curr[:, None, :]], axis=1, name='y')

        t += 1
        return y, t

    def cond(y, t):
        return t < sequence_length

    wl = tf.while_loop(cond, body, [y, 0],
                       parallel_iterations=1, back_prop=False)
    filt = wl[0][:, edge:]

    return filt


def filtfilt(b, a, x):
    filt = lfilt(b, a, x)
    filtfilt = lfilt(b, a, filt[:, ::-1])[:, ::-1]

    return filtfilt
