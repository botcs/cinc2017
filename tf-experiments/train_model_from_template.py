import sys
import json
import os
import numpy as np
import tensorflow as tf
from model.assembler import get_model_logits

from model import train_ops
from model import summary_ops
import data.ops


def main(argv):

    EPOCHS = 1000
    BATCH_SIZE = 8
    TRAIN_STEPS = (EPOCHS * 8525) // BATCH_SIZE

    proto_path = argv[-1]
    # proto_path = 'small_resnet.json'
    proto_name = os.path.splitext(proto_path)[0]
    proto_name = os.path.basename(proto_name)

    tf.reset_default_graph()
    network_params = json.load(open(proto_path))
    bsize = tf.placeholder_with_default(BATCH_SIZE, [], 'batch_size')
    input_op, seq_len, label_op = data.ops.get_batch_producer(
        './data/TFRecords/aug_train.TFRecord', bsize)
    # input_op = tf.placeholder(1, [None, None])
    # seq_len = tf.placeholder(tf.int32, [None])
    # label_op = tf.placeholder(tf.int32, [None])

    # for debug!!!
    summary_ops.add_label_hist(label_op)
    logits = get_model_logits(seq_len, input_op, **network_params)

    t_ops = train_ops.get_trainer(label_op, logits, batch_size=bsize)
    step, train, loss, conf, acc, lrate = t_ops

    tf.summary.histogram('sample_diversity', label_op)
    s_op = summary_ops.get_all_summaries(*t_ops)

    sv = tf.train.Supervisor(
        logdir='./training/%s/' % proto_name, summary_op=s_op,
    )
    with sv.managed_session() as sess:
        it = 0
        while not sv.should_stop() and it < TRAIN_STEPS:
            it, loss_val, acc_val, _ = sess.run([step, loss, acc, train])
            print('it:%d, loss:%02.4f, acc:%02.4f' % (it, loss_val, acc_val))


if __name__ == '__main__':
    main(sys.argv)
