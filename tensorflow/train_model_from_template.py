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
    proto_path = argv[-1]
    #proto_path = 'small_resnet.json'
    proto_name = os.path.splitext(proto_path)[0]
    proto_name = os.path.basename(proto_name)

    tf.reset_default_graph()
    network_params = json.load(open(proto_path))
    bsize = tf.placeholder_with_default(128, [], 'batch_size')
    input_op, seq_len, label_op = data.ops.get_batch_producer(
        './data/TFRecords/aug_train.TFRecord', bsize)

    # for debug!!!
    summary_ops.add_label_hist(label_op)
    logits = get_model_logits(seq_len, input_op, **network_params)

    t_ops = train_ops.get_trainer(label_op, logits)
    step, train, loss, conf, acc, lrate, loss_avg, ver_avg = t_ops

    tf.summary.histogram('sample_diversity', label_op)
    s_op = summary_ops.get_all_summaries(*t_ops)
    
    TRAIN_STEPS = 30000


    sv = tf.train.Supervisor(
        logdir='./training/%s/' % proto_name, summary_op=s_op,
    )
    with sv.managed_session() as sess:
        it = 0
        while not sv.should_stop() and it < TRAIN_STEPS:
            it, loss_val, acc_val, _ = sess.run([step, loss, acc, train])
            print('it:%d, loss:%02.4f, acc:%02.4f'%(it, loss_val, acc_val))

if __name__ == '__main__':
    main(sys.argv)