import json
import numpy as np
from scipy.io import loadmat
import tensorflow as tf
from time import time, sleep


from model.assembler import get_model_logits


class data_handler:

    def __init__(self, basename, fname_reference='RECORDS',
                 answers='answers.txt'):
        self.basename = basename
        self.fname_reference = fname_reference
        self.answers = answers
        self.processed = set()
        self.waiting = set()
        self.last_update = time()
        self.start_time = time()

    def map_label(label):

        str2idx = {'N': 0, 'A': 1, 'O': 2, '~': 3}
        idx2str = {v: k for v, k in str2idx.items()}
        if type(label) == str:
            return str2idx[label]
        elif type(label) == int:
            return ind2str[label]

    def t_since_update(self):
        return time() - self.last_update

    def next(self):
        return loadmat(self.waiting.pop())

    def write(self, fname, logits):
        assert fname in self.waiting and fname not in self.processed
        with open(self.answers, 'a') as f:
            f.write(fname + ',' map_label(max(logits)))

        self.waiting.remove(fname)
        self.processed.add(fname)

    def check(self):
        with open('RECORDS') as f:
            fnames = {fn for fn in f.read().split('\n') if len(fn) > 0}

        new_names = fnames - self.processed - self.waiting
        if len(new_names) > 0:
            self.last_update = time()
            self.waiting.update(new_names)




def main(argv):
    proto_path = argv[-1]
    proto_name = os.path.splitext(proto_path)[0]
    proto_name = os.path.basename(proto_name)

    tf.reset_default_graph()
    network_params = json.load(open(proto_path))
    tf.placeholder()
    logits = get_model_logits(seq_len, input_op, **network_params)

    sv = tf.train.Supervisor(
        logdir='./training/%s/' % proto_name, summary_op=s_op,
    )

    with sv.managed_session() as sess:
        start_time = time()
        t = time() - start_time
        while not sv.should_stop():


            sleep(.5)


if __name__ == '__main__':
    main(sys.argv)
