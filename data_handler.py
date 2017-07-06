from scipy.io import loadmat
import numpy as np
from matplotlib.mlab import specgram
import torch as th
import random

from torch.autograd import Variable
from torch import optim
import torchvision

Float = th.FloatTensor
def_tokens = 'NAO'


class DataSet(th.utils.data.Dataset):
    def __init__(self, elems, load, path=None,
                 remove_noise=True, tokens=def_tokens, **kwargs):
        num_classes = len(tokens)

        super(DataSet, self).__init__()

        if isinstance(elems, str):
            with open(elems, 'r') as f:
                self.list = [line.replace('\n', '') for line in f]
        else:
            # just assume iterable
            self.list = set(elems)

        if remove_noise:
            self.list = [elem for elem in self.list if elem.find('~') == -1]

        self.class_lists = [[] for _ in range(num_classes)]

        for elem in self.list:
            label = elem.split(',')[1]
            self.class_lists[tokens.find(label)].append(elem)

        self.load = load
        self.path = path
        self.remove_noise = remove_noise
        self.tokens = tokens
        self.loadargs = kwargs

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        num_classes = len(self.tokens)
        class_idx = idx % num_classes
        idx = int(idx / num_classes) % len(self.class_lists[class_idx])
        ref = self.class_lists[class_idx][idx]

        if self.path is not None:
            return self.load("%s/%s" % (self.path, ref), **self.loadargs)

        return self.load(self.list[idx], **self.loadargs)

    def disjunct_split(self, ratio=.8):
        # Split keeps the ratio of classes
        A = set()
        for cl in self.class_lists:
            A.update(random.sample(cl, int(len(cl) * ratio)))
        B = set(self.list) - A

        A = DataSet(A, self.load, self.path, self.remove_noise,
                    self.tokens, **self.loadargs)
        B = DataSet(B, self.load, self.path, self.remove_noise,
                    self.tokens, **self.loadargs)
        return A, B


def load_mat(ref, normalize=True):
    data = loadmat(ref)['val'].squeeze()
    if normalize:
        return (data - data.mean()) / data.std()
    return data


def load_raw(line, tokens=def_tokens, **kwargs):
    # gets a line from REFERENCE.csv
    # i.e. A000001,N
    ref, label = line.split(',')
    data = load_mat(ref)
    res = {
        'x': th.from_numpy(data[None, :]),
        'y': tokens.find(label),
        'len': len(data)}
    return res


def load_crop(line, crop_len=2100, tokens=def_tokens, **kwargs):
    # Samples are recorded with 300 Hz

    ref, label = line.split(',')
    data = load_mat(ref)
    if len(data) > crop_len:
        start_idx = np.random.randint(len(data) - crop_len)
        data = data[start_idx: start_idx + crop_len]

    res = {
        'x': th.from_numpy(data[None, :]),
        'y': tokens.find(label),
        'len': len(data)}
    return res


def load_freq(line, NFFT=100, tokens=def_tokens, **kwargs):

    ref, label = line.split(',')
    data = load_mat(ref)
    Sx = specgram(
        x=data,
        NFFT=NFFT,
        Fs=300,
        noverlap=NFFT/2,
        window=np.hamming(NFFT))[0]
    res = {
        'x': th.from_numpy(Sx),
        'y': tokens.find(label),
        'len': Sx.shape[1]}
    return res


def load_freq_crop(line, NFFT=100, crop_len=900, tokens=def_tokens, **kwargs):

    ref, label = line.split(',')
    data = load_mat(ref)

    start_idx = np.random.randint(len(data) - crop_len)
    data = data[start_idx: start_idx + crop_len]

    Sx = specgram(
        x=data,
        NFFT=NFFT,
        Fs=300,
        noverlap=NFFT/1.5,
        window=np.hamming(NFFT))[0]

    res = {
        'x': th.from_numpy(Sx),
        'y': tokens.find(label),
        'len': Sx.shape[1]}
    return res


def load_norm(line, crop=True, crop_len=1000, tokens=def_tokens, **kwargs):
    ref, label = line.split(',')
    data = load_mat(ref)
    peaks = detect_beats(data, 300, lowfreq=3., highfreq=12.5)
    BPM = len(peaks) / (len(data)/300) * 60
    data = scipy.ndimage.zoom(data, BPM / avg_BPM, order=1)

    if crop:
        start_idx = np.random.randint(len(data) - crop_len)
        data = data[start_idx: start_idx + crop_len]

    res = {
        'x': th.from_numpy(data[None, :]),
        'y': self.tokens.find(label),
        'len': len(data)}
    return res


def batchify(batch):
    max_len = max(s['len'] for s in batch)
    num_channels = len(batch[0]['x'])
    x_batch = th.zeros(len(batch), num_channels, max_len)
    for idx in range(len(batch)):
        x_batch[idx, :, :batch[idx]['len']] = batch[idx]['x']

    y_batch = th.LongTensor([s['y'] for s in batch])
    len_batch = Float([s['len'] for s in batch])

    res = {'x': Variable(x_batch),
           'y': Variable(y_batch),
           'len': Variable(len_batch)}
    return res


if __name__ == '__main__':
    dataset = DataSet(
            'data/raw/training2017/REFERENCE.csv', load_raw,
            path='data/raw/training2017/', remove_noise=True, tokens='NAO')
    random.seed(42)
    train_set, eval_set = dataset.disjunct_split(.8)
    assert(len(dataset.list) == 8244)
    assert([len(cl) for cl in dataset.class_lists] == [5050, 738, 2456])
    assert([len(train_set), len(eval_set)] == [6594, 1650])
    assert(len(set(train_set.list).intersection(set(eval_set.list))) == 0)
    assert(next(iter(train_set))['len'] == 18170)
    assert(next(iter(train_set))['y'] == 0)

    train_producer = th.utils.data.DataLoader(
        dataset=train_set, batch_size=12, shuffle=True,
        num_workers=1, collate_fn=batchify)

    test_producer = th.utils.data.DataLoader(
        dataset=eval_set, batch_size=4,
        num_workers=1, collate_fn=batchify)
