from scipy.io import loadmat
import numpy as np
#from matplotlib.mlab import specgram
from scipy import signal
import torch as th
import random

from torch.autograd import Variable
from torch import optim
import torchvision

Float = th.FloatTensor
def_tokens = 'NAO'


class mystr(str):
    def find(self, s):
        if s == self[0]:
            return 0
        return 1
noise_tokens = mystr('~~')
atrif_tokens = mystr('AA')
other_tokens = mystr('OO')
normal_tokens = mystr('NN')


class DataSet(th.utils.data.Dataset):
    def __init__(self, elems, load, path=None,
                 remove_unlisted=False, tokens=def_tokens, **kwargs):
        num_classes = len(tokens)
        super(DataSet, self).__init__()

        if isinstance(elems, str):
            with open(elems, 'r') as f:
                self.list = [line.replace('\n', '') for line in f]
        else:
            # just assume iterable
            self.list = set(elems)

        if kwargs.get('remove_noise'):
            self.list = [elem for elem in self.list if elem.find('~') == -1]

        if remove_unlisted:
            self.list = [elem for elem in self.list if tokens.find(elem[-1]) != -1]

        self.class_lists = [[] for _ in range(num_classes)]

        for elem in self.list:
            label = elem.split(',')[1]
            self.class_lists[tokens.find(label)].append(elem)


        self.remove_unlisted = remove_unlisted
        self.load = load
        self.path = path
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
            return self.load("%s/%s" % (self.path, ref), tokens=self.tokens, **self.loadargs)

        return self.load(self.list[idx], tokens=self.tokens, **self.loadargs)

    def disjunct_split(self, ratio=.8):
        # Split keeps the ratio of classes
        A = set()
        for cl in self.class_lists:
            A.update(random.sample(cl, int(len(cl) * ratio)))
        B = set(self.list) - A

        A = DataSet(A, self.load, self.path, self.remove_unlisted,
                    self.tokens, **self.loadargs)
        B = DataSet(B, self.load, self.path, self.remove_unlisted,
                    self.tokens, **self.loadargs)
        return A, B


def load_mat(ref, normalize=True):
    mat = loadmat(ref)
    data = mat['val'].squeeze()[None]
    #features = mat['features'][0, -5:]
    #features = np.concatenate(features, axis=1).squeeze().astype(np.float32)
    if normalize:
        data = (data - data.mean()) / data.std()
        #features = (features - features.mean()) / features.std()
        #features = (features - features.min()) / features.max()

    return data #, features Crop:
    def __init__(self, crop_len):
        self.crop_len = crop_len

    def __call__(self, data):
        crop_len = self.crop_len
        if len(data[0]) > crop_len:
            start_idx = np.random.randint(len(data[0]) - crop_len)
            data = data[:, start_idx: start_idx + crop_len]
        return data


class Crop:
    def __init__(self, crop_len):
        self.crop_len = crop_len

    def __call__(self, data):
        crop_len = self.crop_len
        if len(data[0]) > crop_len:
            start_idx = np.random.randint(len(data[0]) - crop_len)
            data = data[:, start_idx: start_idx + crop_len]
        return data

class Threshold:
    def __init__(self, threshold=None, sigma=None):
        assert bool(threshold is None) != bool(sigma is None),\
            (bool(threshold is None), bool(sigma is None))
        self.thr = threshold
        self.sigma = sigma


    def __call__(self, data):
        if self.sigma is None:
            data[np.abs(data) > self.thr] = self.thr
        else:
            data[np.abs(data) > data.std()*self.sigma] = data.std()*self.sigma
        return data


class RandomMultiplier:
    def __init__(self, multiplier=-1.):
        self.multiplier = multiplier
    def __call__(self, data):
        multiplier = self.multiplier if random.random() < .5 else 1.
        return data * multiplier

class Logarithm:
    def __call__(self, data):
        return np.log(data)


class Spectogram:
    def __init__(self, NFFT=None, overlap=None):
        self.NFFT = NFFT
        self.overlap = overlap
        if overlap is None:
            self.overlap = NFFT - 1
    def __call__(self, data):
        data = data.squeeze()
        assert len(data.shape) == 1
        Sx = signal.spectrogram(
            x=data,
            nperseg=self.NFFT,
            noverlap=self.overlap)[-1]
        return Sx


def load_composed(line, tokens=def_tokens, transformations=[], **kwargs):
    ref, label = line.split(',')
    data = load_mat(ref)
    data_len = len(data[0])
    for trans in transformations:
        data = trans(data)

    res = {
        'x': th.from_numpy(np.float32(data[None, :])),
        #'features': th.from_numpy(data[None, :]),
        'y': tokens.find(label),
        'len': data_len}
    return res


def batchify(batch):
    max_len = max(s['x'].size(-1) for s in batch)
    num_channels = batch[0]['x'].size(0)
    x_batch = th.zeros(len(batch), num_channels, max_len)
    for idx in range(len(batch)):
        #print(x_batch.size(), batch[idx]['x'].size())
        x_batch[idx, :, :batch[idx]['x'].size(-1)] = batch[idx]['x']

    y_batch = th.LongTensor([s['y'] for s in batch])
    #feature_batch = th.cat([s['features'] for s in batch], dim=0)


    res = {'x': Variable(x_batch),
           'y': Variable(y_batch)
          }
    return res


def load_forked(line, global_transforms, fork_transforms, tokens=def_tokens, **kwargs):
    ref, label = line.split(',')
    in_data = load_mat(ref)
    for trans in global_transforms:
        in_data = trans(in_data)
    forks = {}
    for forkname, transforms in fork_transforms.items():
        data = in_data.copy()
        for trans in transforms:
            data = trans(data)
        assert len(data.shape) < 3, data.shape
        if len(data.shape) == 1:
            data = data[None, :]
        forks[forkname] = {
            'x': th.from_numpy(np.float32(data)), 
            'y': tokens.find(label)}
    return forks

def batchify_forked(batch):
    forked_res = {}
    for key in batch[0].keys():
        #print(key)
        forked_res[key] = batchify(list(sample[key] for sample in batch))

    res = {'x': {}}
    for key, val in forked_res.items():
        res['x'][key] = val['x']
    # Every forks `y` is the same (at least should be) 
    res['y'] = val['y']
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
