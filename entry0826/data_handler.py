from scipy.io import loadmat
import numpy as np
#from matplotlib.mlab import specgram
import random

def load_mat(ref, normalize=True):
    mat = loadmat(ref)
    data = mat['val'].squeeze()[None]
    #features = mat['features'][0, -5:]
    #features = np.concatenate(features, axis=1).squeeze().astype(np.float32)
    if normalize:
        data = (data - data.mean()) / data.std()
        #features = (features - features.mean()) / features.std()
        #features = (features - features.min()) / features.max()

    return data #, features


def load_composed(line, transformations=[], **kwargs):
    ref = line
    data = load_mat(ref)
    data_len = len(data[0])
    for trans in transformations:
        data = trans(data)
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
'''
class Spectogram:
    def __init__(self, NFFT, overlap=None):
        self.NFFT = NFFT
        self.overlap = overlap
        if overlap is None:
            self.overlap = NFFT / 2
    def __call__(self, data):
        data = data.squeeze()
        assert len(data.shape) == 1
        Sx = specgram(
            x=data,
            NFFT=self.NFFT,
            Fs=300,
            noverlap=self.NFFT/2,
            window=np.hamming(self.NFFT))[0]
        return Sx
'''
