#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import pylab as plot
from sklearn import svm

label_dict = {
    'N':0,
    'A':1,
    'O':2,
    '~':3
}

#load data
data=np.load('Data.npy')
label=np.load('Label.npy')

spectrum_length=200

fourier_data=np.zeros((len(data),spectrum_length))

for ind in range(len(data)):
	print ind
	f=np.fft.fft(data[ind])
	fourier_data[ind]=np.absolute(f[:spectrum_length])

np.save('fourier_200.npy',fourier_data)
