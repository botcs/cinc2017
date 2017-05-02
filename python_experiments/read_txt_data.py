import random
import os
import numpy as np
import tensorflow as tf
import Network
from os import listdir
from os.path import isfile, join

Dir = 'text'
OnlyFiles = [f for f in listdir(Dir) if isfile(join(Dir, f))]
DataLength = 2700  # !!!get the first 3000 samples - this is not good...2714 was the length of the shortest sequence
NumData = len(OnlyFiles)
Data = np.zeros((NumData, DataLength))
for DataInd, FileName in enumerate(OnlyFiles):
    with open(Dir + '/' + FileName) as f:
    Lines = f.readlines()
    Parts = Lines[0].split(",")
    for Ind in range(DataLength):
    Data[DataInd][Ind] = float(Parts[Ind])

np.save('Data2700', Data)
# Data=np.load('Data2700')
