import data_handler
import model as M
import dilated_model as DM
import trainer as T

import numpy as np
import torch as th
from torch.autograd import Variable
import pickle
import random
random.seed(42)
import sys, os
os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[1]
import time
from os.path import basename, splitext

th.multiprocessing.set_sharing_strategy('file_system')
name = splitext(basename(sys.argv[0]))[0]

transformations = [
    data_handler.Crop(1800),
    data_handler.Threshold(sigma=2.2),
    data_handler.RandomMultiplier(-1),
]

dataset = data_handler.DataSet(
    'data/marci_features/REFERENCE.csv', data_handler.load_composed,
    transformations=transformations,
    path='data/marci_features/',
    remove_noise=True, tokens='NAO')
train_set, eval_set = dataset.disjunct_split(.9)
net = DM.VGG16(1, 
    channels=[32, 32,  64, 64,  128, 128, 128,  256, 256, 256,  256, 256, 256,  128, 64], 
    dilations=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

train_producer = th.utils.data.DataLoader(
        dataset=train_set, batch_size=512, shuffle=True,
        num_workers=4, collate_fn=data_handler.batchify)
test_producer = th.utils.data.DataLoader(
        dataset=eval_set, batch_size=128, shuffle=True,
        num_workers=4, collate_fn=data_handler.batchify)
trainer = T.trainer('ckpt/'+name)
trainer(net, train_producer, test_producer, epochs=300, gpu_id=0)
trainer.plot(filename='test_plots/'+name)
