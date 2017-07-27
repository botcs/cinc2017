#! /usr/bin/python3.5
import data_handler
import dilated_model as DM
import trainer as T
import numpy as np
import torch as th
from torch.autograd import Variable
import sys, os
os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[1]
from os.path import basename, splitext

th.multiprocessing.set_sharing_strategy('file_system')
name = splitext(basename(sys.argv[0]))[0]

transformations = [
    data_handler.Crop(2400),
    data_handler.Threshold(sigma=2.2),
    data_handler.RandomMultiplier(-1),
]

dataset = data_handler.DataSet(
    'data/raw/training2017/REFERENCE.csv', data_handler.load_composed,
    transformations=transformations,
    path='data/raw/training2017/',
    remove_noise=True, tokens='NAO')
train_set, eval_set = dataset.disjunct_split(.9)

net = DM.ResNet(K_blocks=4, N_res_in_block=1, 
                channel=8, use_selu=True)

train_producer = th.utils.data.DataLoader(
        dataset=train_set, batch_size=128, shuffle=True,
        num_workers=4, collate_fn=data_handler.batchify)
test_producer = th.utils.data.DataLoader(
        dataset=eval_set, batch_size=64, shuffle=True,
        num_workers=4, collate_fn=data_handler.batchify)
trainer = T.Trainer('saved/'+name)
trainer(net, train_producer, test_producer, gpu_id=0, useAdam=True)
