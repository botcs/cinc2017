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

th.multiprocessing.set_sharing_strategy('file_system')

dataset = data_handler.DataSet(
    'data/marci_features/REFERENCE.csv', data_handler.load_crop_thresholded,
    random_invert=True,
    crop_len=1800,
    sigma=2.2,
    path='data/marci_features/',
    remove_noise=True, tokens='NAO')
train_set, eval_set = dataset.disjunct_split(.8)
dilfcn = DM.FCN(1, channels=[128, 128, 256, 256, 512, 512], dilations=[1, 2, 4, 1, 2, 4])

train_producer = th.utils.data.DataLoader(
        dataset=train_set, batch_size=256, shuffle=True,
        num_workers=4, collate_fn=data_handler.batchify)
test_producer = th.utils.data.DataLoader(
        dataset=eval_set, batch_size=64, shuffle=True,
        num_workers=4, collate_fn=data_handler.batchify)
trainer = T.trainer('ckpt/dilfcn2x128-2x256-2x512-124466')
trainer(dilfcn, train_producer, test_producer, epochs=200, gpu_id=0)
