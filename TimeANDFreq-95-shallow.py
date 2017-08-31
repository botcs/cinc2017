#! /usr/bin/python3.5
import data_handler
import dilated_model as DM
import forked_trainer as T
import numpy as np
import torch as th
from torch.autograd import Variable
import sys, os
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
from os.path import basename, splitext

DRYRUN = '--dryrun' in sys.argv
RESTORE = '--restore' in sys.argv
print('DRYRUN:', DRYRUN, '\tRESTORE:', RESTORE)

th.multiprocessing.set_sharing_strategy('file_system')
name = splitext(basename(sys.argv[0]))[0]

global_transforms = [
    data_handler.Crop(6000),
]

transTime = [
    data_handler.Threshold(sigma=2.2),
    data_handler.RandomMultiplier(-1),
]

transFreq = [
    data_handler.RandomMultiplier(-1),
    data_handler.Spectogram(31),
    #data_handler.Logarithm()
]

dataset = data_handler.DataSet(
    'data/raw/training2017/REFERENCE.csv', data_handler.load_forked,
    global_transforms=global_transforms,
    fork_transforms={'time':transTime, 'freq':transFreq},
    path='data/raw/training2017/',
    remove_unlisted=False, tokens='NAO', remove_noise=True)
train_set, eval_set = dataset.disjunct_split(.95)

train_producer = th.utils.data.DataLoader(
    dataset=dataset, batch_size=16, shuffle=True,
    num_workers=8, collate_fn=data_handler.batchify_forked)

test_producer = th.utils.data.DataLoader(
        dataset=eval_set, batch_size=16, shuffle=True,
        num_workers=8, collate_fn=data_handler.batchify_forked)

timeNet = DM.EncodeWideResNetFIXED(in_channel=1, init_channel=16, 
    num_enc_layer=4, N_res_in_block=1, use_selu=True)
#timeNet.load_state_dict(th.load('time-params.pt'))

freqNet = DM.SkipFCN(in_channel=16, use_selu=True,
    channels=[16,16,  32,32,  64,64,64,  128,128,128,  128,128,128])
#freqNet.load_state_dict(th.load('freq-params.pt'))

classifier = th.nn.Sequential(th.nn.BatchNorm1d(256), DM.SELU(), th.nn.Conv1d(256, 3, 1))
#classifier = th.nn.Conv1d(256, 3, 1)
net = DM.CombinedTransform(
#    pretrained=True,
    pretrained=False,
	feature_length=20, 
	time=timeNet, 
	freq=freqNet, 
	classifier=classifier)

trainer = T.Trainer('saved/'+name, class_weight=[1, 1, 1], restore=RESTORE, dryrun=DRYRUN)
trainer(net, train_producer, test_producer, gpu_id=0, useAdam=True)
