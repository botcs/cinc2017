# coding: utf-8
import torch as th
from torch.autograd import Variable

import data_handler
import dilated_model as DM
import trainer as T

import numpy as np
from scipy import signal, ndimage
import sys, os
from os.path import basename, splitext

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
matplotlib.rcParams['figure.figsize'] = [20, 10]
font = {'size': 16}
matplotlib.rc('font', **font)

th.multiprocessing.set_sharing_strategy('file_system')
name = splitext(basename(sys.argv[0]))[0]

# Set up NN, load weights and data
# Output: data iterator (call next(dataiter) for next batch)
def _setup(label_path, data_path, weights_path):
    global_transforms = [
        data_handler.Crop(9000),
    ]
    
    # Time domain transformation
    transformationsA = [
        data_handler.Threshold(sigma=2.2),
        data_handler.RandomMultiplier(-1),
    ]
    
    # Frequency domain transformation
    transformationsB = [
        data_handler.RandomMultiplier(-1),
        data_handler.Spectogram(31)
    ]

    # Load data
    dataset = data_handler.DataSet(
        label_path, data_handler.load_forked,
        global_transforms=global_transforms,
        fork_transforms={'time':transformationsA, 'freq':transformationsB},
        path=data_path,
        remove_unlisted=False, tokens=data_handler.atrif_tokens, remove_noise=True)
    train_set, eval_set = dataset.disjunct_split(.9)
    train_producer = th.utils.data.DataLoader(
        dataset=dataset, batch_size=5, shuffle=True,
        num_workers=0, collate_fn=data_handler.batchify_forked)

    # Load network
    timeNet = DM.EncodeWideResNetFIXED(in_channel=1, init_channel=16, 
        num_enc_layer=4, N_res_in_block=1, use_selu=True)

    freqNet = DM.SkipFCN(in_channel=16, use_selu=True,
        channels=[16,16,  32,32,  64,64,64,  128,128,128,  128,128,128])

    classifier = th.nn.Sequential(th.nn.BatchNorm1d(256), DM.SELU(), th.nn.Conv1d(256, 3, 1))
    net = DM.CombinedTransform(
        pretrained=False,
        feature_length=20, 
        time=timeNet, 
        freq=freqNet, 
        classifier=classifier)
    net.load_state_dict(th.load(weights_path))

    return net, train_producer

def generate_coloring(data, raw_logit):
    logits = np.transpose(raw_logit)
    
    # Smoothing with lowpass filter
    b, a = signal.butter(8, 0.1)
    filt_logit = signal.filtfilt(b, a, logits, axis=0, padlen=150)

    # Normalize along y axis
    prenorm_logit = filt_logit - np.min(filt_logit)
    prenorm_logit = prenorm_logit / np.max(prenorm_logit)

    # Selecting dominant class for short sections with keeping strength of guess
    indices = np.argmax(prenorm_logit, axis=1)
    one_hot = np.eye(3)[indices]
    selected_logit = np.where(one_hot == 1, prenorm_logit, 2)

    # Normalization again
    minimum = np.min(selected_logit)
    norm_logit = selected_logit - minimum
    norm_logit = np.where(norm_logit == 2 - minimum, 0, norm_logit)
    norm_logit = norm_logit / np.max(norm_logit)
    
    # Convert logits to RBG
    logit = np.transpose(norm_logit)
    color = [[],[],[]]
    color[0] = 1 - logit[0] - logit[2]
    color[1] = 1 - logit[1] - logit[2]
    color[2] = 1 - logit[0] - logit[1]
    color = np.transpose(color)
    
    return color

def colorplot(data, raw_logit, label, save_path=""):
    # Plot data
    fig = plt.figure()
    ax = fig.add_subplot(111)
    data_length = len(data)
    t = np.linspace(0, data_length, data_length)
    ax.plot(t, data)

    # Draw background
    def convert(float):
        return round(float*255.).astype(int)
    def convertArray(array):
        return ((convert(array[0]), convert(array[1]), convert(array[2])))
    color = generate_coloring(data, raw_logit)
    color_length = len(color)
    step = data_length/color_length
    bottom = min(data)
    height = max(data) - bottom
    x = np.linspace(0, data_length, color_length)
    def draw_line(x_i, c_i):
        hex_color = '#%02x%02x%02x' % convertArray(c_i)
        ax.add_patch(patches.Rectangle((x_i, bottom), step, height, facecolor=hex_color, alpha=0.5))
    for i in range(color_length):
        draw_line(x[i], color[i])

    # Legend
    normal_patch = patches.Patch(color='#00ff00', label='Normal rhythm')
    af_patch = patches.Patch(color='#ff0000', label='AF rhythm')
    other_patch = patches.Patch(color='#0000ff', label='Other rhythm')
    plt.legend(handles=[normal_patch, af_patch, other_patch], prop={'size': 20})

    # Axis labels and title
    plt.ylabel('Amplitude [mV]')
    plt.xlabel('Time [s]')
    class_dict = ['Normal rhythm', 'AF rhythm', 'Other rhythm']
    decision = np.average(raw_logit, axis=1)
    plt.title('Overall guess: Normal: ' + '%.2f' % decision[0]
                         + ', AF: '     + '%.2f' % decision[1]
                         + ', Other: '  + '%.2f' % decision[2] +
             ', Ground truth: ' + label)

    if save_path == "":
        plt.show()
    else:
        fig6.savefig(save_path, dpi=90, bbox_inches='tight')