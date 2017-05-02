#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import pylab as plot
from sklearn import svm

label_dict = {
    'N': 0,
    'A': 1,
    'O': 2,
    '~': 3
}


# parameters
spectrum_length = 200
num_features = 20
num_train_elements = 1000

# load data
data = np.load('Data.npy')
label = np.load('Label.npy')
fourier_data = np.load('fourier_' + str(spectrum_length) + '.npy')
print "data loaded"


num_test_elements = data.shape[0] - num_train_elements

normal_indices = np.where(label == label_dict['N'])[0]
num_normal = len(normal_indices)

fibrillation_indices = np.where(label == label_dict['A'])[0]
num_fibrillation = len(fibrillation_indices)

other_indices = np.where(label == label_dict['O'])[0]
num_other = len(other_indices)


noisy_indices = np.where(label == label_dict['~'])[0]
num_noisy = len(noisy_indices)

print('Number of normal samples: ' + str(num_normal))
print('Number of fibrillation samples: ' + str(num_fibrillation))
print('Number of other samples: ' + str(num_other))
print('Number of noisy samples: ' + str(num_noisy))


# calculate normal spectrum
normal_spectrum = np.zeros((num_normal, spectrum_length))
normal_spectrum = fourier_data[normal_indices]


# calculate af spectrum
fibrillation_spectrum = np.zeros((num_fibrillation, spectrum_length))
fibrillation_spectrum = fourier_data[fibrillation_indices]


# calculte average spectrum
normal_avg_spectrum = np.mean(normal_spectrum, 0)
fibrillationl_avg_spectrum = np.mean(fibrillation_spectrum, 0)

# select features - those frequencies where the average difference is the
# largest
sepctrum_dif = normal_avg_spectrum - fibrillationl_avg_spectrum
feature_indices = sepctrum_dif.argsort()[-num_features:]

# create training data
random_indices = np.random.choice(data.shape[0], data.shape[0], False)
train_indices = random_indices[:num_train_elements]
test_indices = random_indices[num_train_elements:]
train_data = np.zeros((num_train_elements, num_features))
for ind in range(num_train_elements):
    train_data[ind, :] = fourier_data[train_indices[ind], feature_indices]
train_labels = label[train_indices]

print "train data created"
test_data = np.zeros((num_test_elements, num_features))
for ind in range(num_test_elements):
    fourier = np.fft.fft(data[test_indices[ind]])
    test_data[ind, :] = fourier_data[test_indices[ind], feature_indices]
test_labels = label[test_indices]

print "test data created"

# C=1 we assume the data is linearly separable
# create SVM, linear, and C=1 for balanced data, C should be changed if
# the number of datapoints are not equal in the different classes
clf = svm.SVC(kernel='linear', class_weight='balanced')

# Create train labels

print "Strating SVM training"
# train the SVM
clf.fit(train_data, train_labels)
print "SVM training finished"

result = clf.predict(test_data)


good_predictions = (test_labels == result)

print "Total accuracy: " + str(float(np.sum(good_predictions)) / float(num_test_elements))
ind = np.where(test_labels == label_dict['N'])[0]
print "Normal samples: " + str(len(good_predictions[ind]))
print "Normal accuracy: " + str(float(np.sum(good_predictions[ind])) / float(len(good_predictions[ind])))

ind = np.where(test_labels == label_dict['A'])[0]
print "Fibrillation samples: " + str(len(good_predictions[ind]))
print "Fibrillation accuracy: " + str(float(np.sum(good_predictions[ind])) / float(len(good_predictions[ind])))

ind = np.where(test_labels == label_dict['O'])[0]
print "Other samples: " + str(len(good_predictions[ind]))
print "Other accuracy: " + str(float(np.sum(good_predictions[ind])) / float(len(good_predictions[ind])))

ind = np.where(test_labels == label_dict['~'])[0]
print "Noisy samples: " + str(len(good_predictions[ind]))
print "Noise accuracy: " + str(float(np.sum(good_predictions[ind])) / float(len(good_predictions[ind])))
