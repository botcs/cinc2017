#!/usr/bin/env ipython
# coding: utf-8
from __future__ import print_function
import numpy as np
import tensorflow as tf
import scipy.io as io
import scipy.signal as signal
import random
import matplotlib.pyplot as plt


def pan_tompkins(y0):
  fs = 300.

  # PT algorithm

  # LP filter
  b_LPF = [1. / 32] + [0] * 5 + [-1. / 16] + [0] * 5 + [1. / 32]
  a_LPF = [1., -1.99, 1.] + [0] * 10
  y_LP = signal.filtfilt(b_LPF, a_LPF, y0)

  # HP Filter
  b_HPF = [-1. / 32] + [0] * 15 + [1, -1] + [0] * 14 + [1. / 32]
  a_HPF = [1., -0.99] + [0] * 31
  y_HP = signal.filtfilt(b_HPF, a_HPF, y0)

  # Differentiation
  b_DEV = [1. / 4, 1. / 8, 0, -1. / 8, -1. / 4]
  a_DEV = [1] + [0] * 4
  y_DEV = signal.filtfilt(b_DEV, a_DEV, y_HP)

  # Squaring
  y_SQ = y_DEV * y_DEV

  # Smoothing
  b_SM = [1. / 30] * 30
  a_SM = [1] + [0] * 30
  y_SM1 = signal.filtfilt(b_SM, a_SM, y_SQ)
  y_SM2 = signal.filtfilt(b_SM, a_SM, y_SM1)

  # PT Squaring
  y_QRS = (y_HP * y_HP * y_HP) / 200000.
  y_PT = np.asarray(y0)

  # P-Q-R-S-T indexes and values
  P_index = []
  P_value = []
  y0_P = []
  Q_index = []
  Q_value = []
  y0_Q = []
  R_index = []
  R_value = []
  y0_R = []
  S_index = []
  S_value = []
  y0_S = []
  T_index = []
  T_value = []
  y0_T = []

  # Variability indexes
  pNN50 = []
  NN50 = 0
  pPR20 = []
  PR20 = 0
  pQT20 = []
  QT20 = 0
  pQS5 = []
  QS5 = 0
  ind_QS = 0

  ind = 0
  max_QRS_d = 0.2 * fs
  max_RR_d = 0.8 * fs
  max_PQ_d = 0.45 * fs

  # Q-R-S detection
  for index, value in enumerate(y_QRS):
    if index > max_RR_d and index < len(y_QRS) - max_RR_d:
      if y_SM2[index] == max(
          y_SM2[index - int(max_RR_d / 2):index + int(max_RR_d / 2)]):
        R_index.append(index / fs)
        R_value.append(y_SM2[index])
        y0_R.append(y0[index])
        ind_R = len(R_index) - 1
        if 2 < ind_R:
          if abs(abs(R_index[ind_R] - R_index[ind_R - 1]) -
               abs(R_index[ind_R - 2] - R_index[ind_R - 3])) > 0.05:
            NN50 = NN50 + 1

      if len(R_index) > 1:
        start = int(R_index[len(R_index) - 1] * fs - max_QRS_d / 2)
        middle = int(R_index[len(R_index) - 1] * fs)
        stop = int(R_index[len(R_index) - 1] * fs + max_QRS_d / 2)
        mod_index = int(index - max_QRS_d)

        if start < mod_index < middle:
          if y_QRS[mod_index] == min(y_QRS[start:middle]):
            Q_index.append((index - max_QRS_d) / fs)
            Q_value.append(y_QRS[mod_index])
            y0_Q.append(y0[index])

        if middle < index < stop:
          if value == min(y_QRS[middle:stop]):
            S_index.append(index / fs)
            S_value.append(y_QRS[index])
            y0_S.append(y0[index])
            '''ind_QS=len(Q_index)-1
							if 1<ind_QS<len(R_index):
								#print ind_QS,Q_index[ind_QS],S_index[ind_QS]
								if abs(abs(S_index[ind_QS]-Q_index[ind_QS])-abs(S_index[ind_QS-1]-Q_index[ind_QS-1]))>0.005:
									QS5=QS5+1'''

        # Delet QRS
        if 1 < len(Q_index) < len(R_index) and 1 < len(
            S_index) < len(R_index):
          Relative_Q_indexes = list(
            abs(np.array(Q_index) - R_index[len(R_index) - 1]))
          min_Q_value = min(np.array(Relative_Q_indexes))
          Relative_S_indexes = list(
            abs(np.array(S_index) - R_index[len(R_index) - 1]))
          min_S_value = min(np.array(Relative_S_indexes))
          min_Q_index = Relative_Q_indexes.index(min_Q_value)
          min_S_index = Relative_S_indexes.index(min_S_value)

          if (Q_index[min_Q_index] < R_index[len(
              R_index) - 1] < S_index[min_S_index]):
            # Q_index[len(Q_index)-1]*fs
            Q_ind = Q_index[min_Q_index] * fs
            # S_index[len(S_index)-1]*fs
            S_ind = S_index[min_S_index] * fs
            ind = index - (S_ind - Q_ind)
            if Q_ind < ind < S_ind:
              y_PT[int(ind)] = (
                (y_PT[int(Q_ind)] / y_PT[int(S_ind)]) / (Q_ind / S_ind)) * (ind - Q_ind)

            Relative_Q_indexes_but1 = list(
              abs(np.array(Q_index) - R_index[len(R_index) - 2]))
            min_Q_value_but1 = min(np.array(Relative_Q_indexes))
            Relative_S_indexes_but1 = list(
              abs(np.array(S_index) - R_index[len(R_index) - 2]))
            min_S_value_but1 = min(np.array(Relative_S_indexes))
            min_Q_index_but1 = Relative_Q_indexes.index(
              min_Q_value)
            min_S_index_but1 = Relative_S_indexes.index(
              min_S_value)
            if (Q_index[min_Q_index_but1] < R_index[len(
                R_index) - 2] < S_index[min_S_index_but1]):
              ind_QS = ind_QS + 1
              if abs(abs(S_index[min_S_index] -
                     Q_index[min_Q_index]) -
                   abs(S_index[min_S_index_but1] -
                     Q_index[min_Q_index_but1])) > 0.005:
                QS5 = QS5 + 1

  # Smoothing
  y_SQ_PT = y_PT * y_PT / 10.
  b_SM = [1. / 30] * 30
  a_SM = [1] + [0] * 30
  y_SM1_PT = signal.filtfilt(b_SM, a_SM, y_SQ_PT)
  y_SM2_PT = signal.filtfilt(b_SM, a_SM, y_SM1_PT)
  '''
		# P-T detection
		i=0
		for index,value in enumerate(y_SM2_PT):
			P_ind =int(Q_index[i]*fs-max_QRS_d)
			Q_ind =int(Q_index[i]*fs)
			S_ind =int(S_index[i]*fs)
			T_ind =int(S_index[i]*fs+max_QRS_d)
			if index>T_ind and i<len(Q_index)-1:
				i=i+1

			if P_ind<index<Q_ind:
				if y_SM2_PT[index-1]<value>y_SM2_PT[index+1]:
					P_index.append(index/fs)
					P_value.append(value)
					y0_P.append(y0[index])
					ind_P=len(P_index)-1
					if 1<ind_P:
						#print i,PR20,P_index[ind_P],R_index[i],P_index[ind_T-1],R_index[i-1]
						if abs(abs(R_index[i]-P_index[ind_P])-abs(R_index[i-1]-P_index[ind_P-1]))>0.02:
							PR20=PR20+1

			if S_ind<index<T_ind:
				if y_SM2_PT[index-1]<value>y_SM2_PT[index+1]:
					T_index.append(index/fs)
					T_value.append(value)
					y0_T.append(y0[index])
					ind_T=len(T_index)-1
					if 1<ind_T:
						#print i,QT20,T_index[ind_T],Q_index[i],T_index[ind_T-1],Q_index[i-1]
						if abs(abs(T_index[ind_T]-Q_index[i])-abs(T_index[ind_T-1]-Q_index[i-1]))>0.02:
							QT20=QT20+1'''

  # Features
  pNN50 = round(float(NN50) / (ind_R + 1), 3)
  pQS5 = round(float(QS5) / (ind_QS + 1), 3)
  '''pQT20=round(float(QT20)/(ind_T+1),3)
		pPR20=round(float(PR20)/(ind_P+1),3)
		pPR=round(float(len(P_index))/(len(R_index)-1),3)'''
  pQR = round(float(len(Q_index)) / (len(R_index) - 1), 3)
  pSR = round(float(len(S_index)) / (len(R_index) - 1), 3)
  return [pNN50, pQS5, pQR, pSR]


def estimated_autocorrelation(x):
  n = len(x)
  variance = x.var()
  x = x - x.mean()
  #r = np.correlate(x, x, mode = 'full')[((n/2)-600):((n/2)+600)]
  r = np.correlate(x, x, mode='full')
  #r = r/np.max(r)
  return r


def power_spectrum_density(data):
  ps = np.abs(np.fft.fft(data))**2
  return ps[0:1000]


label_dict = {
  'N': 0,
  'A': 1,
  'O': 2,
  '~': 3
}
# Normal 5154 - use them as they are
# AF	771 -multiply by 7: 5397
# Other rhythm 2557  - multiply by two: 5114
# Noisy	46 - miltiply by 112: 5152

data = []
label = []
#lens = []

annotationdir = '../../training2017/'
annotations = open(annotationdir + 'REFERENCE.csv', 'r').read().splitlines()
NumRecords = len(annotations)

for i, line in enumerate(annotations):
  fname, label_str = line.split(',')
  x = io.loadmat(
    annotationdir +
    fname +
    '.mat')['val'].astype(
    np.float32).squeeze()
  # No Normalization
  #x -= x.mean()
  #x /= x.std()

  # intervall normalization
  #x /= (max(x)-min(x))
  #x -= min(x)

  y = label_dict[label_str]
  label.append(y)
  data.append(x)
data = np.array(data)
label = np.array(label)
noisy_data = data[label == 3]
other_data = data[label == 2]
fibrillation_data = data[label == 1]
normal_data = np.array(data[label == 0])


# variance,interval, changes not working
# autocorrelation
# spectrum
"""
NotGoodNum=0
avg_noisy_power=np.zeros(1000)
for a in range(noisy_data.shape[0]):
  avg_noisy_power+=power_spectrum_density(noisy_data[a])
avg_noisy_power/=noisy_data.shape[0]

avg_normal_power=np.zeros(1000)
for a in range(normal_data.shape[0]):
  avg_normal_power+=power_spectrum_density(normal_data[a])
avg_normal_power/=normal_data.shape[0]
plt.plot(avg_noisy_power,'b')
plt.plot(avg_normal_power,'r')
plt.show()

#display 20 examples
for a in range(20):
  plt.plot(power_spectrum_density(noisy_data[a]),'b')
  plt.plot(power_spectrum_density(normal_data[a]),'r')
  plt.show()
noisy_var=np.zeros(noisy_data.shape[0])

for a in range(20):
  print(pan_tompkins(noisy_data[a]))
print("Normal data")
for a in range(20):
  print(pan_tompkins(fibrillation_data[a]))
plt.plot(noisy_var,'b')
plt.plot(fibrillation_var,'r')
plt.show()
"""
