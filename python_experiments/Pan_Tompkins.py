import matplotlib.pyplot as plt
import numpy as np
import wave
import sys
import scipy
import scipy.io
from numpy import array, ones
from scipy.signal import lfilter, lfilter_zi, butter
from peakutils.plot import plot as pplot

## import signal

signal2=scipy.io.loadmat('RECORD_NAME.mat')
y0=signal2[0]

## PT algorithm

# LP filter
b_LPF=[1./32]+ [0]*5 + [-1./16] +[0]*5 +[1./32]
a_LPF=[1.,-1.99, 1.] + [0]*10
y_LP = signal.filtfilt(b_LPF, a_LPF, y0)

#HP Filter 
b_HPF=[-1./32]+[0]*15+[1,-1]+[0]*14+[1./32]
a_HPF=[1.,-0.99]+[0]*31;
y_HP = signal.filtfilt(b_HPF, a_HPF, y0)

#Differentiation 
b_DEV=[1./4, 1./8, 0, -1./8,-1./4]
a_DEV=[1]+[0]*4
y_DEV = signal.filtfilt(b_DEV, a_DEV, y_HP)


#Squaring
y_SQ=y_DEV*y_DEV

#Smoothing
b_SM=[1./30]*30;
a_SM=[1]+[0]*30;
y_SM1 = signal.filtfilt(b_SM, a_SM, y_SQ)
y_SM2 = signal.filtfilt(b_SM, a_SM, y_SM1)

window_size=5;
zip_window=[]
window=np.linspace(1,window_size,window_size)
for i in range(len(y_SM2)-window_size+1):
	zip_window.append(y_SM2[i:i+window_size])
zip(zip_window)	
peak_index=[]
for index,window in enumerate(zip_window):
	if window[(window_size)/2] == max(window) :
		peak=index+(window_size+1)/2
		print peak, max(window)
		peak_index.append(peak)
		
print peak_index
	
#Plot
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)
time=np.linspace(0,max(y0)/300.,len(y0))
ax1.plot(time,y0)
ax1.pplot(time,y0,peak_index)

ax2.plot(time,y_SM2)
plt.show()
