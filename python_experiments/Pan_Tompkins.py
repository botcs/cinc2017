import matplotlib.pyplot as plt
import numpy as np
import wave
import sys
import scipy
import scipy.io
from numpy import array, ones
from scipy import signal
from scipy.signal import lfilter, lfilter_zi, butter

## import signal
sig=scipy.io.loadmat('REC_NAME.mat')
sig= np.asarray(sig['val'])
y0=sig[0]
fs=300.;

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


#PT Squaring
y_QRS=(y_HP*y_HP*y_HP)/200000.
y_PT=np.asarray(y0);

P_index=[];P_value=[];y0_P=[]
Q_index=[];Q_value=[];y0_Q=[]
R_index=[];R_value=[];y0_R=[]
S_index=[];S_value=[];y0_S=[]
T_index=[];T_value=[];y0_T=[]

ind=0;
max_QRS_d=0.2*fs;
max_RR_d=0.8*fs;
max_PQ_d=0.45*fs
#Q-R-S detection
for index,value in enumerate(y_QRS):
	if index>max_RR_d and index<len(y_QRS)-max_RR_d:
		if y_SM2[index]== max(y_SM2[index-int(max_RR_d/2):index+int(max_RR_d/2)]):
			R_index.append(index/fs)
			R_value.append(y_SM2[index])
			y0_R.append(y0[index])
		
		if len(R_index)>1:
			start=int(R_index[len(R_index)-1]*fs-max_QRS_d/2)
			middle=int(R_index[len(R_index)-1]*fs)
			stop= int(R_index[len(R_index)-1]*fs+max_QRS_d/2)
			mod_index=int(index-max_QRS_d)
			
			
			if start<mod_index<middle:
				if y_QRS[mod_index]==min(y_QRS[start:middle]):
					Q_index.append((index-max_QRS_d)/fs)
					Q_value.append(y_QRS[mod_index])
					y0_Q.append(y0[index])
			
			if middle<index<stop:
				if value==min(y_QRS[middle:stop]):
					S_index.append(index/fs)
					S_value.append(y_QRS[index])
					y0_S.append(y0[index])
		
			if len(Q_index)>1:
				Q_ind=Q_index[len(Q_index)-1]*fs
				S_ind=S_index[len(S_index)-1]*fs
				ind=index-(S_ind-Q_ind)
				if  Q_ind<ind<S_ind:
					y_PT[int(ind)]=((y_PT[int(Q_ind)]/y_PT[int(S_ind)])/(Q_ind/S_ind))*(ind-Q_ind) #*(S_ind-index)+y0[S_ind]*(index-Q_ind))
					

#Smoothing
y_SQ_PT=y_PT*y_PT/10.
b_SM=[1./30]*30;
a_SM=[1]+[0]*30;
y_SM1_PT = signal.filtfilt(b_SM, a_SM, y_SQ_PT)
y_SM2_PT = signal.filtfilt(b_SM, a_SM, y_SM1_PT)

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
	
	if S_ind<index<T_ind:
		if y_SM2_PT[index-1]<value>y_SM2_PT[index+1]:
			T_index.append(index/fs)
			T_value.append(value)
			y0_T.append(y0[index])
	
#Plot
time=np.linspace(0,len(y0)/fs,len(y0))

f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True)
ax1.plot(time,y0)
ax1.plot(P_index,y0_P,'r*')
ax1.plot(Q_index,y0_Q,'g*')
ax1.plot(R_index,y0_R,'ko')
ax1.plot(S_index,y0_S,'c*')
ax1.plot(T_index,y0_T,'m*')

ax2.plot(time,y_SM2_PT)
ax2.plot(P_index,P_value,'r*')
ax2.plot(T_index,T_value,'g*')

#ax2.plot(R_index,R_value,'r*')

ax3.plot(time,y_QRS)
ax3.plot(Q_index,Q_value,'g*')
ax3.plot(S_index,S_value,'k*')

plt.show()
