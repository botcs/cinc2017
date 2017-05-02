import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

def PT_detect(input):

	y0=input[0]
	Q_index=input[1]
	R_index=input[2]
	S_index=input[3]
	fs=input[4]

	
	max_QRS_d=0.2*fs;
	max_PQ_d=0.45*fs
	y_PT=np.asarray(y0);
	
	P_index=[];P_value=[];
	T_index=[];T_value=[];
	
	#Variability indexes
	PR20=0
	QT20=0
	
	#Smoothing
	y_SQ_PT=y_PT*y_PT/10.
	b_SM=[1./30]*30;
	a_SM=[1]+[0]*30;
	y_SM1_PT = signal.filtfilt(b_SM, a_SM, y_SQ_PT)
	y_SM2_PT = signal.filtfilt(b_SM, a_SM, y_SM1_PT)
	
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
				ind_P=len(P_index)-1
				if 1<ind_P:
					#print i,PR20,P_index[ind_P],R_index[i],P_index[ind_T-1],R_index[i-1]
					if abs(abs(R_index[i]-P_index[ind_P])-abs(R_index[i-1]-P_index[ind_P-1]))>0.02:
						PR20=PR20+1
		
		if S_ind<index<T_ind:
			if y_SM2_PT[index-1]<value>y_SM2_PT[index+1]:
				T_index.append(index/fs)
				T_value.append(value)
				ind_T=len(T_index)-1
				if 1<ind_T:
					#print i,QT20,T_index[ind_T],Q_index[i],T_index[ind_T-1],Q_index[i-1]
					if abs(abs(T_index[ind_T]-Q_index[i])-abs(T_index[ind_T-1]-Q_index[i-1]))>0.02:
						QT20=QT20+1
	
	return [P_index,P_value,T_index,T_value,PR20,QT20]