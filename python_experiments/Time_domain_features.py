def Time_domain_features(min_batch):

	import matplotlib.pyplot as plt
	import numpy as np
	import wave
	import sys
	import scipy
	import scipy.io
	from numpy import array, ones
	from scipy import signal
	from scipy.signal import lfilter, lfilter_zi, butter
	import csv
	from Pan_Tompkins_filters import Pan_Tompkins
	from QRS_detection import QRS_detect
	from PT_detection import PT_detect
	
	feature_vector=[]

	for y0 in min_batch:
		## import signal
		fs=300.
		## PT algorithm
		print "Hello Time_domain_features",len(y0)
		y0=np.array(y0)
		[y_LP,y_HP,y_DEV,y_SQ,y_SM1,y_SM2]=Pan_Tompkins(y0)

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

		#Q-R-S detection
		[Q_index,Q_value,R_index,R_value,S_index,S_value,NN50,QS5,ind_R,ind_QS]=QRS_detect([y0,y_HP,y_SM2,fs])

		#P-T detection
		#[P_index,P_value,T_index,T_value,PR20,QT20]=PT_detect([y0,Q_index,R_index,S_index,fs])		

		# Features
		pNN50=round(float(NN50)/(ind_R+1),3)
		pQS5=round(float(QS5)/(ind_QS+1),3)	
		'''pQT20=round(float(QT20)/(ind_T+1),3)	
		pPR20=round(float(PR20)/(ind_P+1),3)
		pPR=round(float(len(P_index))/(len(R_index)-1),3)'''
		pQR=round(float(len(Q_index))/(len(R_index)-1),3)
		pSR=round(float(len(S_index))/(len(R_index)-1),3)
		'''pTR=round(float(len(T_index))/(len(R_index)-1),3)'''
		
		BPM=len(R_index)*60/(len(y0)/fs)
		
		#features=[pNN50,pQS5,pQT20,pPR20,pPR,pQR,pSR,pTR]
		features=[pNN50,pQR,pSR,BPM]
		feature_vector.append(features)
			
	return feature_vector

