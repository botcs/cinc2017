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
from Pan_Tompkins_2 import Pan_Tompkins_2
from QRS_detection import QRS_detect
from PT_detection import PT_detect

def Time_domain_features(min_batch):

    feature_vector = []
    
    # load signal
    for y0 in min_batch:

        fs = 300.
        
        # Pan Tompkins algorithm
        [y_LP,y_HP,y_DEV,y_SM2]=Pan_Tompkins_2(y0);
        print [y_LP,y_HP,y_DEV,y_SM2]
        #Q-R-S detection
        [Q_index,Q_value,R_index,R_value,S_index,S_value,NN50,QS5,ind_R,ind_QS]=QRS_detect([y0,y_SM2,fs])

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
