import matplotlib.pyplot as plt
import numpy as np

def QRS_detect(input):
	
	y0=input[0]
	y_SM2=input[1]
	fs=input[2]
	
	y_PT=np.asarray(y0);
	
	Q_index=[];Q_value=[];
	R_index=[];R_value=[];
	S_index=[];S_value=[];
	
	#Variability indexes
	NN50=0
	QS5=0
	ind_QS=0;

	ind=0;
	max_QRS_d=0.2*fs;
	max_RR_d=0.8*fs;
	
	for index,value in enumerate(y_SM2):
		if index>max_RR_d and index<len(y_SM2)-max_RR_d:
			if y_SM2[index]== max(y_SM2[index-int(max_RR_d/2):index+int(max_RR_d/2)]):
				R_index.append(index/fs)
				R_value.append(y_SM2[index])
				ind_R=len(R_index)-1
				if 2<ind_R:
					if abs(abs(R_index[ind_R]-R_index[ind_R-1])-abs(R_index[ind_R-2]-R_index[ind_R-3]))>0.05:
						NN50=NN50+1
			
			if len(R_index)>1:
				start=int(R_index[len(R_index)-1]*fs-max_QRS_d/2)
				middle=int(R_index[len(R_index)-1]*fs)
				stop= int(R_index[len(R_index)-1]*fs+max_QRS_d/2)
				mod_index=int(index-max_QRS_d)
				
				
				if start<mod_index<middle:
					if y_SM2[mod_index]==min(y_SM2[start:middle]):
						Q_index.append((index-max_QRS_d)/fs)
						Q_value.append(y_SM2[mod_index])
				
				if middle<index<stop:
					if value==min(y_SM2[middle:stop]):
						S_index.append(index/fs)
						S_value.append(y_SM2[index])
						'''ind_QS=len(Q_index)-1
							if 1<ind_QS<len(R_index):
								#print ind_QS,Q_index[ind_QS],S_index[ind_QS]
								if abs(abs(S_index[ind_QS]-Q_index[ind_QS])-abs(S_index[ind_QS-1]-Q_index[ind_QS-1]))>0.005:
									QS5=QS5+1'''
		
				#Delet QRS							
				if 1<len(Q_index)<len(R_index) and 1<len(S_index)<len(R_index) :
					Relative_Q_indexes=list(abs(np.array(Q_index)-R_index[len(R_index)-1]))
					min_Q_value=min(np.array(Relative_Q_indexes))
					Relative_S_indexes=list(abs(np.array(S_index)-R_index[len(R_index)-1]))
					min_S_value=min(np.array(Relative_S_indexes))
					min_Q_index=Relative_Q_indexes.index(min_Q_value)
					min_S_index=Relative_S_indexes.index(min_S_value)
					
					if (Q_index[min_Q_index]<R_index[len(R_index)-1]<S_index[min_S_index]):
						Q_ind=Q_index[min_Q_index]*fs #Q_index[len(Q_index)-1]*fs
						S_ind=S_index[min_S_index]*fs #S_index[len(S_index)-1]*fs
						ind=index-(S_ind-Q_ind)
						if  Q_ind<ind<S_ind:
							y_PT[int(ind)]=((y_PT[int(Q_ind)]/y_PT[int(S_ind)])/(Q_ind/S_ind))*(ind-Q_ind) 
						
						
						Relative_Q_indexes_but1=list(abs(np.array(Q_index)-R_index[len(R_index)-2]))
						min_Q_value_but1=min(np.array(Relative_Q_indexes))
						Relative_S_indexes_but1=list(abs(np.array(S_index)-R_index[len(R_index)-2]))
						min_S_value_but1=min(np.array(Relative_S_indexes))
						min_Q_index_but1=Relative_Q_indexes.index(min_Q_value)
						min_S_index_but1=Relative_S_indexes.index(min_S_value)
						if (Q_index[min_Q_index_but1]<R_index[len(R_index)-2]<S_index[min_S_index_but1]):
							ind_QS=ind_QS+1
							if abs(abs(S_index[min_S_index]-Q_index[min_Q_index])-abs(S_index[min_S_index_but1]-Q_index[min_Q_index_but1]))>0.005:
								QS5=QS5+1

	
	return 0#[Q_index,Q_value,R_index,R_value,S_index,S_value,NN50,QS5,ind_R,ind_QS]
