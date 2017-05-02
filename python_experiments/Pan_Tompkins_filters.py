import matplotlib.pyplot as plt
import numpy as np

<<<<<<< HEAD
N=[19, 50, 7, 30, 30]

b_LPF=[1./72]+ [0]*8 + [-1./36] +[0]*8 +[1./72];
a_LPF=[1.,-1.99, 1.] + [0]*16;

b_HPF=[-1./49]+[0]*24+[1,-1]+[0]*22+[1./49];
a_HPF=[1.,-0.99]+[0]*48;

b_DEV=[1./3.6, 0, 1./8, 0, -1./8,0, -1./3.6];
a_DEV=[1]+[0]*6;

b_SM=[1./30]*30;
a_SM=[1]+[0]*30;
=======
N = [26, 50, 7, 30, 30]

b_LPF = [1. / 72] + [0] * 8 + [-1. / 36] + [0] * 8 + [1. / 72]
a_LPF = [1., -1.99, 1.] + [0] * 23

b_HPF = [-1. / 49] + [0] * 24 + [1, -1] + [0] * 22 + [1. / 49]
a_HPF = [1., -0.99] + [0] * 48

b_DEV = [1. / 3.6, 0, 1. / 8, 0, -1. / 8, 0, -1. / 3.6]
a_DEV = [1] + [0] * 6

b_SM = [1. / 30] * 30
a_SM = [1] + [0] * 30
>>>>>>> 89b6a43314e18ded2ed7ab8f7e2938583d71c218
'''

N=[13, 33, 5, 30, 30]

b_LPF=[1./32]+ [0]*5 + [-1./16] +[0]*5 +[1./32];
a_LPF=[1.,-1.99, 1.] + [0]*10;

b_HPF=[-1./32]+[0]*15+[1,-1]+[0]*14+[1./32];
a_HPF=[1.,-0.99]+[0]*31;

b_DEV=[1./4, 1./8, 0, -1./8,-1./4];
a_DEV=[1]+[0]*4;

b_SM=[1./30]*30;
a_SM=[1]+[0]*30;
'''
<<<<<<< HEAD
DenomCoeff=[b_LPF, b_HPF, b_DEV, b_SM, b_SM]
NumCoeff=[a_LPF,a_HPF,a_DEV,a_SM, a_SM];

def Pan_Tompkins(pSrc):
	signal=np.array(pSrc)
	BLOCK_SIZE=len(pSrc)

	#print "Hello Pan_Tompkins!",len(pSrc), len(signal)

	pDst=np.array([None]*BLOCK_SIZE)
	s=len(N)
	pDst_signals=[]
	
	for i in range(s):
		Reg=[0]*N[i]
		for j in range(BLOCK_SIZE):
			Reg[1:]=Reg[:-1]
	
			# The denominator
			Reg[0] = signal[j];
			for k in range(N[i]):
				Reg[0] = Reg[0]-DenomCoeff[i][k] * Reg[k];

			# The numerator
			y = 0;
			for k in range(N[i]):
				y =y+NumCoeff[i][k] * Reg[k];
				
			pDst[j]=y
		
		if i==2:
			pDst=pDst*pDst
			pDst_signals.append(pDst)
		if i>1:
			signal=pDst
		
		pDst_signals.append(pDst)
		
	
	fs=300.
	
	'''
	time=np.linspace(0,len(pSrc)/fs,len(pSrc))
	f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)
	p=2
	original_signal=np.array(pSrc)/np.mean(pSrc)*np.mean(pDst_signals[p])
	ax1.plot(time,original_signal)
	ax1.plot(time,pDst_signals[p-1])
	ax2.plot(time,pDst_signals[p])
	plt.show()
	'''
	
	return pDst_signals
	
=======
DenomCoeff = [b_LPF, b_HPF, b_DEV, b_SM, b_SM]
NumCoeff = [a_LPF, a_HPF, a_DEV, a_SM, a_SM]


def Pan_Tompkins(pSrc):
    signal = tuple(pSrc)
    BLOCK_SIZE = len(pSrc)
    pDst = np.array([None] * BLOCK_SIZE)
    s = len(N)
    for i in range(s):
        Reg = [0] * N[i]
        for j in range(BLOCK_SIZE):

            Reg[1:] = Reg[:-1]

            # The denominator
            Reg[0] = pSrc[j]
            for k in range(N[i]):
                Reg[0] = Reg[0] - DenomCoeff[i][k] * Reg[k]

            # The numerator
            y = 0
            for k in range(N[i]):
                y = y + NumCoeff[i][k] * Reg[k]

            pDst[j] = y

        if i == 2:
            pDst = pDst * pDst
        if i > 1:
            pSrc = pDst

    fs = 300.
    time = np.linspace(0, len(pDst) / fs, len(pDst))
    print "Hello"
    #f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)
    # plt.plot(time,pDst)
    # plt.show()
    return pDst
>>>>>>> 89b6a43314e18ded2ed7ab8f7e2938583d71c218
