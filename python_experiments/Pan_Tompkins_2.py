from scipy import signal
from scipy.signal import lfilter, lfilter_zi, butter

b_LPF = [1. / 72] + [0] * 8 + [-1. / 36] + [0] * 8 + [1. / 72]
a_LPF = [1., -1.99, 1.] + [0] * 23

b_HPF = [-1. / 49] + [0] * 24 + [1, -1] + [0] * 22 + [1. / 49]
a_HPF = [1., -0.99] + [0] * 48

b_DEV = [1. / 3.6, 0, 1. / 8, 0, -1. / 8, 0, -1. / 3.6]
a_DEV = [1] + [0] * 6

b_SM = [1. / 30] * 30
a_SM = [1] + [0] * 30


def Pan_Tompkins_2 (input):
	
	# LP filter
	y_LP = signal.filtfilt(b_LPF, a_LPF, input)

	# HP Filter
	y_HP = signal.filtfilt(b_HPF, a_HPF, input)

	# Differentiation
	y_DEV = signal.filtfilt(b_DEV, a_DEV, y_HP)

	# Squaring
	y_SQ = y_DEV * y_DEV

	# Smoothing
	y_SM1 = signal.filtfilt(b_SM, a_SM, y_SQ)
	y_SM2 = signal.filtfilt(b_SM, a_SM, y_SM1)
	
	return [y_LP,y_HP,y_DEV,y_SM2]