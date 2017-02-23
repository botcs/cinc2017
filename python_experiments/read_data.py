# coding: utf-8
import pickle
import os

# Találtam egy python könyvtárt, ami R peak-et keres (https://github.com/tru-hy/rpeakdetect)
import rpeakdetect as rpd
# Másik opció: mne (https://github.com/mne-tools/mne-python/blob/master/mne/preprocessing/ecg.py)
# Harmadik opció: bioSPPy https://pypi.python.org/pypi/biosppy/0.2.0

# Példák különböző mintákra
# A00376.mat >> N
# A08500.mat >> A
# A00384.mat >> O
# A01006.mat >> ~

# Beolvassa a .mat fájlt, megkeresi benne az R peak-eket, feldarabolja az R peak-ek mentén
# Bemenet:
#	recordName : fájlnév
# Kimenet:
# 	list of numpy.ndarray-s (adatsor feldarabolva szívverésenként)
def read_data(recordName):
	import scipy.io
	import numpy as np
	mat = scipy.io.loadmat(recordName)['val'][0]
	peaks = rpd.detect_beats(mat, 300)
	data = np.split(mat, peaks)
	return data

# Beolvassa az egyes felvételeket tartalmazó fájlok osztályát, 
# eltárolja egy dict-ben, aminek a kulcsai a fájlnevek és értékei az osztályok, és ezzel a dict-tel visszatér
def read_class():
	import csv
	reader = csv.reader(open("REFERENCE.csv"))
	classNames = {}
	for row in reader:
		classNames[ row[0] ] = row[1]
	return classNames

## Beolvasom az összes adatot és azt, hogy melyik osztályba tartoznak, majd az egészet elmentem egy fájlba pickle segítségével.
# Adat szerkezete: egy dict, aminek a kulcsai az osztályok betűjelei (N, A, O, ~), 
#									az értékei pedig listák, amik a feldarabolt mintákat tartalmazó listákat tartalmaznak
import glob, os

os.chdir("training2017/")
files = glob.glob("*.mat")

classNames = read_class()
data = {}
data["N"] = data["A"] = data["O"] = data["~"] = []

for i, file in enumerate(files):
	data[ classNames[ os.path.splitext(file)[0] ] ].append( read_data( file ) )
	print(i,len(files))

os.chdir("../")
pickle.dump(data, open("data.pkl", "wb") )


