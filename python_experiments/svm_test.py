from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt

'''
x = [1, 5, 1.5, 8, 1, 9]
y = [2, 8, 1.8, 8, 0.6, 11]

plt.scatter(x,y)
plt.show()
'''
<<<<<<< HEAD
#The same data as 2d vector
X = np.array([[1,2],
             [5,8],
             [1.5,1.8],
             [8,8],
             [1,0.6],
             [9,11]])

Z =np.load('feature_vector.npy')

X=[]
X.append(Z[:,0])
X.append(Z[:,2])
X.append(Z[:,3])
X=np.transpose(X)

X_valid = X[-200:]
X = X[:-200]
#labels
y = [0,1,0,1,0,1]
y =np.load('annotation.npy')
y_valid = y[-200:]
y = y[:-200]
#create SVM, linear, and C=1 for balanced data, C should be changed if the number of datapoints are not equal in the different classes
clf = svm.SVC(kernel='linear', C = 1.0)


#train the SVM
clf.fit(X,y)


#create a classificaiton with the trained SVM
=======
# The same data as 2d vector
X = np.array([[1, 2],
              [5, 8],
              [1.5, 1.8],
              [8, 8],
              [1, 0.6],
              [9, 11]])

Z = np.load('feature_vector.npy')

X = []
X.append(Z[:, 0])
X.append(Z[:, 2])
X.append(Z[:, 3])
X = np.transpose(X)

X_valid = X[-200:]
X = X[:-200]
# labels
y = [0, 1, 0, 1, 0, 1]
y = np.load('annotation.npy')
y_valid = y[-200:]
y = y[:-200]
# create SVM, linear, and C=1 for balanced data, C should be changed if
# the number of datapoints are not equal in the different classes
clf = svm.SVC(kernel='linear', C=1.0)


# train the SVM
clf.fit(X, y)


# create a classificaiton with the trained SVM
>>>>>>> 89b6a43314e18ded2ed7ab8f7e2938583d71c218
print(clf.predict(X_valid))
print(np.mean(clf.predict(X_valid) == y_valid))
#print(clf.predict([0.58, 0.76,0.58, 0.76]))

# Visualize SVM data- can be done in 2D
w = clf.coef_[0]
print(w)

a = -w[0] / w[1]

xx = np.linspace(0, 12)
yy = a * xx - clf.intercept_[0] / w[1]

h0 = plt.plot(xx, yy, 'k-', label="separator plane")

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.legend()
plt.show()
