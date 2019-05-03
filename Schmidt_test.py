import numpy as np
import os
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn import svm

directory = ['./Data/KO', './Data/OK']
train_data = []
test_data = []
for dir in directory:
    for filename in os.listdir(dir)[:96]:
        if filename.endswith('.out'):
            dir_filname = os.path.join(dir, filename)
            dat = np.loadtxt(dir_filname, delimiter=',')[1]
            train_data.append((dat, directory.index(dir)))

for dir in directory:
    for filename in os.listdir(dir)[96:]:
        if filename.endswith('.out'):
            dir_filname = os.path.join(dir, filename)
            dat = np.loadtxt(dir_filname, delimiter=',')[1]
            test_data.append((dat, directory.index(dir)))


X = []
Y = []
for d in train_data:
    x, y = d
    X.append(x)
    Y.append(y)

X_test = []
Y_test = []
for d in test_data:
    x,y = d
    X_test.append(x)
    Y_test.append(y)



clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100,50,20,10), random_state=1)
clf.fit(X,Y)
prediction = clf.predict(X_test)

print('Neural Network Success = {0} / {1}'.format(len(Y_test) - sum(np.abs(prediction-Y_test)), len(Y_test)))


clf = svm.SVC(gamma='scale')
clf.fit(X, Y)
prediction = clf.predict(X_test)

print('Support Vector Machine Success = {0} / {1}'.format(len(Y_test) - sum(np.abs(prediction-Y_test)), len(Y_test)))

