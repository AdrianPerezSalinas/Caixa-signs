import numpy as np
import os
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn import svm
import random

met=5
print(met)
directory = ['./Firmas2/KO/CSV_0.2_'+str(met), './Firmas2/OK/CSV_0.2_'+str(met)]
KO_data = []
OK_data = []
dir = directory[0]
for filename in os.listdir(dir):
    if filename.endswith('.csv'):
        dir_filname = os.path.join(dir, filename)
        dat = pd.read_csv(dir_filname, delimiter=',')
        dat=dat.values
        dat = dat[:,1:].astype('float')
        dat=dat.flatten()
        dat = np.delete(dat, 4)
        KO_data.append((dat, directory.index(dir)))


dir = directory[1]
for filename in os.listdir(dir):
    if filename.endswith('.csv'):
        dir_filname = os.path.join(dir, filename)
        dat = pd.read_csv(dir_filname, delimiter=',')
        dat=dat.values
        dat = dat[:,1:].astype('float')
        dat=dat.flatten()
        dat = np.delete(dat, 4)
        OK_data.append((dat, directory.index(dir)))

print('ok')
lim = 200
steps=100
NN=np.zeros(steps)
SVM=np.zeros(steps)
for i in range(steps):
    random.shuffle(OK_data)
    random.shuffle(KO_data)
    train_data = OK_data[:lim] + KO_data[:lim]
    test_data = OK_data[lim:] + KO_data[lim:]
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

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(30,20,10,5), random_state=1)
    clf.fit(X,Y)
    prediction = clf.predict(X_test)
    #print(prediction)
    NN[i] = (len(Y_test) - sum(np.abs(prediction - Y_test))) / len(Y_test)

    clf = svm.SVC(gamma='scale',probability=True)
    clf.fit(X, Y)
    prediction = clf.predict(X_test)
    #print(prediction)
    SVM[i] = (len(Y_test) - sum(np.abs(prediction - Y_test))) / len(Y_test)

print('NN',np.mean(NN),np.std(NN))
print('SVM',np.mean(SVM),np.std(SVM))
