import numpy as np
import os
import random
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from scipy.optimize import minimize

scales=[0, 2, 4, 6]
print(scales)
directory = ['./Firmas2/KO/CSV_0.3_' + str(scales), './Firmas2/OK/CSV_0.3_' + str(scales)]
KO_data = []
OK_data = []
dir = directory[0]
for filename in os.listdir(dir):
    if filename.endswith('.out'):
        dir_filname = os.path.join(dir, filename)
        dat = np.loadtxt(dir_filname, delimiter=',')[0]
        KO_data.append((dat, directory.index(dir)))


dir = directory[1]
for filename in os.listdir(dir):
    if filename.endswith('.out'):
        dir_filname = os.path.join(dir, filename)
        dat = np.loadtxt(dir_filname, delimiter=',')[0]
        OK_data.append((dat, directory.index(dir)))


'''
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


    clf = MLPClassifier(solver='lbfgs', activation='relu', learning_rate='adaptive')
    clf.fit(X,Y)
    prediction = clf.predict(X_test)
    NN[i] = 1 - (np.sum(np.abs(prediction - Y_test)) / len(Y_test))
'''
    # nn = 1
    # T = 0
    # for threshold in np.linspace(0,1):
    #     vector = np.asfarray(prediction[:,1]>threshold)
    #     nn_ = (np.sum(np.abs(vector - Y_test)) / len(Y_test))
    #     if nn_ < nn:
    #         T = threshold
    #         nn = nn_

    # NN[i] = 1 - nn
'''
    clf = svm.SVC(gamma='auto', kernel='sigmoid')
    clf.fit(X, Y)
    prediction = clf.predict(X_test)
    SVM[i] = 1 - (np.sum(np.abs(prediction - Y_test)) / len(Y_test))
    # nn = 1
    # T = 0
    # for threshold in np.linspace(0, 1):
    #     vector = np.asfarray(prediction[:, 1] > threshold)
    #     nn_ = (np.sum(np.abs(vector - Y_test)) / len(Y_test))
    #     if nn_ < nn:
    #         T = threshold
    #         nn = nn_
    #
    # SVM[i] = 1 - nn
'''
#print('NN',np.mean(NN),np.std(NN))
#print('SVM',np.mean(SVM),np.std(SVM))

lim=200
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
    x, y = d
    X_test.append(x)
    Y_test.append(y)
clf = MLPClassifier(solver='lbfgs', activation='relu', learning_rate='adaptive')
def adapt_weights(weights, X, Y, X_test, Y_test):
    X_ = []
    for x in X:
        X_.append(x*weights)


    clf.fit(X_, Y)
    prediction = clf.predict(X_test)
    return (np.sum(np.abs(prediction - Y_test)) / len(Y_test))

weights = np.ones(len(X[0]))

adapt_weights(weights, X, Y, X_test, Y_test)

result = minimize(adapt_weights, weights, args=(X, Y, X_test, Y_test))

print(result)
