import numpy as np
import os
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn import svm

scales=[0,3,6]
directory = ['./Firmas2/OK/CSV_0.3_' + str(scales), './Firmas2/KO/CSV_0.3_' + str(scales)]
KO_data = []
OK_data = []


black_patch = mpatches.Patch(color='black', label='OK')
red_patch = mpatches.Patch(color='red', label='KO')

plt.show()
dir = directory[0]
fig, ax = plt.subplots(2)
for filename in os.listdir(dir):
    if filename.endswith('.out'):
        dir_filname = os.path.join(dir, filename)
        dat = np.loadtxt(dir_filname, delimiter=',')[0]
        #dat = np.flip(dat)
        dat = (np.cumsum(dat))
        ax[0].plot(dat, 'k.', alpha = .3, markersize = 1)
        KO_data.append((dat, directory.index(dir)))

ax[0].set(ylim = (0, 1 ), ylabel='$\sum\lambda$')
ax[0].legend(handles=[black_patch])
dir = directory[1]
for filename in os.listdir(dir):
    if filename.endswith('.out'):
        dir_filname = os.path.join(dir, filename)
        dat = np.loadtxt(dir_filname, delimiter=',')[0]
        #dat = np.flip(dat)
        dat = (np.cumsum(dat))
        ax[1].plot(dat, 'r.', alpha = .3, markersize = 1)
        OK_data.append((dat, directory.index(dir)))

ax[1].set(ylim = (0, 1 ), xlabel = r'Number of Eigenvalue', ylabel='$\sum\lambda$')
ax[1].legend(handles = [red_patch])
fig.savefig('Eigenvalues/cum_sum/' + str(scales))
'''
lim1 = 200
lim2 = 400
steps=2
NN=np.zeros(steps)
SVM=np.zeros(steps)
for i in range(steps):
    random.shuffle(OK_data)
    random.shuffle(KO_data)
    train_data = OK_data[:lim1] + KO_data[:lim1]
    val_data = OK_data[lim1:lim2] + KO_data[lim1:lim2]
    test_data = OK_data[lim2:] + KO_data[lim2:]
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
'''
