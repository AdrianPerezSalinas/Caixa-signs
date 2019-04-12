from signs_aux import *
import csv
import numpy as np
import os
import pandas as pd

directory = ['./Data/KO','./Data/OK']
data = []
for dir in directory:
    for filename in os.listdir(dir):
        if filename.endswith('.csv'):
            dir_filname = os.path.join(dir, filename)
            dat = pd.read_csv(dir_filname, delimiter=',')
            dat=dat.values
            dat = dat[:,1:].astype('float')
            data.append((dat, directory.index(dir)))

            '''
            mat = image_preprocess(os.path.join(directory, filename), 8)
            with open(dir_filname + '.csv', mode='w') as csv_file:
                fields = ['Measurement', 'Entropy 0', 'Entropy 1', 'Entropy 2', 'Entropy 3', 'Entropy 4',
                          'Entropy 5', 'Entropy 6', 'Entropy 7']
                writer = csv.DictWriter(csv_file, fieldnames=fields)
                writer.writeheader()
    
                sc_ket = scale_ket(mat)
                #d = CumScalesEntropy(sc_ket)
    
                #print(d)
    
                sc_ket = coords_ket(mat)
                d = CumCoordsEntropy(sc_ket)
    
                print(d)
            '''
shape = data[0][0].shape
M = list(range(shape[0]))
E = list(range(shape[1]))
ds = [d[0] for d in data]
ls = [d[1] for d in data]
maxs = np.maximum(ds[0], ds[1])
mins = np.minimum(ds[0], ds[1])
for i, d in enumerate(ds[2:]):
    maxs = np.maximum(maxs, d)
    mins = np.minimum(mins, d)


Thresholds = np.zeros(maxs.shape)
Rating = []
for m in M:
    for e in E:
        R = np.linspace(mins[m, e], maxs[m, e])
        rates = np.array([R[0], 0, 0])
        for r in R:
            success = 0
            fail = 0
            for d, label in zip(ds, ls):
                if (d[m, e] > r) == label: success += 1
                else: fail += 1
            rates_candidate = np.array([r, success, fail])

            if np.max(rates) < np.max(rates_candidate):
                rates = rates_candidate

        Rating.append(rates)
        if rates[1] < rates[2]: Rat = -rates[0]
        else: Rat = +rates[0]
        Thresholds[m, e] = Rat

signs = np.sign(Thresholds)
aciertos = 0
fallos = 0
for d, label in zip(ds, ls):
    sign_d = Thresholds*d
    suc = sign_d > Thresholds
    claim = np.sum(suc)
    if claim >= 11:
        aciertos += 1
    else: fallos += 1

print('ACIERTOS {}'.format(aciertos))


'''
    mins = np.minimum(maxs, d)

print(maxs)
print(mins)
for m in M:
    for e in E:

        print([d[0] for d in data])
        for d in data:
            nums = d[0]
            label = d[1]
            maximum = np.max([d[0][m, e] for d in data])

        #
        #minimum = np.max([d[0][m, e] for d in data])
        #print(m, e, maximum, minimum)
'''