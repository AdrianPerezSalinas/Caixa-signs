from signs_aux import *
import csv
import numpy as np
import os
import pandas as pd
import random

directory = ['./Data/KO', './Data/OK']
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
random.shuffle(data)
ds = [d[0] for d in data[:60]]
for i,ent in enumerate(ds):
    ent = np.delete(ent, 4)
    ds[i] = ent

ds_test = [d[0] for d in data[60:]]
for i,ent in enumerate(ds_test):
    ent = np.delete(ent, 4)
    ds_test[i] = ent

ls = [d[1] for d in data[:31]]
ls_test = [d[1] for d in data[31:]]
maxs = np.maximum(ds[0], ds[1])
mins = np.minimum(ds[0], ds[1])
for i, d in enumerate(ds[2:]):
    maxs = np.maximum(maxs, d)
    mins = np.minimum(mins, d)

Thresholds = np.zeros(maxs.shape)
Rating = []
for m in range(maxs.shape[0]):
    R = np.linspace(mins[m], maxs[m])
    rates = np.array([R[0], 0, 0])
    for r in R:
        success = 0
        fail = 0
        for d, label in zip(ds, ls):
            if (d[m] > r) == label: success += 1
            else: fail += 1
        rates_candidate = np.array([r, success, fail])

        if np.max(rates) < np.max(rates_candidate):
            rates = rates_candidate

    Rating.append(rates)
    if rates[1] < rates[2]: Rat = -rates[0]
    else: Rat = +rates[0]
    Thresholds[m] = Rat

signs = np.sign(Thresholds)
aciertos = 0
for d, label in zip(ds_test, ls_test):
    sign_d = signs*d
    suc = sign_d > Thresholds
    claim = np.sum(suc)
    if claim > 12:
        lab = 1
    else:
        lab = 0

    aciertos += int(lab == label)

print('ACIERTOS: {}/{}'.format(aciertos, len(ds_test)))

