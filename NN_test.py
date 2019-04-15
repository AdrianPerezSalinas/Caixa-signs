import numpy as np
import os
import pandas as pd
import Neural_Network as NN

directory = ['./Data/KO','./Data/OK']
train_data = []
test_data = []
for dir in directory:
    for filename in os.listdir(dir)[:40]:
        if filename.endswith('.csv'):
            dir_filname = os.path.join(dir, filename)
            dat = pd.read_csv(dir_filname, delimiter=',')
            dat=dat.values
            dat = dat[:,1:].astype('float')
            dat=dat.flatten()
            train_data.append((dat, directory.index(dir)))

for dir in directory:
    for filename in os.listdir(dir)[40:]:
        if filename.endswith('.csv'):
            dir_filname = os.path.join(dir, filename)
            dat = pd.read_csv(dir_filname, delimiter=',')
            dat=dat.values
            dat = dat[:,1:].astype('float')
            dat=dat.flatten()
            test_data.append((dat, directory.index(dir)))


net = NN.Network([24, 20, 5, 20, 2])

net.SGD(train_data, 5, 50, eta=3, test_data=test_data)

