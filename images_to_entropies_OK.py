from signs_aux import *
import csv
import numpy as np
import os

directory = './Data/OK'
scales=5
for filename in os.listdir(directory):
    if filename.endswith('.png'):
        mat = image_preprocess(os.path.join(directory, filename), scales)
        dir_name = directory + '/'+str(scales) + '_scales'
        createFolder(dir_name)
        dir_filname = dir_name + '/' + filename[:-4]
        with open(dir_filname + '.csv', mode='w') as csv_file:
            fields = ['Measurement','Entropy 0', 'Entropy 1', 'Entropy 2', 'Entropy 3', 'Entropy 4',
                      'Entropy 5', 'Entropy 6']
            writer = csv.DictWriter(csv_file, fieldnames=fields)
            writer.writeheader()

            ket = scale_ket(mat)
            rho = np.outer(ket, ket)
            d, l = CumScalesEntropy(rho, scales)
            d['Measurement'] = 'Cumulative Scale Entropy'
            writer.writerow(d)

            d, l = InvCumScalesEntropy(rho, scales)
            d['Measurement'] = 'Inverse Cumulative Scale Entropy'
            writer.writerow(d)
            
            ket = coords_ket(mat)
            rho = np.outer(ket, ket)
            
            d, l = CumCoordsEntropy(rho,'x', scales)
            d['Measurement'] = 'Cumulative x-coord Entropy'
            writer.writerow(d)
            
            d, l = InvCumCoordsEntropy(rho, 'x', scales)
            d['Measurement'] = 'Inverse cumulative x-coord Entropy'
            writer.writerow(d)

            d, l = CumCoordsEntropy(rho, 'y', scales)
            d['Measurement'] = 'Cumulative y-coord Entropy'
            writer.writerow(d)

            d, l = InvCumCoordsEntropy(rho, 'y', scales)
            d['Measurement'] = 'Inverse cumulative y-coord Entropy'
            writer.writerow(d)

        





'''
    if filename.endswith(".png"):
        print(os.path.join(directory, filename))
        print(filename[:-4])
        continue
    else:
        continue
'''
