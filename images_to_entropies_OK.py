from signs_aux import *
import csv
import numpy as np
import os

directory = './Data/OK'
for filename in os.listdir(directory):
    if filename.endswith('.png'):
        dir_filname = os.path.join(directory, filename[:-4])
        mat = image_preprocess(os.path.join(directory, filename))
        with open(dir_filname + '.csv', mode='w') as csv_file:
            fields = ['Measurement','Entropy 0', 'Entropy 1', 'Entropy 2', 'Entropy 3', 'Entropy 4',
                      'Entropy 5', 'Entropy 6']
            writer = csv.DictWriter(csv_file, fieldnames=fields)
            writer.writeheader()

            sc_ket = scale_ket(mat)
            rho = np.outer(sc_ket, sc_ket)
            d, l = CumScalesEntropy(rho)
            d['Measurement'] = 'Cumulative Scale Entropy'
            print(d)
            writer.writerow(d)

            d, l = InvCumScalesEntropy(rho)
            d['Measurement'] = 'Inverse Cumulative Scale Entropy'
            print(d)
            writer.writerow(d)

            d, l = CumCoordsEntropy(rho,'x')
            d['Measurement'] = 'Cumulative x-coord Entropy'
            print(d)
            writer.writerow(d)

            d, l = CumCoordsEntropy(rho, 'y')
            d['Measurement'] = 'Cumulative y-coord Entropy'
            print(d)
            writer.writerow(d)

            d, l = InvCumCoordsEntropy(rho, 'x')
            d['Measurement'] = 'Inverse cumulative x-coord Entropy'
            print(d)
            writer.writerow(d)

            d, l = InvCumCoordsEntropy(rho, 'y')
            d['Measurement'] = 'Inverse cumulative y-coord Entropy'
            print(d)
            writer.writerow(d)

        print(dir_filname)





'''
    if filename.endswith(".png"):
        print(os.path.join(directory, filename))
        print(filename[:-4])
        continue
    else:
        continue
'''
