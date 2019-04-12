from signs_aux import *
import csv
import numpy as np
import os

<<<<<<< HEAD
directory = './Data/KO'
for filename in os.listdir(directory):
    if filename == '5152867_p1_s0.png': #.endswith('.png'):
        dir_filname = os.path.join(directory, filename[:-4])
        mat = image_preprocess(os.path.join(directory, filename), 8)
=======
directory = './Data/OK'
scales=5
for filename in os.listdir(directory):
    if filename.endswith('.png'):
        mat = image_preprocess(os.path.join(directory, filename), scales)
        dir_name = directory + '/'+str(scales) + '_scales'
        createFolder(dir_name)
        dir_filname = dir_name + '/' + filename[:-4]
>>>>>>> d001298afae9cf9dee7702e0566112042bb2d94e
        with open(dir_filname + '.csv', mode='w') as csv_file:
            fields = ['Measurement','Entropy 0', 'Entropy 1', 'Entropy 2', 'Entropy 3', 'Entropy 4',
                      'Entropy 5', 'Entropy 6', 'Entropy 7']
            writer = csv.DictWriter(csv_file, fieldnames=fields)
            writer.writeheader()

<<<<<<< HEAD
            sc_ket = scale_ket(mat)
            d1 = CumScalesEntropy(sc_ket)
            d1['Measurement'] = 'CumScales'
            writer.writerow(d1)

            d1 = ScalesEntropy(sc_ket)
            d1['Measurement'] = 'Scales'
            writer.writerow(d1)

            coord_ket = coords_ket(mat)
            d1 = CumScalesEntropy(coord_ket)
            d1['Measurement'] = 'Coords'
            writer.writerow(d1)
            '''
            coord_ket = coords_ket(mat)
            d2 = CumCoordsEntropy(coord_ket)
            d2['Measurement'] = 'Coords'
            writer.writerow(d2)
            '''
=======
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
>>>>>>> d001298afae9cf9dee7702e0566112042bb2d94e

        





'''
    if filename.endswith(".png"):
        print(os.path.join(directory, filename))
        print(filename[:-4])
        continue
    else:
        continue
'''
