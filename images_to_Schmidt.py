from signs_aux import *
import csv
import numpy as np
import os

directory = './Data/OK'
for filename in os.listdir(directory):
    if filename.endswith('.png'):
        dir_filname = os.path.join(directory, filename[:-4])
        mat = image_preprocess(os.path.join(directory, filename), 8)
        sc_ket = scale_ket(mat)

        rho = partial_trace_scale(sc_ket, [0, 1, 2, 3])
        lambdas = np.abs(np.linalg.eigvalsh(rho))
        log_lambdas = np.log(lambdas)
        np.savetxt(dir_filname + '.out', (lambdas, log_lambdas), delimiter=',')  # X is an array
        print(dir_filname)
        '''
        with open(dir_filname + '.csv', mode='w') as csv_file:
            fields = ['Measurement','Entropy 0', 'Entropy 1', 'Entropy 2', 'Entropy 3', 'Entropy 4',
                      'Entropy 5', 'Entropy 6', 'Entropy 7']
            writer = csv.DictWriter(csv_file, fieldnames=fields)
            writer.writeheader()

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


        print(dir_filname)
        '''