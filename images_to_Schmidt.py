from signs_aux import *
import csv
import numpy as np
import os


scales=[0,2,4]
directory = '/media/adrianps/Files/Archivos/Trabajos/19_LaCaixa/Firmas/Caixa-signs/Firmas2/OK'
files = '/media/adrianps/Files/Archivos/Firmas2/OK'
print(len(os.listdir(files)))
for filename in os.listdir(files)[:min(len(os.listdir(files)), 1000)]:
    if filename.endswith('.png'):
        dir=directory + '/CSV_0.4_'+str(scales) + '_coords'
        createFolder(dir)
        dir_filname = os.path.join(dir, filename[:-4])
        mat = image_preprocess(os.path.join(files, filename), 8)
        sc_ket = coords_ket(mat)
        rho = partial_trace_scale(sc_ket, scales)
        lambdas_ = (np.linalg.eigvalsh(rho))
        r = (np.linalg.matrix_rank(rho))
        lambdas = np.array([0]*(2**(2*len(scales)) - r) + list(np.linalg.eigvalsh(rho)[-r:]))
        np.savetxt(dir_filname + '.out', lambdas, delimiter=',')  # X is an array
        print(dir_filname)

directory = '/media/adrianps/Files/Archivos/Trabajos/19_LaCaixa/Firmas/Caixa-signs/Firmas2/KO'
files = '/media/adrianps/Files/Archivos/Firmas2/KO'
print(len(os.listdir(files)))
for filename in os.listdir(files)[:min(len(os.listdir(files)), 1000)]:
    if filename.endswith('.png'):
        dir = directory + '/CSV_0.4_' + str(scales) + '_coords'
        createFolder(dir)
        dir_filname = os.path.join(dir, filename[:-4])
        mat = image_preprocess(os.path.join(files, filename), 8)
        sc_ket = coords_ket(mat)
        rho = partial_trace_scale(sc_ket, scales)
        lambdas_ = (np.linalg.eigvalsh(rho))
        r = (np.linalg.matrix_rank(rho))
        lambdas = np.array([0] * (2 ** (2 * len(scales)) - r) + list(np.linalg.eigvalsh(rho)[-r:]))
        np.savetxt(dir_filname + '.out', lambdas, delimiter=',')  # X is an array
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