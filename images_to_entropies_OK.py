from signs_aux import *
import csv
import numpy as np
import os
directory = '/home/adrianps/PhD-BSC/github/Caixa-signs/Firmas2/KO'
files = '/media/adrianps/Files/Archivos/Firmas2/KO'
print(len(os.listdir(files)))
function = 1
for filename in os.listdir(files):
    if filename.endswith('.png'):
        dir=directory + '/CSV_0.2_' + str(function)
        dir_filname = os.path.join(dir, filename[:-4])
        mat = image_preprocess(os.path.join(files, filename), 8)
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
    if filename.endswith(".png"):
        print(os.path.join(directory, filename))
        print(filename[:-4])
        continue
    else:
        continue
'''
