from signs_aux import *
import csv
import numpy as np
import os
import pandas as pd
import random

scales_list=[[0],[1],[2],[3],[4],[5],[6],[7],[0,1],[0,1,2],[0,1,2,3],[0,2],[0,2,4],[0,2,4,6]]
directory = '/media/adrianps/Files/Archivos/Trabajos/19_LaCaixa/Firmas/Caixa-signs/Firmas2/OK'
files = '/media/adrianps/Files/Archivos/Firmas2/OK'
print(len(os.listdir(files)))
for scale in scales_list:
        print(os.listdir(directory + '/CSV_0.4_'+str(scale) + '_scales')[362])

