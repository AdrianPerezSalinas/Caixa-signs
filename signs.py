import cv2
import numpy as np
from signs_aux import *
import matplotlib.pyplot as plt


KOs = ['Data/KO/5149433_p1_s0.png',
'Data/KO/5149435_p1_s0.png',
'Data/KO/5149436_p1_s0.png',
'Data/KO/5149728_p1_s0.png',
'Data/KO/5149788_p4_s0.png',
'Data/KO/5149789_p1_s0.png',
'Data/KO/5149790_p1_s0.png',
'Data/KO/5150142_p1_s0.png',
'Data/KO/5150187_p1_s0.png',
'Data/KO/5150305_p1_s0.png',
'Data/KO/5150306_p1_s0.png',
'Data/KO/5150403_p1_s0.png',
'Data/KO/5150404_p1_s0.png',
'Data/KO/5150405_p1_s0.png',
'Data/KO/5150436_p4_s0.png',
'Data/KO/5150406_p1_s0.png',
'Data/KO/5150735_p2_s0.png',
'Data/KO/5150805_p2_s0.png',
'Data/KO/5150806_p8_s0.png',
'Data/KO/5150845_p1_s0.png',
'Data/KO/5150846_p2_s0.png',
'Data/KO/5150862_p1_s0.png',
'Data/KO/5151056_p1_s0.png',
'Data/KO/5150950_p1_s0.png',
'Data/KO/5151057_p1_s0.png',
'Data/KO/5151058_p1_s0.png',
'Data/KO/5151071_p1_s0.png',
'Data/KO/5151364_p2_s0.png',
'Data/KO/5151511_p1_s0.png',
'Data/KO/5151512_p2_s0.png',
'Data/KO/5151513_p2_s0.png',
'Data/KO/5151516_p2_s0.png',
'Data/KO/5151675_p1_s0.png',
'Data/KO/5151676_p1_s0.png',
'Data/KO/5151795_p1_s0.png',
'Data/KO/5151842_p1_s0.png',
'Data/KO/5152272_p1_s0.png',
'Data/KO/5152273_p1_s0.png',
'Data/KO/5152274_p1_s0.png',
'Data/KO/5152286_p1_s0.png',
'Data/KO/5152394_p8_s0.png',
'Data/KO/5152395_p3_s0.png',
'Data/KO/5152403_p1_s0.png',
'Data/KO/5152483_p1_s0.png',
'Data/KO/5152487_p12_s0.png',
'Data/KO/5152488_p3_s0.png',
'Data/KO/5152528_p1_s0.png',
'Data/KO/5152867_p1_s0.png',
'Data/KO/5152875_p1_s0.png']

OKs = ['Data/OK/5148594_p3_s0.png',
'Data/OK/5148632_p4_s0.png',
'Data/OK/5148624_p3_s0.png',
'Data/OK/5148614_p6_s0.png',
'Data/OK/5148623_p1_s0.png',
'Data/OK/5148622_p1_s0.png',
'Data/OK/5148621_p2_s0.png',
'Data/OK/5148616_p1_s0.png',
'Data/OK/5148615_p1_s0.png',
'Data/OK/5148630_p1_s0.png',
'Data/OK/5148613_p2_s0.png',
'Data/OK/5148638_p3_s0.png',
'Data/OK/5148633_p1_s0.png',
'Data/OK/5148639_p2_s0.png',
'Data/OK/5148641_p6_s0.png',
'Data/OK/5148640_p1_s0.png',
'Data/OK/5148646_p1_s0.png',
'Data/OK/5148647_p2_s0.png',
'Data/OK/5148649_p4_s0.png',
'Data/OK/5148650_p3_s0.png',
'Data/OK/5148655_p1_s0.png',
'Data/OK/5148656_p1_s0.png',
'Data/OK/5148657_p6_s0.png',
'Data/OK/5148658_p8_s0.png',
'Data/OK/5148663_p6_s0.png',
'Data/OK/5148664_p2_s0.png',
'Data/OK/5148665_p1_s0.png',
'Data/OK/5148666_p8_s0.png',
'Data/OK/5148671_p3_s0.png',
'Data/OK/5148672_p1_s0.png',
'Data/OK/5148673_p2_s0.png',
'Data/OK/5148674_p1_s0.png',
'Data/OK/5148679_p8_s0.png',
'Data/OK/5148680_p9_s0.png',
'Data/OK/5148681_p1_s0.png',
'Data/OK/5148682_p6_s0.png',
'Data/OK/5149158_p8_s0.png',
'Data/OK/5149159_p1_s0.png',
'Data/OK/5149160_p3_s0.png',
'Data/OK/5149185_p4_s0.png',
'Data/OK/5149182_p1_s0.png',
'Data/OK/5149181_p4_s0.png',
'Data/OK/5149180_p2_s0.png',
'Data/OK/5149179_p1_s0.png',
'Data/OK/5149172_p1_s0.png',
'Data/OK/5149171_p3_s0.png',
'Data/OK/5149170_p1_s0.png',
'Data/OK/5149169_p1_s0.png',
'Data/OK/5149161_p3_s0.png']

fig, ax = plt.subplots()
for _ in range(min(len(KOs), len(OKs))): #range(min(len(OKs),len(KOs))):
    img_path = OKs[_]
    img = cv2.imread(img_path, 0)
    rows, cols = (64, 64)

    dst = cv2.resize(img, (cols, rows), interpolation=cv2.INTER_CUBIC)

    #cv2.imshow('sign', dst)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    dst = np.max(dst) - dst

    psi = locator(dst)
    rho = np.outer(psi, psi)
    Entropies_OK = []
    rho = traces_scale(rho, 0)
    for i in range(5):
        red_rho = traces_scale(rho, i)
        e = np.real(entropy(red_rho))
        print('Entropy for size {} = {}'.format(i, e))
        Entropies_OK.append(e)


    img_path = KOs[_]
    img = cv2.imread(img_path, 0)
    rows, cols = (64, 64)

    dst = cv2.resize(img, (cols, rows), interpolation=cv2.INTER_CUBIC)

    #cv2.imshow('scrawl', dst)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    dst = np.max(dst) - dst

    psi = locator(dst)
    rho = np.outer(psi, psi)
    Entropies_KO = []
    rho = traces_scale(rho, 0)
    for i in range(5):
        red_rho = traces_scale(rho, i)
        e = np.real(entropy(red_rho))
        print('Entropy for size {} = {}'.format(i, e))
        Entropies_KO.append(e)

    x = [1, 2, 3, 4, 5]
    ax.plot(x, Entropies_OK, 'k', label = 'signs')
    ax.plot(x, Entropies_KO, 'r', label = 'scrawls')
    ax.set(xlabel='sizes', ylabel='Entropy', title='Entropies // 0')
    fig.savefig('Entropies respect 0')


