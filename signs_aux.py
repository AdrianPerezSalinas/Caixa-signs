import numpy as np
from scipy.linalg import logm
import cv2


def image_preprocess(img_path, scales):
    img = cv2.imread(img_path, 0)
    rows, cols = (2**scales, 2**scales)

    dst = cv2.resize(img, (cols, rows), interpolation=cv2.INTER_CUBIC)
    dst = np.max(dst) - dst

    return dst


def scale_ket(image_matrix):
    #This function encodes a matrix inside a quantum state using the scaling procedure.
    #The matrix has to be a square
    rows = max(np.shape(image_matrix))
    cols = rows
    qubits = 2*int(np.floor(np.log2(rows)))
    psi = np.zeros(2**qubits)
    for r in range(rows):
        r_string = np.binary_repr(r, width=int(qubits/2))
        for c in range(cols):
            c_string = np.binary_repr(c, width=int(qubits/2))
            string = ''
            for i in range(int(qubits/2)):
                string += r_string[i] + c_string[i]
            index = int(string, base=2)
            psi[index] = image_matrix[r,c]

    psi = psi/np.linalg.norm(psi)
    return psi

def coords_ket(image_matrix):
    #This function encodes a matrix inside a quantum state using the x/y procedure.
    #The matrix has to be a square
    rows = max(np.shape(image_matrix))
    cols = rows
    qubits = 2 * int(np.floor(np.log2(rows)))
    psi = np.zeros(2 ** qubits)
    for r in range(rows):
        for c in range(cols):
            string = np.binary_repr(r, width=int(qubits/2)) + np.binary_repr(c, width=int(qubits/2))
            index = int(string, base=2)
            psi[index] = image_matrix[r, c]

    psi = psi / np.linalg.norm(psi)
    return psi


def pond_count_scales(scales):
    numbers = []
    for i in range(2**(2*len(scales))):
        string = np.binary_repr(i, 2*len(scales))
        number = 0
        for k,s in enumerate(scales):
            number += int(string[2*k]) * 2**(2*s)
            number += int(string[2 * k + 1]) * 2 ** (2 * s + 1)
        numbers.append(number)
    return numbers


def partial_trace_scale(ket, partition):
    #partition is required to be a list of scales we want to maintain
    scale_list = list(range(int(np.log2(len(ket))/2)))
    for p in partition:
        scale_list.remove(p)

    summands = pond_count_scales(scale_list)
    parts = pond_count_scales(partition)
    rho = np.zeros((len(parts), len(parts)))
    for i, p in enumerate(parts):
        for j in range(i + 1):
            rho[i, j] = np.sum(ket[p + s] * np.conj(ket[parts[j] + s]) for s in summands)
            rho[j, i] = np.conj(rho[i, j])

    return rho

def ScalesEntropy(ket):
    num_scales = int(np.log2(len(ket))/2)
    dicti = {}
    for i in range(8):
        part = [i]
        rho_ = partial_trace_scale(ket, part)
        entr = entropy(rho_)
        dicti['Entropy {}'.format(i)] = entr

    return dicti



def CumScalesEntropy(ket):
    num_scales = int(np.log2(len(ket))/2)
    dicti = {}
    for i in range(num_scales // 2):
        part = list(range(i + 1))
        rho_ = partial_trace_scale(ket, part)
        entr = entropy(rho_)
        dicti['Entropy {}'.format(part[-1])] = entr

    dicti['Entropy {}'.format(4)] = entr

    for i in range(num_scales // 2 - 1):
        part = list(range(num_scales - 1, num_scales // 2 + i, -1))
        rho_ = partial_trace_scale(ket, part)
        entr = entropy(rho_)
        dicti['Entropy {}'.format(part[-1])] = entr

    return dicti

#############COORDS#######################

def pond_count_coords(scales):
    numbers = []
    for i in range(2**(2*len(scales))):
        string = np.binary_repr(i, 2*len(scales))
        number = 0
        for k,s in enumerate(scales):
            number += int(string[k]) * 2**(s)
            number += int(string[len(scales) + k]) * 2 ** (8 + s) #This number 8 is valid only for a total amount of scales of 8
        numbers.append(number)
    return numbers

def partial_trace_coords(ket, partition):
    #partition is required to be a list of scales we want to maintain
    scale_list = list(range(int(np.log2(len(ket))/2)))
    for p in partition:
        scale_list.remove(p)

    summands = pond_count_coords(scale_list)
    parts = pond_count_coords(partition)
    rho = np.zeros((len(parts), len(parts)))
    for i, p in enumerate(parts):
        for j in range(i + 1):
            rho[i, j] = np.sum(ket[p + s] * np.conj(ket[parts[j] + s]) for s in summands)
            rho[j, i] = np.conj(rho[i, j])

    return rho


def CoordsEntropy(ket):
    num_scales = int(np.log2(len(ket)) / 2)
    dicti = {}
    for i in range(8):
        part = [i]
        rho_ = partial_trace_coords(ket, part)
        entr = entropy(rho_)
        dicti['Entropy {}'.format(i)] = entr

    return dicti

def CumCoordsEntropy(ket):
    num_scales = int(np.log2(len(ket))/2)
    dicti = {}
    for i in range(num_scales // 2):
        part = list(range(i + 1))
        print(part)
        rho_ = partial_trace_coords(ket, part)
        entr = entropy(rho_)
        dicti['Entropy {}'.format(part[-1])] = entr

    dicti['Entropy {}'.format(4)] = entr

    for i in range(num_scales // 2 - 1):
        part = list(range(num_scales - 1, num_scales // 2 + i, -1))
        print(part)
        rho_ = partial_trace_coords(ket, part)
        entr = entropy(rho_)
        dicti['Entropy {}'.format(part[-1])] = entr

    return dicti

def entropy(rho):
    lambdas = np.linalg.eigvalsh(rho)
    log_lambdas = np.zeros(len(lambdas))
    for l in lambdas:
        if l > 1e-5:
            log_lambdas = np.log(l)
    return -np.sum(lambdas*log_lambdas)

'''
def partial_trace(rho, i):
    #This function performs the partial trace respect to one of the qubits
    qubits = int(np.round(np.log2(rho.shape[0])))
    #rho = np.outer(psi, np.conj(psi))
    red_rho = np.zeros((2**(qubits - 1), 2**(qubits - 1)))
    for r in range(2**(qubits-1)):
        r_ = r % (2**i) + 2*(r - r % (2**i))
        for c in range(r, 2**(qubits-1)):
            c_ = c % (2 ** i) + 2 * (c - c % (2 ** i))

            red_rho[r, c] = rho[r_, c_] + rho[r_ + 2**i, c_ + 2**i]
            red_rho[c, r] = np.conj(red_rho[r, c])


    return red_rho


def trace_scale(rho, i):
    #This function performs the partial trace respect to one of the scales
    #It will work only for kets encoded properly
    qubits = int(np.round(np.log2(rho.shape[0])))
    if i > qubits/2:
        raise ValueError('Non sense size')

    rho_ = partial_trace(rho, 2 * i)
    red_rho = partial_trace(rho_, 2 * i) #This second line is important!!

    return red_rho

def trace_coords(rho, i, coord):
    # This function performs the partial trace respect to one of the scales
    # It will work only for kets encoded properly
    qubits_per_coord = int(np.round(np.log2(rho.shape[0]))/2)
    if coord == 'x': q = i
    if coord == 'y': q = qubits_per_coord + i

    rho_ = partial_trace(rho, q)

    return rho_



def CumScalesEntropy(rho, scales):
    rho_ = rho.copy()
    dict = {}
    for i in range(scales):
        rho_ = trace_scale(rho_, 0)
        e = np.real(entropy(rho_))
        dict['Entropy {}'.format(i)] = e

    return dict, [i for i in range(scales)]


def InvCumScalesEntropy(rho, scales):
    rho_ = rho.copy()
    dict = {}
    for i in range(scales-1,-1,-1):
        rho_ = trace_scale(rho_, i)
        e = np.real(entropy(rho_))
        dict['Entropy {}'.format(i)] = e

    return dict, [i for i in range(scales-1,-1,-1)]

def CumCoordsEntropy(rho, coord, scales):
    rho_ = rho.copy()
    dict = {}
    for i in range(scales):
        rho_ = trace_coords(rho_, 0, coord)
        e = np.real(entropy(rho_))
        dict['Entropy {}'.format(i)] = e

    return dict, [i for i in range(scales)]

def InvCumCoordsEntropy(rho, coord, scales):
    rho_ = rho.copy()
    dict = {}
    for i in range(scales-1, -1, -1):
        rho_ = trace_coords(rho_, i, coord)
        e = np.real(entropy(rho_))
        dict['Entropy {}'.format(i)] = e

    return dict, [i for i in range(scales-1,-1,-1)]
'''
