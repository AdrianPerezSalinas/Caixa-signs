import numpy as np
from scipy.linalg import logm
def locator(image_matrix):
    rows = max(np.shape(image_matrix))
    cols = rows
    qubits = 2*int(np.floor(np.log2(rows)))
    pow = 2**(qubits - 1)
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


def partial_trace(rho, i):
    qubits = int(np.round(np.log2(rho.shape[0])))
    #rho = np.outer(psi, np.conj(psi))
    red_rho = np.zeros((2**(qubits - 1), 2**(qubits - 1)))
    for r in range(2**(qubits-1)):
        r_ = r % (2**i) + 2*(r - r % (2**i))
        for c in range(r, 2**(qubits-1)):
            c_ = c % (2 ** i) + 2 * (c - c % (2 ** i))

            red_rho[r,c] = rho[r_, c_] + rho[r_ + 2**i, c_ + 2**i]
            red_rho[c, r] = np.conj(red_rho[r,c])


    return red_rho


def traces_scale(rho, i):
    qubits = int(np.round(np.log2(rho.shape[0])))
    if i > qubits/2:
        raise ValueError('Non sense size')

    rho_ = partial_trace(rho, 2*i)
    red_rho = partial_trace(rho_, 2 * i) #This second line is important!!

    return red_rho

def entropy(rho):
    lambdas = np.linalg.eigvalsh(rho)
    log_lambdas = np.zeros(len(lambdas))
    for l in lambdas:
        if l > 1e-5:
            log_lambdas = np.log(l)
    return -np.sum(lambdas*log_lambdas)