import numpy as np
import cv2


def to_pauli(x):
    pauli = np.zeros_like(x)
    pauli[0] = x[0] - x[1]
    pauli[1] = x[0] + x[1]
    pauli[2] = 2*x[2]
    return pauli


def to_coherency(x):
    ''' 
    calculate the upper triangle elements of the coherency matrix
    https://en.wikipedia.org/wiki/Polarization_(waves)#Coherency_matrix
    '''
    x_conjugated = np.conj(x)
    channels, height, width = x.shape

    if channels == 2:
        coherency = np.zeros((3, height, width), dtype=np.complex)
        coherency[0] = x[0] * x_conjugated[0]
        coherency[1] = x[0] * x_conjugated[1]
        coherency[2] = x[1] * x_conjugated[1]
        return coherency

    if channels == 3:
        coherency = np.zeros((6, height, width), dtype=np.complex)
        coherency[0] = x[0] * x_conjugated[0]
        coherency[1] = x[0] * x_conjugated[1]
        coherency[2] = x[0] * x_conjugated[2]
        coherency[3] = x[1] * x_conjugated[1]
        coherency[4] = x[1] * x_conjugated[2]
        coherency[5] = x[2] * x_conjugated[2]
        return coherency


def box_filter(x, filter_size=9):
    filter = (filter_size, filter_size)

    # average boxfilter
    for i in range(len(x)):
        x[i] = cv2.blur(x[i].real, filter) + cv2.blur(x[i].imag, filter) * 1j

    return x
