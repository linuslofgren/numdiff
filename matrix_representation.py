from scipy.linalg import toeplitz
import numpy as np

def first_derivative(delta_x, N):
    S = 1/(2*delta_x)*toeplitz([-1]+[0]*(N-1), [-1, 0, 1] + [0]*(N-3))[:-2,]
    row1 = np.roll(S[0,:], -1)
    rowN = np.roll(S[-1,:], 1)
    S = np.vstack((row1, S, rowN))
    return S

def second_derivative(delta_x, N):
    T = 1/(delta_x**2)*toeplitz([-2, 1]+[0]*(N-2), [-2, 1] + [0]*(N-2))
    T[0][-1]=T[1][0]
    T[-1][0]=T[-2][-1]
    return T