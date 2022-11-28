from scipy import sparse
from scipy.linalg import toeplitz
from scipy.sparse.linalg import inv, spsolve_triangular, spsolve
import numpy as np


def twopBVP(f_vec, alpha, beta, L, N):
    delta_x = L/(N+1)
    delta_x_squared = delta_x ** 2
    T_times_delta_x = sparse.csr_matrix(toeplitz([-2, 1]+[0]*(N-2), [-2, 1] + [0]*(N-2)))
    bc = [-alpha] + [0]*(N-2) + [-beta]
    Y = spsolve(T_times_delta_x, (f_vec*delta_x_squared + bc))
    return Y
    
