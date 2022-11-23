from twopBVP import twopBVP
from scipy import sparse
from scipy.linalg import toeplitz
from scipy.sparse.linalg import inv
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
import scipy.linalg as linalg
import numpy as np
import matplotlib.pyplot as plt

def SL(N, L, alpha, beta):
    delta_x = L/(N+1)
    delta_x_squared = delta_x ** 2
    T = (1/delta_x_squared) * sparse.csc_matrix(toeplitz([-2, 1]+[0]*(N-2), [-2, 1] + [0]*(N-2)))
    w, v = eigh(T.toarray())
    return w, v

eigen_values, eigen_vectors = SL(499, 1, 0, 0)
print(len(eigen_values))
plt.plot(np.linspace(0, 1, len(eigen_vectors[0])), eigen_vectors[0])
plt.show()