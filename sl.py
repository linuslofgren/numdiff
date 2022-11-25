from twopBVP import twopBVP
from scipy import sparse
from scipy.linalg import toeplitz, inv
from scipy.sparse.linalg import eigs
from scipy.linalg import eig, eigh
import scipy.linalg as linalg
import numpy as np
import time
import matplotlib.pyplot as plt

def SL(N, L, alpha, beta):
    # Approach 2
    delta_x = L/(N+1/2)
    t1 = np.array((N-2)*[0]+[1, -1])

    delta_x_squared = delta_x ** 2
    
    T = toeplitz([-2, 1]+[0]*(N-2), [-2, 1] + [0]*(N-2))    #gör en rad kortare och ändra där nere också
    
    T[-1,:] = t1
    
    T = (1/delta_x_squared) * T
    number_eigenvalue = N
    
    l, u = eig(T)
    
    u = np.vstack(([alpha]*number_eigenvalue, u))
    sorted_indexes = l.argsort()[::-1]
    u = np.vstack((u, 1/3*(2*beta+4*u[-1,:]-u[-2,:])))
    
    return l, u , sorted_indexes

def v(j, x):
    return np.sqrt(2)*np.sin((2*j-1)*np.pi*x/(2))

def lam(j):
    return -(2*j-1)**2*np.pi**2/(4)

N = 200
l, u, idx = SL(N, 1, 0, 0)
x = np.linspace(0, 1, N+2)

for i in range(1,6,2):
    plt.plot(x, -4.5*u[:,idx[i]])
plt.show()

def l_eig(j, L=1):
    return -((2*j-1)**2)*(np.pi**2)/(4*L**2)

errors = []
to = 100
for N in range(10, to):
    l, u, idx = SL(N , 1, 0, 0)
    errors.append(np.abs(l_eig(1)-l[idx[0]]))
    lambdaexact = [l_eig(1), l_eig(2), l_eig(3)]
plt.loglog([x for x in range(10, to)], errors, "2")
plt.loglog([x for x in range(10, to)], [(x)**-2 for x in range(10, to)])
plt.show()