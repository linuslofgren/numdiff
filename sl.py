from twopBVP import twopBVP
from scipy import sparse
from scipy.linalg import toeplitz, inv
from scipy.linalg import eigh
import scipy.linalg as linalg
import numpy as np
import matplotlib.pyplot as plt

def SL(N, L, alpha, beta):
    delta_x = L/(N+1)
    delta_x_squared = delta_x ** 2
    T = np.array(toeplitz([-2, 1]+[0]*(N-2), [-2, 1] + [0]*(N-2)))      #gör en rad kortare och ändra där nere också
    print(T)
    t1 = np.array((N-2)*[0]+[2/3, -2/3])
    T = T[:-1, :]
    T = np.concatenate((T, [t1]))
    print(T)
    T = (1/delta_x_squared) * T
    l, u = eigh(T)
    l = np.flip(l)
    u = np.flip(u)
    #bc = np.array([-alpha] + [0]*(N-2) + [-beta])
    return l, u

eigen_values, eigen_vectors = SL(499 , 1, 0, 0)
print(len(eigen_values))
x = np.linspace(0, 1, 499)
print(eigen_values[0:3])
plt.plot(x, eigen_vectors[1], "2")
#plt.plot(x, eigen_vectors[0])
l = -4*500**2*np.sin(np.pi/(2*500))**2
#print(l, np.max(eigen_values), eigen_values[0])
#plt.plot(x, l, '2')
plt.show()