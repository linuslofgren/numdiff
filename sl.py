from twopBVP import twopBVP
from scipy import sparse
from scipy.linalg import toeplitz, inv
from scipy.sparse.linalg import eigs
from scipy.linalg import eig, eigh
import scipy.linalg as linalg
import numpy as np
import time
import matplotlib.pyplot as plt

def SL2(N, L, alpha, beta):
    delta_x = L/(N)
    delta_x_squared = delta_x ** 2
    T = np.array(toeplitz([-2, 1]+[0]*(N-2), [-2, 1] + [0]*(N-2)))      #gör en rad kortare och ändra där nere också
    # print(T)
    t1 = np.array((N-2)*[0]+[2, -2])
    T = T[:-1, :]
    T = np.concatenate((T, [t1]))
    print(inv(T))
    T = (1/delta_x_squared) * T
    l, u = eig(T)
    sorted_indexes = l.argsort()
    # res = np.flip(sorted(zip(l, u)))
    #bc = np.array([-alpha] + [0]*(N-2) + [-beta])
    return l, u , sorted_indexes


def SL(N, L, alpha, beta):
    # Approach 1
    # delta_x = L/(N)
    # t1 = np.array((N-2)*[0]+[2, -2])

    # Approach 2
    delta_x = L/(N+1/2)
    t1 = np.array((N-2)*[0]+[1, -1])

    # Approach 3
    # delta_x = L/(N+1)
    # t1 = np.array((N-2)*[0]+[-2, 2])
    # T = toeplitz([2, -1]+[0]*(N-2), [2, -1] + [0]*(N-2))    #gör en rad kortare och ändra där nere också

    delta_x_squared = delta_x ** 2
    # print("making toeplitz")
    T = toeplitz([-2, 1]+[0]*(N-2), [-2, 1] + [0]*(N-2))    #gör en rad kortare och ändra där nere också
    # T = toeplitz([0, 1]+[0]*(N-2), [0, 1] + [0]*(N-2))    #gör en rad kortare och ändra där nere också
    # print("made toeplitz")
    # https://www.scirp.org/pdf/JEMAA_2014092510103331.pdf
    
    # T[-1,:] = t1
    # print(T)
    # T = sparse.csr_matrix(T)
    # Eller:
    # t1 = np.array((N-2)*[0]+[2, -2])
    T[-1,:] = t1
    # T = sparse.csr_matrix(T).asfptype()
    # T = np.concatenate((T, [t1]))
    # T_i = inv(T)
    # print(T)
    T = (1/delta_x_squared) * T
    number_eigenvalue = N
    # tim = time.time()
    # l, u = eigs(T, k=number_eigenvalue, which="SR")
    # print(time.time()-tim)
    # tim = time.time()
    l, u = eig(T)
    # l = 1/delta_x_squared * (-2 + l)
    # print(time.time()-tim)
    u = np.vstack(([alpha]*number_eigenvalue, u))
    sorted_indexes = l.argsort()[::-1]
    # res = np.flip(sorted(zip(l, u)))
    u = np.vstack((u, 1/3*(2*beta+4*u[-1,:]-u[-2,:])))
    #bc = np.array([-alpha] + [0]*(N-2) + [-beta])
    return l, u , sorted_indexes

def v(j, x):
    return np.sqrt(2)*np.sin((2*j-1)*np.pi*x/(2))

def lam(j):
    return -(2*j-1)**2*np.pi**2/(4)

N = 200
l, u, idx = SL(N, 1, 0, 0)
x = np.linspace(0, 1, N+2)
# for i in [0,2]:
#     plt.plot(x, v(i, x))

# print(l[idx[0]],lam(0), lam(1), lam(2), l[idx[1]])
for i in range(1,6,2):
    plt.plot(x, -4.5*u[:,idx[i]])
plt.show()

# def l_eig(j, L=1):
#     return -j**2*np.pi**2
def l_eig(j, L=1):
    return -((2*j-1)**2)*(np.pi**2)/(4*L**2)
# print(l[idx[0]])
# print([l_eig(1), l_eig(2), l_eig(3)])
# (-2.5941534926321834+0j) -2.4674011002723395 -2.4674011002723395 -22.206609902451056 (-23.246546059158415+0j)
# (-2.7324055946641863+0j) -2.4674011002723395 -2.4674011002723395 -22.206609902451056 (-24.47978724853579+0j)
errors = []
errors2 = []
errors3 = []
to = 100
for N in range(10, to):
    # print(res[0][0])
    l, u, idx = SL(N , 1, 0, 0)
    # plt.plot(np.linspace(0, 1, 10), res[1][0], "2")
    # plt.show()
    errors.append(np.abs(l_eig(1)-l[idx[0]]))
    errors2.append(np.sqrt(1/(N+1))*linalg.norm(l_eig(2)-l[idx[1]]))
    errors3.append(np.sqrt(1/(N+1))*linalg.norm(l_eig(3)-l[idx[2]]))
    # print("First three", res[0:3, 1])

    lambdaexact = [l_eig(1), l_eig(2), l_eig(3)]
    print(lambdaexact)
    print(l[idx[0:3]])
plt.loglog([x for x in range(10, to)], errors, "2")
# plt.loglog([x for x in range(10, 100)], errors2, "3")
# plt.loglog([x for x in range(10, 100)], errors3, "4")
plt.loglog([x for x in range(10, to)], [(x)**-2 for x in range(10, to)])
# plt.loglog([x for x in range(10, 100)], [(x)**2 for x in range(10, 100)])
# plt.loglog([x for x in range(10, 100)], [(x)**-3 for x in range(10, 100)])
plt.show()