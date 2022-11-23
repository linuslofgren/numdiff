from twopBVP import twopBVP
from scipy import sparse
from scipy.linalg import toeplitz, inv
from scipy.linalg import eig
import scipy.linalg as linalg
import numpy as np
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
    delta_x = L/(N+1)
    delta_x_squared = delta_x ** 2
    T = np.array(toeplitz([-2, 1]+[0]*(N-2), [-2, 1] + [0]*(N-2)))      #gör en rad kortare och ändra där nere också
    # print(T)
    # https://www.scirp.org/pdf/JEMAA_2014092510103331.pdf
    t1 = np.array((N-2)*[0]+[1, -1])
    # Eller: t1 = np.array((N-2)*[0]+[2, -2])
    T = T[:-1, :]
    T = np.concatenate((T, [t1]))
    # T_i = inv(T)
    print(T)
    T = (1/delta_x_squared) * T
    l, u = eig(T)
    sorted_indexes = l.argsort()[::-1]
    # res = np.flip(sorted(zip(l, u)))
    #bc = np.array([-alpha] + [0]*(N-2) + [-beta])
    return l, u , sorted_indexes

def v(j, x):
    return np.sqrt(2)*np.sin((2*j-1)*np.pi*x/(2))

def lam(j):
    return -(2*j-1)**2*np.pi**2/(4)

N = 100
l, u, idx = SL(N, 1, 0, 0)
x = np.linspace(0, 1, N)
# for i in [0,2]:
#     plt.plot(x, v(i, x))

print(l[idx[0]],lam(0), lam(1), lam(2), l[idx[1]])
plt.plot(x, -4.5*u[:,idx[0]])
plt.plot(x, -4.5*u[:,idx[1]])
plt.plot(x, -4.5*u[:,idx[2]])
plt.show()
# (-2.5941534926321834+0j) -2.4674011002723395 -2.4674011002723395 -22.206609902451056 (-23.246546059158415+0j)
# (-2.7324055946641863+0j) -2.4674011002723395 -2.4674011002723395 -22.206609902451056 (-24.47978724853579+0j)
# errors = []
# errors2 = []
# errors3 = []
# for N in range(10, 100):
#     # print(res[0][0])
#     res = SL(N , 1, 0, 0)
#     # plt.plot(np.linspace(0, 1, 10), res[1][0], "2")
#     def l_eig(j, L=1):
#         return -((2*j-1)**2)*(np.pi**2)/(4*L**2)
#     # plt.show()
#     errors.append(np.sqrt(1/(N+1))*linalg.norm(l_eig(1)-res[0][1]))
#     errors2.append(np.sqrt(1/(N+1))*linalg.norm(l_eig(2)-res[1][1]))
#     errors3.append(np.sqrt(1/(N+1))*linalg.norm(l_eig(3)-res[2][1]))
#     # print("First three", res[0:3, 1])

#     lambdaexact = [l_eig(1), l_eig(2), l_eig(3)]
#     # print(lambdaexact)
# plt.loglog([x for x in range(10, 100)], errors, "2")
# plt.loglog([x for x in range(10, 100)], errors2, "3")
# plt.loglog([x for x in range(10, 100)], errors3, "4")
# plt.loglog([x for x in range(10, 100)], [(x)**-2 for x in range(10, 100)])
# plt.loglog([x for x in range(10, 100)], [(x)**-3 for x in range(10, 100)])
# plt.show()