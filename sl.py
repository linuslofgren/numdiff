from twopBVP import twopBVP
from scipy import sparse
from scipy.linalg import toeplitz, inv, norm
from scipy.sparse.linalg import eigs, eigsh
from scipy.linalg import eig, eigh_tridiagonal
import scipy.linalg as linalg
import numpy as np
import time
import matplotlib.pyplot as plt

def SL(N, L, alpha, beta, approach=1):
    # # Approach 1
    # delta_x = L/(N)
    # btm_row = np.array((N-2)*[0]+[2, -2])

    # Approach 3
    # delta_x = L/(N+1)
    # t1 = np.array((N-2)*[0]+[2/3, -2/3])

    # Approach 2
    delta_x = L/(N+1/2)
    btm_row = (N-2)*[0]+[1, -1]
    u_1 = lambda u: u[-1,:]+beta*delta_x/2

    delta_x_squared = delta_x ** 2
    
    # T = sparse.csc_matrix(
    #     np.vstack((
    #         toeplitz([-2, 1]+[0]*(N-2))[:-1],
    #         btm_row
    #     ))
    # ).asfptype()

    nbr_eig = np.min((N-1, 3))

    # Use shift invert mode (by setting sigma) to improve performance https://docs.scipy.org/doc/scipy/tutorial/arpack.html
    # Search for eigenvalues of largest magnitude close to sigma (SM -> LM in the inverted problem )
    # l, u = eigsh(T, nbr_eig, which="LM", sigma=0)#, mode="normal", tol=1e-5, v0=np.ones(N))
    # sorted_indexes = l.argsort()[::-1]
    # l, u = (1/delta_x_squared) * l, (1/delta_x_squared) * u
    # l, u = eig(T.todense())
    # sorted_indexes = l.argsort()[::-1]
    # l, u = (1/delta_x_squared) * l, (1/delta_x_squared) * u

    l, u = eigh_tridiagonal(
        np.hstack((2*np.ones(N-1), 1)),
        -1*np.ones(N-1),
        select="i",
        select_range=(0,nbr_eig)
    )
    sorted_indexes = l.argsort()#[::-1]
    l, u = -1*(1/delta_x_squared) * l, -1*(1/delta_x_squared) * u
    
    # Approach 2
    u = np.vstack(([alpha]*u.shape[1], u))
    u = np.vstack((u, u_1(u)))

    # Normalize
    for i in range(u.shape[1]):
        u[:,i] /= norm(u[:,i])
    # Approach 3
    # u = np.vstack((u, 1/3*(2*beta+4*u[-1,:]-u[-2,:])))
    
    return l, u , sorted_indexes

def v(j, x):
    return np.sqrt(2)*np.sin((2*j-1)*np.pi*x/(2))

def lam(j):
    return -(2*j-1)**2*np.pi**2/(4)

N = 499
l, u, idx = SL(N, 1, 0, 0)
x = np.linspace(0, 1, u.shape[0])
print("First three eigenvalues: \n", l[idx[0]], l[idx[1]], l[idx[2]]) # (-2.4673990672394255+0j) (-22.206445196618926+0j) (-61.68375662939049+0j)
for i in range(0,3):
    plt.plot(x, -4.5*u[:,idx[i]], label=f"$u_{i+1}$")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.show()

def l_eig(j, L=1):
    return -((2*j-1)**2)*(np.pi**2)/(4*L**2)
# raise Exception("-")
errors = []
errors2 = []
errors3 = []
n_range = 2**np.arange(2, 14, dtype=float)
for N in n_range:
    l, u, idx = SL(int(N) , 1, 0, 0)
    errors.append(np.abs(l_eig(1)-l[idx[0]]))
    errors2.append(np.abs(l_eig(2)-l[idx[1]]))
    errors3.append(np.abs(l_eig(3)-l[idx[2]]))
    lambdaexact = [l_eig(1), l_eig(2), l_eig(3)]
plt.loglog([x for x in n_range], errors, "1", label="$\lambda_{\Delta x 1}-\lambda_1$")
plt.loglog([x for x in n_range], errors, "2", label="$\lambda_{\Delta x 2}-\lambda_2$")
plt.loglog([x for x in n_range], errors, "3", label="$\lambda_{\Delta x 3}-\lambda_3$")
plt.loglog([x for x in n_range], [(x)**-2 for x in n_range], label="$\mathcal{O}(\Delta x^2)$")
plt.legend()
# plt.xlabel(f"N ({','.join(map(str, n_range.astype(int)))})")
plt.xlabel("N")
plt.ylabel("Eigenvalue error")
plt.xscale('log', base=2)
plt.show()