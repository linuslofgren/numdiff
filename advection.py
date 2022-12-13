from scipy.linalg import toeplitz
from scipy.linalg import solve
import numpy.linalg as linalg
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def laxwen(u, amu):
    N = len(u)
    A = toeplitz(
        [1-amu**2, amu/2*(amu+1)]+([0]*(N-3)) + [amu/2*(amu-1)],
        [1-amu**2, amu/2*(amu-1)]+([0]*(N-3)) + [amu/2*(amu+1)],
    )
    return A @ np.transpose(u)

def laxwenapprox(N, M, a, t_end=5):
    delta_x = 1/N
    delta_t = t_end/M
    mu = delta_t/delta_x
    amu = a*mu  #violation om >1.1
    print(f'amu: {amu}')

    x = np.linspace(0, 1, N)
    u = np.exp(-100*(x-0.5)**2)
    us = [u]

    for i in range(M):
        n = list(laxwen(us[-1], amu))
        us.append(n)

    tt = np.linspace(0,t_end,np.abs(M+1))
    norm = [linalg.norm(u) for u in us]
    plt.plot(tt, norm)

    xx = np.linspace(0,1,N)
    X, Y = np.meshgrid(xx, tt)
    z = np.array(us)
    z = np.resize(z, (np.abs(M+1), N))

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
    plt.show()

#violation 
laxwenapprox(200, 1000, 1.008)

#amu = 1
laxwenapprox(200, 1000, 1)

#amu = 0.9
laxwenapprox(180, 1000, 1)
