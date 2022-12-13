from scipy.linalg import toeplitz
from scipy.linalg import solve
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def laxwen(u, amu):
    A = toeplitz(
        [1-amu**2, amu/2*(amu+1)]+([0]*(N-3)) + [amu/2*(amu-1)],
        [1-amu**2, amu/2*(amu-1)]+([0]*(N-3)) + [amu/2*(amu+1)],
    )
    return A @ np.transpose(u)

N = 100 # Violation
M = 71 # Violation
# N = 100
# M = 1000
delta_x = 1/N
t_end = 1
delta_t = t_end/M
a = 0.8
mu = delta_t/delta_x
amu = a*mu 
print(amu)

x = np.linspace(0, 1, N)
u = np.exp(-100*(x-0.5)**2)

us = [u]

for i in range(M):
    n = list(laxwen(us[-1], amu))
    us.append(n)

xx = np.linspace(0,1,N)
tt = np.linspace(0,t_end,np.abs(M+1))
X, Y = np.meshgrid(xx, tt)
z = np.array(us)
z = np.resize(z, (np.abs(M+1), N))

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
plt.show()
