from scipy.linalg import toeplitz
from scipy.linalg import solve
import numpy.linalg as linalg
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from diffusion import tr_step

d = 0.1
a = 0
N = max(int(abs(a/(2*d))), 100)
dx = 1/N
print(dx)
print(f'Pe: {abs(a/d)}, meshPe: {abs(a/d*dx)}')
M = 50
dt = 1/M
t_end = 1

def convdif(u, a, d, dt, dx):
    N = len(u)

    T = 1/dx**2*toeplitz([-2, 1]+[0]*(N-2), [-2, 1] + [0]*(N-2))
    T[0][-1]=T[1][0]
    T[-1][0]=T[-2][-1]

    S = 1/(2*dx)*toeplitz([-1]+[0]*(N-1), [-1, 0, 1] + [0]*(N-3))[:-2,]
    row1 = np.roll(S[0,:], -1)
    rowN = np.roll(S[-1,:], 1)
    S = np.vstack((row1, S, rowN))

    Tdx = d*T-a*S
    u_new = tr_step(Tdx, u, dt)
    return  u_new

x = np.linspace(0, 1, N)
u = np.exp(-100*(x-0.5)**2)
us = [u]

for i in range(M):
    n = list(convdif(us[-1], a, d, dt, dx))
    us.append(n)

tt = np.linspace(0,t_end,np.abs(M+1))
norm = [linalg.norm(u) for u in us]
plt.plot(tt, norm)

xx = np.linspace(0,1,N)
X, Y = np.meshgrid(xx, tt)
z = np.array(us)
z = np.resize(z, (np.abs(M+1), N))

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, z, cmap=cm.viridis,
                    linewidth=0, antialiased=False)
plt.show()