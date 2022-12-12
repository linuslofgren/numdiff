from scipy import sparse
from scipy.linalg import toeplitz
from scipy.linalg import solve
import numpy as np
import matplotlib.pyplot as plt

def euler_step(Tdx, u_old, dt):
    return u_old + dt * Tdx @ u_old

N = 100
M = 100
L = 1
t_end = 10
delta_t = 1/(M)
alpha = 0
beta = 0

u = np.sin(np.linspace(0, L, N))
plt.plot(u)
plt.show()

delta_x = L/(N+1)
delta_x_squared = delta_x ** 2
T_times_delta_x = toeplitz([-2, 1]+[0]*(N-2), [-2, 1] + [0]*(N-2))
T = T_times_delta_x/delta_x_squared

us = [u]
for i in range(M):
    us.append(euler_step(T, us[-1], delta_t))

print(us[0])

plt.show()

# bc = [-alpha] + [0]*(N-2) + [-beta]
# Y = spsolve(T_times_delta_x, (f_vec*delta_x_squared + bc))
