from scipy import sparse
from scipy.linalg import toeplitz
from scipy.linalg import solve
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

def euler_step(Tdx, u_old, dt):
    return u_old + dt * (Tdx @ u_old)

N = 41

L = 1
u = 10*(1-np.abs(2*(np.linspace(0, L, N)-1/2)))

u_max = np.max(u)
print(u_max)
delta_x = L/(N+1)

M = 10*int(u_max/delta_x)
t_end = 1
delta_t = 1/(M)
alpha = 0
beta = 0

print(f"Courant number: {u_max*delta_t/delta_x}")



# print(delta_x)
delta_x_squared = delta_x ** 2
# print(delta_x_squared)
T_times_delta_x = toeplitz([-2, 1]+[0]*(N-2), [-2, 1] + [0]*(N-2))
# print(T_times_delta_x)
T = T_times_delta_x/delta_x_squared
# print(T)

us = [u]
# plt.plot(us[0])



for i in range(M):
    n = [alpha]+list(euler_step(T, us[-1], delta_t))[1:-1]+[beta]
    us.append(n)
    # if i % 10 == 0:
        # plt.plot(us[-1])

xx = np.linspace(0,1,N)
tt = np.linspace(0,t_end,M+1)
X, Y = np.meshgrid(xx, tt)
z = np.array(us)
print(z.shape)
z = np.resize(z, (M+1, N))
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
plt.show()

# bc = [-alpha] + [0]*(N-2) + [-beta]
# Y = spsolve(T_times_delta_x, (f_vec*delta_x_squared + bc))
