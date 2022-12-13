from scipy.linalg import toeplitz
from scipy.linalg import solve
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# def euler_step(Tdx, u_old, dt):
#     return u_old + dt * (Tdx @ u_old)

def tr_step(Tdx, u_old, dt):
    lhs = np.eye(Tdx.shape[0]) - dt/2 * Tdx
    rhs = (np.eye(Tdx.shape[0]) + dt/2 * Tdx) @ u_old
    return solve(lhs, rhs)

if __name__ == "__main__":
    N = 41
    L = 1
    u = 10*(1-np.abs(2*(np.linspace(0, L, N)-1/2)))
    # u = np.sin(np.linspace(0, L, N))

    delta_x = L/(N+1)

    u_max = np.max(u)
    t_end = 1
    # M = int(1e3*1.406) # Small violation
    # M = 100
    M = int(u_max/(delta_x)**2/t_end)
    delta_t = t_end/(M)
    alpha = 0
    beta = 0

    print(f"Courant number: {u_max*delta_t/delta_x**2}, delta x:{delta_x:.4f} delta t: {delta_t:.4f}")

    delta_x_squared = delta_x ** 2
    T_times_delta_x = toeplitz([-2, 1]+[0]*(N-2), [-2, 1] + [0]*(N-2))
    T = T_times_delta_x/delta_x_squared

    us = [u]

    for i in range(M):
        n = [alpha]+list(tr_step(T, us[-1], delta_t))[1:-1]+[beta]
        us.append(n)

    xx = np.linspace(0,1,N)
    tt = np.linspace(0,t_end,M+1)
    X, Y = np.meshgrid(xx, tt)
    z = np.array(us)
    z = np.resize(z, (M+1, N))

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
    plt.show()
