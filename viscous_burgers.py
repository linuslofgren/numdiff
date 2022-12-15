import numpy as np
from diffusion import tr_step
from scipy.linalg import toeplitz
from numpy.linalg import solve
import matplotlib.pyplot as plt
from matplotlib import cm
from matrix_representation import second_derivative, first_derivative

# def n_plus_one_old(dt, dx, u_j, u_j_plus, u_j_minus):
#     return (
#         u_j
#         - dt/(2*dx)*u_j*(u_j_plus-u_j_minus)
#         + dt**2/(4*dx**2) * u_j * (u_j_plus-u_j_minus)**2
#         + dt**2/(2*dx**2) * u_j**2 * (u_j_plus- 2*u_j + u_j_minus)
#     )

def LW(u, dx, dt, T, S):
    u_new = u - u * (dt * S @ u)+ u * dt**2 / 2 * (2 * (S @ u)**2 + u**2 * T @ u)
    return u_new


def step(u, d, dx, dt, T, S):
    N = len(u)
    u_prime = LW(u, dx, dt, T, S)
    lhs = np.eye(N) - d*dt/2 * T
    rhs = u_prime + (d * dt/2) * T @ u
    return solve(lhs, rhs)

def step_with(d, delta_x, delta_t, T, S):
    def take_step(u):
        return step(u, d, delta_x, delta_t, T, S)
    return take_step

def vis_burg(d, N, M):
    # d = 0.01

    L = 1
    # N = 100
    delta_x = L/N
    x = np.linspace(0, L, N)
    
    t_end = 5
    # M = 1000
    delta_t = t_end/(M+1)
    t = np.linspace(0,t_end,M+1)

    T = second_derivative(delta_x, N)
    S = first_derivative(delta_x, N)

    step = step_with(d, delta_x, delta_t, T, S)

    u = 2*np.exp(-250*(x-0.5)**2)
    us = [u]

    for _ in range(M):
        n = step(us[-1])
        us.append(n)

    
    us = np.array(us)
    us[:,-1] = us[:,0]

    X, Y = np.meshgrid(x, t)
    z = np.array(us)
    z = np.resize(z, (M+1, N))

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    surf = ax.plot_surface(X, Y, z, cmap=cm.plasma,
                        linewidth=0, antialiased=True)
    plt.show()

vis_burg(d=0.001, N=80, M=10000)