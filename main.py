from twopBVP import twopBVP
from schrödinger import SE
import numpy as np
import numpy.linalg as linalg
import math
import matplotlib.pyplot as plt


def test_error_sin():
    L = math.pi
    
    error = []
    n_range = 2**np.arange(2, 14)
    for N in n_range:
        x = np.linspace(0, L, N+2)
        delta_x = L/(N+1)
        y_2 = -np.sin(x)
        alpha = 0
        beta = 0
        res = [alpha]+list(twopBVP(y_2[1:-1], 0, 0, L, N))+[beta]
        error += [np.sqrt(delta_x)*linalg.norm(np.sin(x) - res)]
        
    plt.loglog([L/(N+1) for N in n_range], error, label="$||\ell||_{rms}$")
    plt.loglog([L/(N+1) for N in n_range], [(L/(N+1))**2 for N in n_range], label="$\mathcal{O}(\Delta x^2)$")
    plt.ylabel("RMS error")
    plt.xlabel("$\Delta x$")
    plt.legend()
    plt.show()

def SE0():
    V = 0
    N = 99

    energies, wave, idx = SE(V, N)
    c = 1000
    pd = c*wave**2+np.abs(energies)

    for level in range(7):
        plt.plot(np.linspace(0, 1, N+2), pd[:,idx[level]])
    plt.show()

def SEV():
    N = 99
    x = np.linspace(0+1/(N+1), 1-1/(N+1), N)
    # V = 700*(0.5-np.abs(x-0.5))
    # V = 800*np.sin(np.pi*x)**2
    # V = 0*x
    V = 800*np.sin(np.pi*2*x)**2       #triplet

    energies, wave, idx = SE(V, N)
    c = 1200
    d = 1
    pd = c*wave**2+np.abs(energies)
    wavelevel = c*wave + 10*np.abs(energies)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,8))
    for level in range(7):
        print(np.abs(energies[idx[level]]))
        ax1.plot(np.linspace(0, 1, N+2), wavelevel[:,idx[level]], label = f"$\psi_{level+1}$")
        ax2.plot(np.linspace(0, 1, N+2), pd[:,idx[level]], label = f"$|\psi_{level+1}|^2$")
        ax2.axhline(y=pd[:,idx[level]][0], color="gray", linestyle="dotted")
    ax1.plot(np.linspace(0, 1, N+2), np.hstack(([0], 10*d*V, [0])), linestyle="dashed")
    ax1.axvline(x=0, linestyle="solid") 
    ax1.axvline(x=1, linestyle="solid")
    ax2.plot(np.linspace(0, 1, N+2), np.hstack(([0], d*V, [0])), linestyle="dashed")
    ax2.axvline(x=0, linestyle="solid")
    ax2.axvline(x=1, linestyle="solid")

    ax1.legend(loc='lower right', bbox_to_anchor=(0, 0), ncol=1)
    ax2.legend(loc='lower left', bbox_to_anchor=(1, 0),ncol=1)
    ax1.axes.get_yaxis().set_ticks([])
    ax2.axes.get_yaxis().set_ticks([])
    ax1.set_ylabel("Energy")
    ax2.set_ylabel("Energy")
    ax1.set_xlabel("x")
    ax2.set_xlabel("x")
    plt.show()


if __name__ == "__main__":
    SEV()