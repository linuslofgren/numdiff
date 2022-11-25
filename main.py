from twopBVP import twopBVP
from schrödinger import SE
import numpy as np
import numpy.linalg as linalg
import math
import matplotlib.pyplot as plt


def test_error_sin():
    L = math.pi
    
    error = []
    for N in range(10, 100):
        x = np.linspace(0, L, N+2)
        delta_x = L/(N+1)
        y_2 = -np.sin(x)
        alpha = 0
        beta = 0
        res = [alpha]+list(twopBVP(y_2[1:-1], 0, 0, L, N))+[beta]
        error += [np.sqrt(delta_x)*linalg.norm(np.sin(x) - res)]
        
    plt.loglog([L/(N+1) for N in range(10, 100)], error)
    plt.loglog([L/(N+1) for N in range(10, 100)], [(L/(N+1))**2 for N in range(10, 100)])
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
   #V = 700*(0.5-np.abs(x-0.5))
    V = 800*np.sin(np.pi*x)**2
    #V = 800*np.sin(np.pi*2*x)**2       #triplet

    energies, wave, idx = SE(V, N)
    c = 1000
    d = 1
    pd = c*wave**2+np.abs(energies)
    wavelevel = c*wave + np.abs(energies)

    for level in range(7):
        plt.plot(np.linspace(0, 1, N+2), pd[:,idx[level]], label = f"$E_{level+1}$")
    #plt.plot(np.linspace(0, 1, N+2), np.hstack(([0], d*V, [0])))
    plt.legend()
    plt.show()


if __name__ == "__main__":
    SEV()