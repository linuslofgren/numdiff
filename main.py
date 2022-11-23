from twopBVP import twopBVP
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

if __name__ == "__main__":
    test_error_sin()