from scipy.linalg import toeplitz
from scipy.linalg import eig
import numpy as np

def SE(V, N):
    delta_x = 1/(N+1)

    delta_x_squared = delta_x ** 2
    
    Tsd = (1/delta_x_squared) * toeplitz([-2, 1]+[0]*(N-2), [-2, 1] + [0]*(N-2))    #gör en rad kortare och ändra där nere också
    
    T = Tsd-np.eye(N)*V

    energies, wave = eig(T)
    
    bc = [0]*N

    wave = np.vstack((bc, wave, bc))

    sorted_indexes = energies.argsort()[::-1]
    
    return energies, wave , sorted_indexes