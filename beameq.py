from twopBVP import twopBVP
import numpy as np
import matplotlib.pyplot as plt

L = 10 #m
N = 999
x = np.linspace(0, L, N+2)[1:-1] #m
E = 1.9e11 #N/m^2
I = 1e-3*(3-2*np.cos(np.pi*x/L)**12)
q = -50e3 #N/m
alpha = 0 
beta = 0

print(x.size)
M = twopBVP(np.ones(len(x))*q, alpha, beta, L, N)
u = [alpha] + list(twopBVP(M/(E*I), alpha, beta, L, N)) + [beta]
print(u[500]) #obs fett fel, kanske -0.011741059085880013 i oklar enhet
plt.plot(np.hstack((0, x, L)), u)
plt.ylabel("Deflection [m]")
plt.xlabel("x [m]")
plt.show()
