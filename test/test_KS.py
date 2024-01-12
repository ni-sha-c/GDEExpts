import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
sys.path.append('../examples')
from KS import *
from torch import *

def test_KuramotoSivashinsky():
    '''
    state vector, u, has following index i

    i = 0, 1, 2, ..., n, n+1
    where i = 1, ..., n are internal nodes
          i = 0, n+1 are boundary nodes
          i = -1, n+2 are ghost nodes
    '''
    # tensor(0.5040)
    # tensor(0.5042)
    L = 128
    n = 511 # number of interior nodes= (L - 2)*10**n
    c = 0.4
    x = torch.linspace(0, L, n+2) # 0, 0.25, ... 128 # shape: [513]
    x = x[1:-1] # shape: [511]
    dx = L/(n+1) # 0.25

    u = sin(2*pi*x/L) # only the internal nodes

    up = cos(2*pi*x/L)*2*pi/L
    upup = -sin(2*pi*x/L)*(2*pi/L)**2
    upupup = -cos(2*pi*x/L)*(2*pi/L)**3
    upupupup = sin(2*pi*x/L)*(2*pi/L)**4
    # --- ana_rhs_KS: -(u + c)*up - upup - upupupup --- #
    # ana_rhs_KS = -(u+c)*up
    ana_rhs_KS = -(u+c)*up -upup - upupupup

    # --- num_rhs_KS: rhs_KS(u, c, dx) --- #
    num_rhs_KS = rhs_KS_implicit(u, dx) + rhs_KS_explicit(u, c, dx)
    # num_rhs_KS = rhs_KS_explicit(u, c, dx)
 
    print("answer", ana_rhs_KS[0:3], ana_rhs_KS[-3:])
    print("predicted", num_rhs_KS[0:3], num_rhs_KS[-3:])
    print(norm(ana_rhs_KS))
    print(norm(num_rhs_KS))
    assert np.allclose(ana_rhs_KS, num_rhs_KS, rtol=1e-5, atol=1e-5)



def KS_Simulate():
    # Solution of Kuramoto-Sivashinsky equation
    # u_t = -u*u_x - u_xx - u_xxxx, periodic boundary conditions on [0,32*pi]
    # computation is based on v = fft(u), so linear term is diagonal
    #
    # Using this program:
    # u is the initial condition
    # h is the time step
    # N is the number of points calculated along x
    # a is the max value in the initial condition
    # b is the min value in the initial condition
    # x is used when using a periodic boundary condition, to set up in terms of pi
    #


    # Initial condition and grid setup
    N = 1024 # 1024 # initial condition (same as s from markov github example)
    x = np.transpose(np.conj(np.arange(1, N+1))) / N # x is the grid => 0 to 1
    u = np.cos(x/16)*(1+np.sin(x/16)) # signal, period=16*2pi
    v = np.fft.fft(u)

    # scalars for ETDRK4 (Exponential Time Diff RK4)
    h = 0.25
    k = np.transpose(np.conj(np.concatenate((np.arange(0, N/2), np.array([0]), np.arange(-N/2+1, 0))))) / 16 # k is frequency
    L = k**2 - k**4 # coef of linear part
    E = np.exp(h*L)
    E_2 = np.exp(h*L/2)
    M = 16
    r = np.exp(1j*np.pi*(np.arange(1, M+1)-0.5) / M)
    LR = h*np.transpose(np.repeat([L], M, axis=0)) + np.repeat([r], N, axis=0) #linear part
    Q = h*np.real(np.mean((np.exp(LR/2)-1)/LR, axis=1))
    f1 = h*np.real(np.mean((-4-LR+np.exp(LR)*(4-3*LR+LR**2))/LR**3, axis=1))
    f2 = h*np.real(np.mean((2+LR+np.exp(LR)*(-2+LR))/LR**3, axis=1))
    f3 = h*np.real(np.mean((-4-3*LR-LR**2+np.exp(LR)*(4-LR))/LR**3, axis=1))

    # main loop
    # This whole for loop will be what we are going to differentiate
    uu = np.array([u])
    tt = 0
    tmax = 150
    nmax = round(tmax/h)
    nplt = int((tmax/100)/h) # 6
    g = -0.5j*k

    for n in range(1, nmax+1):
        t = n*h
        Nv = g*np.fft.fft(np.real(np.fft.ifft(v))**2)
        a = E_2*v + Q*Nv
        Na = g*np.fft.fft(np.real(np.fft.ifft(a))**2)
        b = E_2*v + Q*Na
        Nb = g*np.fft.fft(np.real(np.fft.ifft(b))**2)
        c = E_2*a + Q*(2*Nb-Nv)
        Nc = g*np.fft.fft(np.real(np.fft.ifft(c))**2)
        v = E*v + Nv*f1 + 2*(Na+Nb)*f2 + Nc*f3          # v = solution of ODE
        if n%nplt == 0:
            u = np.real(np.fft.ifft(v))
            uu = np.append(uu, np.array([u]), axis=0)
            tt = np.hstack((tt, t))
        

    return tt, uu, x

if __name__ == '__main__':
    test_KuramotoSivashinsky()

