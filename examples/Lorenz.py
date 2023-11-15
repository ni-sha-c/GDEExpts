import torch
from numpy import *

def lorenz(t, u, rho=28.0):
    """ Lorenz chaotic differential equation: du/dt = f(t, u)
    t: time T to evaluate system
    u: state vector [x, y, z] 
    return: new state vector in shape of [3]"""

    sigma = 10.0
    #rho = 28.0
    beta = 8/3

    if u.ndim == 1: 
        du = torch.stack([
            sigma * (u[1] - u[0]),
            u[0] * (rho - u[2]) - u[1],
            (u[0] * u[1]) - (beta * u[2])
        ])
    else:
        print("multi", u.shape)
        du = torch.stack([
            sigma * (u[:, 1] - u[:, 0]),
            u[:, 0] * (rho - u[:, 2]) - u[:, 1],
            (u[:, 0] * u[:, 1]) - (beta * u[:, 2])
        ])

    return du

def lorenz_jac(x):
    '''lorenz for creating jacobian matrix.
        x: torch.randn(1,3)
        call F.jacobian(lorenz, x)'''
    sigma = 10
    rho = 28
    beta = 8/3
    dx = torch.zeros(3)
    dx[0] = sigma*(x[0][1] - x[0][0])
    dx[1] = x[0][0]*(rho - x[0][2]) - x[0][1]
    dx[2] = x[0][0]*x[0][1] - beta*x[0][2]
    return dx