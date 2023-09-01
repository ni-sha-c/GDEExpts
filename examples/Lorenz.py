import torch
from numpy import *
def lorenz(t, u):
    """ Lorenz chaotic differential equation: dy/dt = f(t, y)
    t: time tk to evaluate system
    y: state vector [x, y, z] """

    sigma = torch.Tensor([10.0])
    rho = torch.Tensor([28.0])
    beta = torch.Tensor([8/3])

    return torch.Tensor([
        sigma * (u[1] - u[0]),
        u[0] * (rho - u[2]) - u[1],
        (u[0] * u[1]) - (beta * u[2])
    ])

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