import torch

def brusselator(t, X):
    """ func: Return the derivatives, dx/dt and dy/dt.
        a, b were decided so that it generates periodic orbits 
        (B > A^2 + 1) """
    x, y = X
    a = 1
    b = 2.02
    dxdt = a - (1+b)*x + x**2 * y
    dydt = b*x - x**2 * y
    return torch.stack([dxdt, dydt])