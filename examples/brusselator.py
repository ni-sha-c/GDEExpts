import torch

def brusselator(t, X):
    """Return the derivatives, dx/dt and dy/dt."""
    x, y = X
    a = 1
    b = 2.02
    dxdt = a - (1+b)*x + x**2 * y
    dydt = b*x - x**2 * y
    return torch.tensor([dxdt, dydt])