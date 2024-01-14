import torch

def coupled_brusselator(t, X):
    ''' https://www.sciencedirect.com/science/article/pii/S0960077923001418 '''
    x1, y1, x2, y2 = X
    a = 2.0
    b = 6.375300171526684
    lambda_1 = 1.2
    lambda_2 = 80.0
    dxdt = a - (1+b)*x1 + x1**2 * y1 + lambda_1*(x2 - x1)
    dydt = b*x1 - x1**2 * y1 + lambda_2*(y2 - y1)
    dx2dt = a - (1+b)*x2 + x2**2 * y2 + lambda_1*(x1 - x2)
    dy2dt = b*x2 - x2**2 * y2 + lambda_2*(y1 - y2)

    return torch.stack([dxdt, dydt, dx2dt, dy2dt])