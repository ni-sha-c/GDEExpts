import torch

def lorenz_periodic(t, u):
    """ For 4 exception values of rho, lorenz can be periodic
        For current implementation, we use simplest orbit out of 4, rho = 350
        
        Reference: Colin Sparrow, The Lorenz Equations: Bifurcations, Chaos, and Strange Attractors, Springer, 1982"""

    sigma = 10.0
    rho = 350.0
    beta = 8/3

    return torch.stack([
        sigma * (u[1] - u[0]),
        u[0] * (rho - u[2]) - u[1],
        (u[0] * u[1]) - (beta * u[2])
    ])