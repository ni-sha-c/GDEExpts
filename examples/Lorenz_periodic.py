import torch

def lorenz(t, u):
    """ For 4 exception values of rho, lorenz can be periodic
        For current implementation, we use simplest orbit out of 4, rho = 350
        
        Reference: Colin Sparrow, The Lorenz Equations: Bifurcations, Chaos, and Strange Attractors, Springer, 1982"""

    sigma = torch.Tensor([10.0])
    rho = torch.Tensor([350.0])
    beta = torch.Tensor([8/3])

    return torch.Tensor([
        sigma * (u[1] - u[0]),
        u[0] * (rho - u[2]) - u[1],
        (u[0] * u[1]) - (beta * u[2])
    ])