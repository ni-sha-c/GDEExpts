import torch

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