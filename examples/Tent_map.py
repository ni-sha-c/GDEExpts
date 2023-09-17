import torch


def tent_map(x):
    ''' 1D Discrete Dynamical System '''
    
    if 0 <= x < 1:
        return torch.tensor(2 * x, requires_grad=True)
    elif 1 <= x <= 2:
        return torch.tensor(4 - 2 * x, requires_grad=True)
