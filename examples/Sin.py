import torch
import numpy as np

def sin(t, x):
    """ [x,t] => [velocity, acceleration] """
    dxdt = np.cos(t)
    dydt = -np.sin(t)
    return torch.tensor([dxdt, dydt])