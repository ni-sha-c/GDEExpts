import numpy as np
import torch
from numba import *
import time

import sys
sys.path.append('..')
from src.NODE_solve import *

# True Models 
from examples.Brusselator import *
from examples.Lorenz import *
from examples.Lorenz_periodic import *
from examples.Sin import *
from examples.Tent_map import *
import torchdiffeq


@jit(parallel=True)
def kernel_lorenz(x0, N, lorenz_func):

    sigma = 10.0  # Sigma parameter
    rho = 28.0  # Rho parameter
    beta = 8.0 / 3.0  # Beta parameter

    dt = 0.01
    T = np.arange(0, 100, dt)
    len_T = T.shape[0]

    # initialize N traj
    traj = torch.zeros((N, len_T, 3)) # N initial point x number of time steps x dim of lorenz
    traj[:, 0, :] = torch.tensor(x0)

    for n in range(N):
        for i in range(1, len_T):
            u = traj[n, i-1, :]
            du = lorenz_func(0., u) #.reshape(-1, 3)
            traj[n, i, :] = traj[n, i-1, :] + dt * du

            # dx = sigma * (traj[:, i-1, 1] - traj[:, i-1, 0])
            # dy = (traj[:, i-1, 0] * (rho - traj[:, i-1, 2]) - traj[:, i-1, 1])
            # dz = (traj[:, i-1, 0] * traj[:, i-1, 1] - beta * traj[:, i-1, 2])
            # traj[:, i, 0] = traj[:, i-1, 0] + dt * dx
            # traj[:, i, 1] = traj[:, i-1, 1] + dt * dy
            # traj[:, i, 2] = traj[:, i-1, 2] + dt * dz

    return traj




device = torch.device('cuda')
N = 5  # Number of initial points

x0 = np.random.rand(N, 3)

start = time.time()
x = kernel_lorenz(x0, N, lorenz)
end = time.time()
print("Elapsed time = %s" % (end - start))
print(x.shape)

# res = simulate(lorenz, 0., 100., x0[0, :], 0.01)
res = torchdiffeq.odeint(lorenz, torch.tensor(x0[0, :]), torch.arange(0, 100, 0.01), method='euler', rtol=1e-8)
print(x[0, :20], x[0, -20:])
print(res[0:20], res[-20:])



