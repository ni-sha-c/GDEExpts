import torch
import numba
from numba import prange
import numpy as np
import time
import torchdiffeq


import sys
sys.path.append('..')
from src.NODE_solve import *

# True Models 
from examples.Brusselator import *
from examples.Lorenz import *
from examples.Lorenz_periodic import *
from examples.Sin import *
from examples.Tent_map import *

@numba.jit(nopython=True)
def lorenz_func(t, u):
    # Assuming u is a 1D Numpy array or a Numba-compatible type
    x, y, z = u[0], u[1], u[2]

    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0

    du = np.zeros(3)  # Create a Numba-compatible array for the derivative

    du[0] = sigma * (y - x)
    du[1] = x * (rho - z) - y
    du[2] = x * y - beta * z

    return du



@numba.jit(parallel=True)
def compute_trajectory(traj, N, len_T, dt):
    for n in prange(N):  # Use prange for parallel looping
        for i in range(1, len_T):
            u = traj[n, i - 1, :]
            du = lorenz_func(0., u)
            traj[n, i, :] = traj[n, i - 1, :] + dt * du
            # traj[n, i, :] = torchdiffeq.odeint(lorenz, torch.tensor(u), torch.linspace(0, 0.01, 2), method='euler', rtol=1e-8)[-1].numpy()

# Call the parallelized function
device = torch.device('cuda')
N = 5  # Number of initial points
x0 = np.random.rand(N, 3)

dt = 0.01
T = np.arange(0, 100, dt)
len_T = T.shape[0]

# initialize N traj
traj = np.zeros((N, len_T, 3))  # N trajectories x number of time steps x dimensions of Lorenz
traj[:, 0, :] = x0

start = time.time()
compute_trajectory(traj, N, len_T, dt)
end = time.time()

print("Elapsed time = %s" % (end - start))

res = torchdiffeq.odeint(lorenz, torch.tensor(x0[0, :]), torch.arange(0, 100, 0.01), method='euler', rtol=1e-8)
print(res.shape, traj.shape)
print(traj[0, :5], traj[0, 5000:1005])
print(res[0:5], res[1000:1005])

