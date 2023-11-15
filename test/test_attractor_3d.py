import numpy as np
from scipy import integrate
import seaborn as sns

import torch
import numba
from numba import prange
import time
import torchdiffeq

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import cnames
from matplotlib import animation
import itertools
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
N_trajectories = 1500  # Number of initial points
np.random.seed(42)
bound_attractor = 50.
# Adding np.array([0, 0, -30]) so that attractor lies in the center of cube
x0 = bound_attractor * np.random.uniform(-1.0, 1.0, (N_trajectories,3)) #15 np.array([-25, 25,-25])+
#x0 = np.random.rand(N_trajectories, 3)

dt = 0.01
T = np.arange(0, 100, dt)
len_T = T.shape[0]

# initialize N traj
traj = np.zeros((N_trajectories, len_T, 3))  # N trajectories x number of time steps x dimensions of Lorenz
traj[:, 0, :] = x0

start = time.time()
compute_trajectory(traj, N_trajectories, len_T, dt)
end = time.time()

print("Elapsed time = %s" % (end - start))



# Current code is inspired by: https://jakevdp.github.io/blog/2013/02/16/animating-the-lorentz-system-in-3d/

'''N_trajectories = 3000

# Choose random starting points, uniformly distributed from -30 to 30
np.random.seed(42)
bound_attractor = 40.
# Adding np.array([0, 0, -30]) so that attractor lies in the center of cube
x0 = bound_attractor * np.random.uniform(-1.0, 1.0, (N_trajectories,3)) #15 np.array([-25, 25,-25])+
print(x0)

# Load the saved model
device = "cuda" if torch.cuda.is_available() else "cpu"
time_step = 0.01
x0 = torch.tensor(x0).to(device).double()
model = ODE_Lorenz().to(device).double()
# Pick the model!
model_path = "../test_result/expt_lorenz/AdamW/"+str(time_step)+'/'+'model_MSE_0.pt'
model.load_state_dict(torch.load(model_path))
model.eval()
print("Finished Loading model")

x_t = np.asarray([simulate(model, 0., 15., x0i.double(), 0.01).detach().to('cpu') for x0i in x0])
print("Trajectory all computed!")'''
x_t = traj

# Set up figure & 3D axis for animation
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1], projection='3d')
ax.set_aspect("equal")
ax.set_facecolor('black')
ax.axis('off')

# Set Cube
r = [-55., 55.]
for s, e in itertools.combinations(np.array(list(itertools.product(r, r, r))), 2):
    if np.sum(np.abs(s-e)) == r[1]-r[0]:
        ax.plot3D(*zip(s, e), color="white", linewidth=0.5, alpha=0.5)

# choose a different color for each trajectory
colors = plt.cm.hsv(np.linspace(0, 1, N_trajectories))
# gist_rainbow

# set up lines and points
lines = [ax.plot([], [], [], '-', c=c, alpha=1., linewidth=2)[0]
for c in colors]
pts = [ax.plot([], [], [], 'o', c=c, alpha=0.95, markersize=2)[0]
for c in colors]


# prepare the axes limits
ax.set_xlim((-50, 50)) #ax.set_xlim((-30, 20))
ax.set_ylim((-50, 50))
ax.set_zlim((-50, 50)) #ax.set_zlim((0, 40))

# set point-of-view: specified by (altitude degrees, azimuth degrees)
ax.view_init(20, -60) # we turned the cube turned to right side and then lifted it up slightly so we can slightly see bottom side. #-20

# initialization function: plot the background of each frame
def init():
    for line, pt in zip(lines, pts):
        line.set_data([], [])
        line.set_3d_properties([])

        pt.set_data([], [])
        pt.set_3d_properties([])
    return lines + pts

# animation function.  This will be called sequentially with the frame number
def animate(i):
    # we'll step two time-steps per frame.  This leads to nice results.
    i = (2 * i) % x_t.shape[1]

    for line, pt, xi in zip(lines, pts, x_t):
        x, y, z = xi[:i].T

        if i > 10 and i < 25:
            x, y, z = xi[i-4:i].T #5
            x_point, y_point, z_point = xi[0:1].T
        elif i >= 25 and i < 50:
            x, y, z = xi[i-10:i].T
        elif i >= 50:
            x, y, z = xi[i-20:i].T
            
        line.set_data(x, y)
        line.set_3d_properties(z)

        # pt.set_data(x[-1:], y[-1:])
        # pt.set_3d_properties(z[-1:])
        if i < 200:

            rate = 1 - i / 200
            x_point, y_point, z_point = xi[0:1].T
            pt.set_data(x_point, y_point)
            pt.set_3d_properties(z_point)
            pt.set_alpha(rate)
        else:
            pt.set_data(x[-1:], y[-1:])
            pt.set_3d_properties(z[-1:])
    
    fig.canvas.draw()
    return lines + pts





# instantiate the animator.
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=1000, interval=20, blit=True)
anim.save('animation.gif', writer='PillowWriter', fps=500)

# Save as mp4. This requires mplayer or ffmpeg to be installed
# anim.save('lorentz_attractor.mp4', fps=15) #extra_args=['-vcodec', 'libx264']

plt.show()


