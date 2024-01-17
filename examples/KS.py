import torch
from torch import *
import torch.sparse as tosp



def rhs_KS_implicit(u, dx):
    n = u.shape[0]     # u contains boundary nodes i = 0, 1, 2, ... , n, n+1

    # ----- second derivative ----- #
    A = tosp.spdiags(torch.vstack((ones(n), -2*ones(n), ones(n)))/(dx*dx), torch.tensor([-1, 0, 1]), (n, n))
    A = A.to_dense()

    # ----- fourth derivative ----- #
    dx4 = dx*dx*dx*dx
    B = tosp.spdiags(torch.vstack((ones(n), -4*ones(n), 6*ones(n), -4*ones(n), ones(n)))/dx4, torch.tensor([-2, -1, 0, 1, 2]), (n-2, n-2))
    B = B.to_dense()

    # Create the pad 
    C = torch.zeros(n, n)
    C[1:n-1, 1:n-1] = B

    # Boundary Condition (i = 2, 3, ... , n-1)
    C[1, 1] = 7/dx4
    C[1, 2] = -4/dx4
    C[1, 3] = 1/dx4
    C[-2, -2] = 7/dx4
    C[-2, -3] = -4/dx4
    C[-2, -4] = 1/dx4

    #print("Second  last row of C", C[-2, -5:]*dx4)
    # A += C
    #print(torch.dot(C[-2, :], u))
    #implicit_dudt = -torch.matmul(A, u)
    #implicit_dudt -= torch.matmul(C, u)
    return -(A+C)


def rhs_KS_explicit_nl(u, c, dx):
    # u contains boundary nodes
    n = u.shape[0]

    B = tosp.spdiags(torch.vstack((ones(n), -ones(n)))/(2*dx), torch.tensor([1,-1]), (n, n))
    B = B.to_dense()

    exp_term = - torch.matmul(B, u*u)/2
    exp_term[0], exp_term[-1] = 0., 0. # du_0/dx = 0, du_n/dx = 0

    return exp_term

def rhs_KS_explicit_linear(u, c, dx):
    # u contains boundary nodes
    n = u.shape[0]

    B = tosp.spdiags(torch.vstack((ones(n), -ones(n)))/(2*dx), torch.tensor([1,-1]), (n, n))
    B = B.to_dense()
    # du_0/dx = 0, du_n/dx = 0

    exp_term = - torch.matmul(B*c, u)
 
    return exp_term

def explicit_rk(u, c, dx, dt):
    k1 = rhs_KS_explicit_nl(u, c, dx) + rhs_KS_explicit_linear(u, c, dx)
    k2 = rhs_KS_explicit_nl(u + dt/3*k1, c, dx) + rhs_KS_explicit_linear(u + dt/3*k1, c, dx)
    k3 = rhs_KS_explicit_nl(u + dt*k2, c, dx) + rhs_KS_explicit_linear(u + dt*k2, c, dx)
    k4 = rhs_KS_explicit_nl(u + dt*(0.75*k2 + 0.25*k3), c, dx) + rhs_KS_explicit_linear(u + dt*(0.75*k2 + 0.25*k3), c, dx)
    return dt*(3/4*k2 - 1/4*k3 + 1/2*k4)

def implicit_rk(u, c, dx, dt):
    n = u.shape[0]
    A = rhs_KS_implicit(u, dx)
    Au = torch.matmul(A, u)
    k2 = torch.linalg.solve(eye(n) - dt/3*A, Au)
    k3 = torch.linalg.solve(eye(n) - dt/2*A, Au + dt/2*matmul(A, k2))
    k4 = torch.linalg.solve(eye(n) - dt/2*A, Au + dt/4*matmul(A, 3*k2-k3))
    return dt * (3/4*k2 - 1/4*k3 + 1/2*k4)


def run_KS(u, c, dx, dt, T):
    # plot time averages of spatial average of u for a given values of c.


    # u contains boundary nodes
    n = u.shape[0]
    t = 0
    spatial_avg = 0
    while t < T:
        u = u + explicit_rk(u, c, dx, dt) + implicit_rk(u, c, dx, dt) 
        u[0], u[-1] = 0., 0.
        t += dt
        if t >=100:
            spatial_avg += torch.mean(u)
        # print("At time", t, " Max u", torch.max(u))
        # print("Min u", torch.min(u))
    time_avg = spatial_avg / 2000
    return u, time_avg
    
