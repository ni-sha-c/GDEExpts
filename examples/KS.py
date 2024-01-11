import torch
from torch import *
import torch.sparse as tosp
# add boundary condition, test implicit and explicit
# time integration imex
# plot timeaverages of spatial average of u for different values of c.

def rhs_KS_implicit(dx):
    n = u.shape[0]
    A = tosp.spdiags(torch.vstack((-ones(n), 2*ones(n), -ones(n)))/dx/dx, torch.tensor([-1, 0, 1]), (n, n))
    A -= tosp.spdiags(torch.vstack((ones(n), -4*ones(n), 6*ones(n), -4*ones(n), ones(n)))/dx/dx/dx/dx, torch.tensor([-2, -1, 0, 1, 2]), (n, n))
    return A 

def rhs_KS_explicit(u, c, dx):
    n = u.shape[0]
    B = tosp.spdiags(torch.vstack((ones(n), -ones(n)))/(2*dx), torch.tensor([1,-1]), (n, n))
    
    A = -B*c
    return torch.matmul(A, u) - torch.matmul(B, u)*u

def explicit_rk(u, c, dx, dt):
    k1 = rhs_KS_explicit(u, c, dx)
    k2 = rhs_KS_explicit(u + dt/3*k1, c, dx)
    k3 = rhs_KS_explicit(u + dt*k2, c, dx)
    k4 = rhs_KS_explicit(u + dt*(0.75*k2 + 0.25*k3), c, dx)
    return u + dt*(3/4*k2 - 1/4*k3 + 1/2*k4)

def implicit_rk(u, c, dx, dt):
    n = u.shape[0]
    A = rhs_KS_implicit(dx)
    Au = torch.matmul(A, u)
    k2 = linalg.solve(eye(n) - dt/3*A, Au)
    k3 = linalg.solve(eye(n) - dt/2*A, Au + dt/2*matmul(A, k2))
    k4 = linalg.solve(eye(n) - dt/2*A, Au + dt/4*matmul(A, 3*k2-k3))
    return dt * (3/4*k2 - 1/4*k3 + 1/2*k4)



"""
def KuramotoSivashinsky(u, c, dt, L):
    # From ... paper

    # 1-D modified KS equation
    # du/dt = -(u+c)du/dx - d^2u/dx^2 - d^4u/dx^4
    # x is in [0, L]
    # N is the number of points calculated along x
    # Boundary condition with u(0,t) = u(L,t) = 0

    # Total number of nodes
    # i = 0, 1, 2, ... n, n+1
    n = 127 # number of interior nodes, i = 1, 2, ..., n
    dx = L/(n+1)
    dx_square = dx**2
    dx_fourth = dx**4

    # Matrix A
    A = torch.zeros(128, 128)
    A[range(1, A.shape[0]), range(0, A.shape[1] - 1)] = -1  
    A[range(0, A.shape[0] - 1), range(1, A.shape[1])] = 1  
    print(A)

    # Matrix B
    B = torch.zeros(128, 128)
    B[range(1, B.shape[0]), range(0, B.shape[1] - 1)] = 1
    B[range(0, B.shape[0] - 1), range(1, B.shape[1])] = 1
    B[range(0, B.shape[0]), range(0, B.shape[1])] = -2
    print(B)

    # Matrix C
    C = torch.zeros(128,128)
    C[range(1, C.shape[0]), range(0, C.shape[1] - 1)] = -4
    C[range(0, C.shape[0] - 1), range(1, C.shape[1])] = -4
    C[range(2, C.shape[0]), range(0, C.shape[1] - 2)] = 1
    C[range(0, C.shape[0] - 2), range(2, C.shape[1])] = 1
    C[range(0, C.shape[0]), range(0, C.shape[1])] = 6
    print(C[0:3, :6])


    # first term: dudx
    first_term = 1/(2*dx) * torch.matmul(A, u)
    print("ft", first_term.shape)

    # second term: ududx
    u_square = u * u # Element-wise mult
    print("square", u[:2], u_square[:2])
    second_term = 1/(4*dx) * torch.matmul(A, u_square)

    # third term: d^2u/dx^2
    third_term = 1/dx_square * torch.matmul(B, u)

    # fourth term: d^4/dx^4
    fourth_term = 1/dx_fourth * torch.matmul(C, u)

    # dudt
    dudt = -second_term - c*first_term - third_term - fourth_term
    print(dudt)

    # Boundary Condition
    dudt[0] = 0
    dudt[-1] = 0

    # compute dudt[1]
    f_1 = u[2]/(2*dx)
    f_2 = u_square[2]/(4*dx_square)
    f_3 = (u[2]-u[1])/dx_square
    f_4 = (7*u[1]-4*u[2]+u[3])/dx_fourth
    dudt[1] = -f_2 - c*f_1 - f_3 - f_4    

    # compute dudt[n]
    n_1 = -u[n-1]/(2*dx)
    n_2 = -u_square[n-1]/(4*dx_square)
    n_3 = (u[n-1]-2*u[n])/dx_square
    n_4 = (7*u[n]-4*u[n-1]+u[n-2])/dx_fourth
    dudt[n] = -n_2 - c*n_1 - n_3 - n_4  

    # Rest
    second_4 = (-4*u[1] + 6*u[2]-4*u[3]+u[4])/dx_fourth
    second_last_4 = (-4*u[n] + 6*u[n-1]-4*u[n-2]+u[n-3])/dx_fourth

    dudt[2] = dudt[2] - second_4
    dudt[n-1] = dudt[n-1] - second_last_4

    return dudt
"""

if __name__ == '__main__':
    u = torch.arange(3).float()
    dx = 1.0
    print(second_derivative(u, dx))
    #L = 128

    #u = torch.randn(L).T # [L, 1]
    #print("u shape", u.shape)
    #print("u", u[:2])
    #u[0] = 0
    #u[-1] = 0

    #c = 0.4
    #dt = 0.1

    #KS = KuramotoSivashinsky(u, c, dt, L)
    
