import numpy as np
import NODE_solve_Lorenz as sol 
import torch


def Jacobian_Matrix(v, sigma, r, b):
    '''Compute Jacobian Matrix'''

    x, y, z = [k for k in v]
    return np.array([[-sigma, sigma, 0], [r - z, -1, -x], [y, x, -b]])



def lyap_exps(iters=10**5, x0 = np.ones(3)):
    '''Compute lyap_exps'''

    #initial parameters
    sigma = 10
    r = 28
    b = 8/3

    # U is tangent vector
    # v = regular system
    # QR Method 
    U = np.eye(3)
    x0 = torch.Tensor([ 0.1, 0.1, 0.1])
    lyap = [] #empty list to store the lengths of the orthogonal axes

    iters=10**5
    dt=0.001
    real_time = iters * dt
    print("real time length: ", real_time)


    tran = 0
    true_traj = sol.simulate(0, real_time, x0, iters)
    true_traj = true_traj[tran:]
    print(true_traj.shape)
    I = np.eye(3)


    for i in range(0, iters):

        x0 = true_traj[i] #update x0
        J = np.matmul(I + dt * Jacobian_Matrix(x0, sigma, r, b), U)
        # QR Decomposition for J
        Q, R = np.linalg.qr(J)
        lyap.append(np.log(abs(R.diagonal())))
        U = Q #new axes after iteration

    LE = [sum([lyap[i][j] for i in range(iters)]) / (real_time) for j in range(3)]
    
    return LE



def long_time_avg(pred_result, true_result):
    '''For Calculating Long Time Average'''

    pred = np.loadtxt(pred_result, delimiter=",", dtype=float)
    true = np.loadtxt(true_result, delimiter=",", dtype=float)

    print(pred[0:5, 2])
    print(pred[1000:1005, 2])

    print(np.mean(pred[:,2]))
    print(np.mean(true[:,2]))


##### ----- test run ----- #####
LE = lyap_exps()
print(LE)

# from import step
# def lyap_exps(N, x0):
#     a = 0
#     x = copy(x0)
#     d = x.shape[0]
#     v = rand(d)
#     for i  in range(N):
#         v += dot(dstep(x), v)
#         b = norm(v)
#         a += log(b)/N
#         v /= norm(v)
#         x = step(x)
#     return a

# def time_average(N, x0):
#     z = x0[-1]
#     x = copy(x0)
#     d = x.shape[0]
#     a = 0
#     for i  in range(N):
#         a += x[-1]/N
#         x = step(x)
#     return a