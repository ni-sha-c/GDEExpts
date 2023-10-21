import ctypes
import itertools
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import multiprocessing
import numpy as np
from scipy.integrate import odeint
from scipy.signal import argrelextrema
from test_metrics import *

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
sys.path.append('..')

from src.NODE_solve import *
from examples.Brusselator import *
from examples.Lorenz import *
from examples.Lorenz_periodic import *
from examples.Sin import *
from examples.Tent_map import *


''' List of functions included in test_util.py:

    1. plot_3d_space() 
    2. compute_lorenz_bif()
    3. plot_lorenz_bif()
    4. plot_3d_trajectory()
    5. create_lorenz_with_diff_rho()
    6. LE_diff_rho() 
'''



def plot_3d_space(n, data, dyn_sys, time_step, optim_name, NODE, integration_time, ALL=True, tran_state=100):
    ''' func: plot true phase or multi-time step simulated phase for 3D dynamic system, each from different random initial point 
        param:  n = num of time step size
                data = simulated or true trajectory
                dyn_sys = dynamical system (str)
                NODE = plot Neural ODE instead of true traj (bool) '''

    DYNSYS_MAP = {'sin' : [sin, 1],
                  'tent_map' : [tent_map, 1],
                  'brusselator' : [brusselator, 2],
                  'lorenz_periodic' : [lorenz_periodic, 3],
                  'lorenz' : [lorenz, 3]}
    dyn_system, dim = DYNSYS_MAP[dyn_sys]
    ti, tf = integration_time
    init_state = torch.randn(dim)
    true_data = simulate(dyn_system, ti, tf, init_state, time_step)[tran_state:]

    # If want to plot either NODE or true trajectory only,
    if ALL == True:
        # If want to plot true trajectory only,
        if NODE == False:
            path = '../test_result/expt_'+str(dyn_sys)+'/'+ optim_name + '/' + str(time_step) + '/'+'phase_plot_' + str(time_step) +'.pdf'
            data = true_data
        # If want to plot NODE trajectory only,
        elif NODE == True:
            path = '../test_result/expt_'+str(dyn_sys)+'/'+ optim_name + '/' + str(time_step) + '/'+'NODE_phase_plot_' + str(time_step) +'.pdf'
        
        fig, (ax1, ax2, ax3) = subplots(1, 3, figsize=(18,6))
        my_range = np.linspace(-1,1,n)
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]

        ax1.scatter(x, y, c=z, s = 3, cmap='plasma', alpha=0.5)
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")

        ax2.scatter(x, z, c=z, s = 3,cmap='plasma', alpha=0.5)
        ax2.set_xlabel("X")
        ax2.set_ylabel("Z")

        ax3.scatter(y, z, c=z, s = 3,cmap='plasma', alpha=0.5)
        ax3.set_xlabel("Y")
        ax3.set_ylabel("Z")

        #plt.colorbar(sc)
        fig.savefig(path, format='pdf', dpi=1200)
    
    # If want to plot comparison plot between NODE vs true trajectory
    else:
        print("Plot comparison!")
        path = '../plot/Compare_phase_plot_' + str(time_step) +'.pdf'

        limit = 10000
        x = data[:limit-tran_state, 0]
        y = data[:limit-tran_state, 1]
        z = data[:limit-tran_state, 2]

        x_true = true_data[:limit-tran_state, 0]
        y_true = true_data[:limit-tran_state, 1]
        z_true = true_data[:limit-tran_state, 2]
        t = torch.arange(ti, tf, time_step)[:limit-tran_state]
        print("t", t.shape)
        print("x", x.shape)
        print("x_true", x_true.shape)

        fig = figure(figsize=(18, 6))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05])  # 3 columns: 2 for subplots and 1 for colorbar

        # Create the first subplot
        ax1 = subplot(gs[0])
        ax1.scatter(x, z, c=t, s = 4, cmap='winter', alpha=0.5)
        ax1.set_xlabel("X", fontsize=18)
        ax1.set_ylabel("Z", fontsize=18)
        ax1.set_title("Phase Portrait of Neural ODE", fontsize=18)

        # Create the second subplot
        ax2 = subplot(gs[1], sharey=ax1)  # sharey ensures equal height
        sc = ax2.scatter(x_true, z_true, c=t, s = 4,cmap='winter', alpha=0.5)
        ax2.set_xlabel("X", fontsize=18)
        ax2.set_ylabel("Z", fontsize=18)
        ax2.set_title("Phase Portrait of True ODE", fontsize=18)

        # Add a colorbar to the right of the subplots
        cax = subplot(gs[2])
        cbar = colorbar(sc, cax=cax)
        xaxis.set_tick_params(labelsize=24)
        tight_layout()

        fig.savefig(path, format='pdf', dpi=1200)
    return



def lorenz_system(x, y, z, rho, beta=8/3, sigma=10.0):
    x_dot = sigma * (y - x)
    y_dot = x * (rho - z) - y
    z_dot = x * y - beta * z
    return x_dot, y_dot, z_dot



# def lorenz_bif_plot(dyn_sys, time_step, r_range=200, dr=0.1, tf=200, n=100):
#     ''' func: plot bifurcation plot for 3D system 
#               while also checking ergodicity of system
#               at certain rho value
#         param:  r_range = range of rho
#                 dr = step size of rho
#                 tf = integration time from [0, tf] 
#                 n = number of initial condition '''

#     # r_H = 24.74 when sigma = 10, beta = 8/3
#     # r < 1 | 1 < r < 24.74 | r = 24.74 | r > 24.74
#     # R * n * T = 200 * 2 * 10 = 4000

#     # range of parameter rho
#     r = np.arange(0, r_range, dr)
#     R = len(r)
#     # range of time
#     t = np.arange(0, tf, time_step)  # time range # 50
#     T = len(t)
#     # transition_phase
#     trans_t = 5000

#     # initialize solution arrays
#     xs, ys, zs = (np.empty(T+ 1) for i in range(3))
#     # initial values x0,y0,z0 for the system
#     z_maxes, z_mins = ([] for i in range(4))

#     for R in r:
#         print(f"{R=:.2f}")
#         for ni in range(n):
#             init_state = torch.rand(3) * 10 #np.random.rand(3)
#             xs[0], ys[0], zs[0] = init_state
#             # Find solution
#             for i in range(len(t)):
#                 x_dot, y_dot, z_dot = lorenz_system(xs[i], ys[i], zs[i], R)
#                 xs[i + 1] = xs[i] + (x_dot * time_step)
#                 ys[i + 1] = ys[i] + (y_dot * time_step)
#                 zs[i + 1] = zs[i] + (z_dot * time_step)
#             # save global maximum
#             z_maxes.append(np.max(zs[trans_t:]))
#             z_mins.append(np.min(zs[trans_t:]))
#             if (z_maxes[-1] - z_mins[-1]) < 1e-7:
#                 print("fixed point at rho=", R)
#         if (z_maxes[-1] - z_maxes[-10]) < 1e-7:
#             print("ergodic at rho=", R)

#     print("z", np.asarray(z_maxes).shape)

#     fig, ax = subplots(figsize=(36,12))
#     ax.scatter(r, z_maxes, color=(0.25, 0.25, 0.25), s=1, alpha=0.4)
#     ax.scatter(r, z_mins, color="lightgreen", s=1, alpha=0.4)
#     # Plot the bifurcation plot values for r = 24.74 as a dashed line
#     ax.plot(np.array([24.74, 24.74]), np.array([0, 150]), linestyle='--', color="lightgray")
#     ax.xaxis.set_tick_params(labelsize=24)
#     xlim(0, r_range)
#     ylim(0, 500)

#     path = '../plot/'+'bifurcation_plot_modified_longer_'+str(n)+'.svg'
#     fig.savefig(path, format='svg', dpi=400)
#     return



def compute_lorenz_bif(hyper_params):
    ''' func: plot bifurcation plot for 3D system 
              while also checking ergodicity of system
              at certain rho value
        param:  R = rho of interest
                tf = integration time from [0, tf] 
                n = number of initial condition '''

    # r_H = 24.74 when sigma = 10, beta = 8/3
    # r < 1 | 1 < r < 24.74 | r = 24.74 | r > 24.74
    # R * n * T = 200 * 2 * 10 = 4000

    R, ni = hyper_params
    print(R, ni)
    # range of time
    tf = 50
    time_step = 0.001
    t = np.arange(0, tf, time_step)  # time range # 50
    T = len(t)
    # transition_phase
    trans_t = 20000

    # initialize solution arrays
    xs, ys, zs = (np.empty(T+ 1) for i in range(3))
    # initial values x0,y0,z0 for the system
    z_maxes, z_mins = ([] for i in range(2))

    xs[0], ys[0], zs[0] = torch.rand(3) * 10 #np.random.rand(3)
    # Find solution
    for i in range(len(t)):
        x_dot, y_dot, z_dot = lorenz_system(xs[i], ys[i], zs[i], R)
        xs[i + 1] = xs[i] + (x_dot * time_step)
        ys[i + 1] = ys[i] + (y_dot * time_step)
        zs[i + 1] = zs[i] + (z_dot * time_step)
    # save global maximum
    z_maxes = np.max(zs[trans_t:])
    z_mins = np.min(zs[trans_t:])

    res = [z_mins, z_maxes]
    return res



def plot_lorenz_bif(dyn_sys, r, z_mins, z_maxes, n):
    ''' func: plot bifurcation plot for 3D system 
              while also checking ergodicity of system
              at certain rho value '''


    fig, ax = subplots(figsize=(36,12))
    ax.scatter(r, z_maxes, color=(0.25, 0.25, 0.25), s=1, alpha=0.4)
    ax.scatter(r, z_mins, color="lightgreen", s=1, alpha=0.4)
    # Plot the bifurcation plot values for r = 24.74 as a dashed line
    ax.plot(np.array([24.74, 24.74]), np.array([0, 150]), linestyle='--', color="lightgray")
    ax.xaxis.set_tick_params(labelsize=24)
    ax.yaxis.set_tick_params(labelsize=24)
    xlim(0, r_range)
    ylim(0, 300)

    path = '../plot/'+'bifurcation_plot_'+str(n)+'.svg'
    fig.savefig(path, format='svg', dpi=400)
    return



def plot_3d_trajectory(Y, pred_test, comparison=False):
    figure(figsize=(20, 15))
    ax = axes(projection='3d')
    ax.grid()
    ax.plot3D(Y[:, 0], Y[:, 1], Y[:, 2], 'gray', linewidth=4)

    if comparison == True:
        z = pred_test[:, 2]
        ax.scatter3D(pred_test[:, 0], pred_test[:, 1], z, c=z, cmap='hsv', alpha=0.3, linewidth=0)
        ax.set_title(f"Iteration {iter+1}")
        ax.xaxis.set_tick_params(labelsize=24)
        ax.yaxis.set_tick_params(labelsize=24)
        savefig('expt_'+str(dyn_sys)+'/'+ optimizer_name + '/trajectory/' +str(iter+1)+'.png', format='png', dpi=400, bbox_inches ='tight', pad_inches = 0.1)
        close("all")
    else:
        ax.set_title(f"Iteration {iter+1}")
        ax.xaxis.set_tick_params(labelsize=24)
        ax.yaxis.set_tick_params(labelsize=24)
        savefig('expt_'+str(dyn_sys)+'/'+ optimizer_name + '/trajectory/' +str(iter+1)+'.png', format='png', dpi=400, bbox_inches ='tight', pad_inches = 0.1)
        close("all")

    return



def create_lorenz_with_diff_rho(rho):
  """ Creates a Lorenz function with a different rho value. """
  def lorenz_function(t, u):
    """ Lorenz chaotic differential equation: du/dt = f(t, u) """

    sigma = 10.0
    beta = 8/3

    res = torch.stack([
        sigma * (u[1] - u[0]),
        u[0] * (rho - u[2]) - u[1],
        (u[0] * u[1]) - (beta * u[2])
    ])
    return res
  return lorenz_function



def LE_diff_rho(dyn_sys="lorenz", r_range=200, dr=5, time_step=0.01):
    ''' func: save csv file that stores LE of different rho and create plot '''

    DYNSYS_MAP = {'sin' : [sin, 1],
                  'tent_map' : [tent_map, 1],
                  'brusselator' : [brusselator, 2],
                  'lorenz_periodic' : [lorenz_periodic, 3],
                  'lorenz' : [lorenz, 3]}

    dyn_sys_info = DYNSYS_MAP[dyn_sys]
    dyn_sys_func, dim = dyn_sys_info
    
    r = np.arange(0, r_range, dr) #40
    init_state = torch.tensor([1., 1., 1.])
    iters = 5*(10**4)
    real_time = iters * time_step
    longer_traj = simulate(dyn_sys_func, 0, real_time, init_state,time_step)

    lyap_exp = np.zeros((len(r), 2))

    for i in range(len(r)):
        new_lorenz = create_lorenz_with_diff_rho(r[i])
        dyn_sys_info = [new_lorenz, dim]
        LE_rk4 = lyap_exps(dyn_sys, dyn_sys_info, longer_traj, iters=iters, time_step= time_step, optim_name="AdamW", method="rk4")

        print(i, "LE: ", LE_rk4)
        lyap_exp[i, 0] = int(r[i])
        lyap_exp[i, 1] = LE_rk4[0]

    # plot
    fig, ax = subplots(figsize=(18,6))
    ax.scatter(r, lyap_exp[:, 1], color="lime", s=40, alpha=0.5, edgecolors='black')
    ax.xaxis.set_tick_params(labelsize=24)
    ax.yaxis.set_tick_params(labelsize=24)
    path = '../plot/'+'LE_diff_rho'+'.pdf'
    fig.savefig(path, format='pdf')

    # save
    np.savetxt('../test_result/expt_'+dyn_sys+'/'+ "LE_diff_rho.csv", lyap_exp, delimiter=",")
    return



# def test_error_diff_rho():
#     ''' func: save csv file that stores test error of different rho and create plot '''

#     return




def plot_time_space_lorenz(X, X_test, Y_test, pred_train, true_train, pred_test, loss_hist, optim_name, lr, num_epoch, time_step, periodic):
    '''plot time_space for training/test data and training loss for lorenz system'''

    pred_train = np.array(pred_train)
    true_train = np.array(true_train)
    pred_test = np.array(pred_test)
    pred_train_last = pred_train[-1]
    true_train_last = true_train[-1]

    plt.figure(figsize=(40,10))

    num_timestep = 1500
    substance_type = 0
    x = list(range(0,num_timestep))
    x_loss = list(range(0,num_epoch))

    plt.subplot(2,2,1)
    plt.plot(pred_train_last[:num_timestep, substance_type], marker='+', linewidth=1)
    plt.plot(true_train_last[:num_timestep, substance_type], alpha=0.7, linewidth=1)
    plt.plot(X[:num_timestep, substance_type], '--', linewidth=1)
    plt.legend(['y_pred @ t + {}'.format(1), 'y_true @ t + {}'.format(1), 'x @ t + {}'.format(0)])
    plt.title('Substance type A prediction at {} epoch, Train'.format(num_epoch))

    plt.subplot(2,2,2)
    plt.plot(pred_test[:num_timestep, substance_type], marker='+', linewidth=1)
    plt.plot(Y_test[:num_timestep, substance_type], linewidth=1)
    plt.plot(X_test[:num_timestep, substance_type], '--', linewidth=1)
    plt.legend(['y_pred @ t + {}'.format(1), 'y_true @ t + {}'.format(1), 'x @ t + {}'.format(0)])
    plt.title('Substance type A prediction at {} epoch, Test'.format(num_epoch))

    ##### Plot Training Loss #####
    plt.subplot(2,2,3)
    plt.plot(x_loss, loss_hist)
    plt.title('Training Loss')
    plt.xticks()
    plt.yticks()
    if periodic == True:
        plt.savefig('expt_lorenz_periodic/' + optim_name + '/' + str(time_step) + '/' + 'Time Space, Training Loss, Test Loss with ' + 'lr=' + str(lr), format='png', dpi=400, bbox_inches ='tight', pad_inches = 0.1)
    else:
        plt.savefig('expt_lorenz/' + optim_name + '/' + str(time_step) + '/' + 'Time Space, Training Loss, Test Loss with ' + 'lr=' + str(lr), format='png', dpi=400, bbox_inches ='tight', pad_inches = 0.1)
    plt.show()
    plt.close("all")

    return



if __name__ == '__main__':

    #----- test plot_3d_space() -----#
    # traj = np.genfromtxt("../test_result/expt_lorenz/AdamW/0.01/pred_traj.csv", delimiter=",", dtype=float)
    # n = len(traj)
    # plot_3d_space(n, traj, "lorenz", 0.01, "AdamW", True, [0, 500], False, 100)

    #LE_diff_rho(dyn_sys="lorenz", r_range=200, dr=5, time_step=0.01)


    #----- test bifurcation plot -----#

    # 1. initialize
    r_range=200
    r = np.arange(0, r_range, 0.1) # range of parameter rho
    n = range(100) # num of new initial condition
    param_list = list(itertools.product(r, n))

    # 2. run parallel
    with multiprocessing.Pool(processes=1000) as pool:
        res = pool.map(compute_lorenz_bif, param_list)
        res = np.array(res)
        z_mins, z_maxes = res[:, 0], res[:, 1]
    
    # 3. create plot
    print("creating plot... ")
    r_axis = [el for el in r for i in range(len(n))]
    r_axis = np.array(r_axis)
    print(len(n))
    plot_lorenz_bif("lorenz", r_axis, z_mins, z_maxes, len(n))

    # 4. Check property
    fixed_point = []
    ergodic_point = []
    print("z_max", z_maxes[:5])
    print("min", z_mins[:5])
    for i in range(0, len(r_axis), 100):
        if (np.abs(z_maxes[i] - z_mins[i])) < 1e-8:
            print("fixed point at rho=", r_axis[i])
            fixed_point.append([r_axis[i], z_maxes[i]])
        if (np.abs(z_maxes[i+10] - z_maxes[i+90])) < 1e-8:
            print("ergodic at rho=", r_axis[i])
            ergodic_point.append([r_axis[i], z_maxes[i+10]])
    np.savetxt('../test_result/expt_lorenz/'+ "fixed_point.csv", np.asarray(fixed_point), delimiter=",")
    np.savetxt('../test_result/expt_lorenz/'+ "ergodic_point.csv", np.asarray(ergodic_point), delimiter=",")
    


