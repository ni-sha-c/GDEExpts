from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from test_metrics import *
from scipy.integrate import odeint

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


def plot_3d_space(n, data, dyn_sys, time_step, optim_name, NODE, integration_time):
    ''' func: plot true phase or simulated phase for 3D dynamic system 
        param:  n = num of time step size
                data = simulated or true trajectory '''

    DYNSYS_MAP = {'sin' : [sin, 1],
                  'tent_map' : [tent_map, 1],
                  'brusselator' : [brusselator, 2],
                  'lorenz_periodic' : [lorenz_periodic, 3],
                  'lorenz' : [lorenz, 3]}

    # If want to plot true trajectory,
    if NODE == False:
        dyn_system, dim = DYNSYS_MAP[dyn_sys]
        ti, tf = integration_time
        init_state = torch.randn(dim)
        simulate(dyn_system, ti, tf, init_state, time_step)
        print("Finished simulating")

        path = '../test_result/expt_'+str(dyn_sys)+'/'+ optim_name + '/' + str(time_step) + '/'+'phase_plot_' + str(time_step) +'.pdf'
    elif NODE == True:
        path = '../test_result/expt_'+str(dyn_sys)+'/'+ optim_name + '/' + str(time_step) + '/'+'NODE_phase_plot_' + str(time_step) +'.pdf'

    fig, (ax1, ax2, ax3) = subplots(1, 3, figsize=(18,6))
    my_range = np.linspace(-1,1,n)
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    ax1.scatter(x, y, c=z, s = 2, cmap='plasma', alpha=0.5)
    ax1.set_xlabel("X-axis")
    ax1.set_ylabel("Y-axis")

    ax2.scatter(x, z, c=z, s = 2 ,cmap='plasma', alpha=0.5)
    ax2.set_xlabel("X-axis")
    ax2.set_ylabel("Z-axis")

    ax3.scatter(y, z, c=z, s = 2 ,cmap='plasma', alpha=0.5)
    ax3.set_xlabel("Y-axis")
    ax3.set_ylabel("Z-axis")

    fig.savefig(path, format='pdf', dpi=1200)

    return



def lorenz_system(x, y, z, rho, beta=8/3, sigma=10.0):
    x_dot = sigma * (y - x)
    y_dot = x * (rho - z) - y
    z_dot = x * y - beta * z
    return x_dot, y_dot, z_dot



def lorenz_bifurcation_plot(time_step, r_range, dr=0.1):
    ''' func: plot bifurcation plot for 3D system 
        param:  r_range = range of rho
                dr = step size of rho '''
    # should modify more!!

    r = np.arange(20, r_range, dr)  # parameter range
    dt = 0.001  # time step
    t = np.arange(0, 50, dt)  # time range

    # initialize solution arrays
    xs = np.empty(len(t) + 1)
    ys = np.empty(len(t) + 1)
    zs = np.empty(len(t) + 1)

    # initial values x0,y0,z0 for the system
    xs[0], ys[0], zs[0] = (1, 1, 1)

    # Save the plot points coordinates and plot the with a single call to plt.plot
    # instead of plotting them one at a time, as it's much more efficient
    r_maxes, z_maxes, r_mins, z_mins = ([] for i in range(4))

    for R in r:
        # Print something to show everything is running
        print(f"{R=:.2f}")
        for i in range(len(t)):
            # approximate numerical solutions to system
            x_dot, y_dot, z_dot = lorenz_system(xs[i], ys[i], zs[i], R)
            xs[i + 1] = xs[i] + (x_dot * dt)
            ys[i + 1] = ys[i] + (y_dot * dt)
            zs[i + 1] = zs[i] + (z_dot * dt)
        # calculate and save the peak values of the z solution
        for i in range(1, len(zs) - 1):
            # save the local maxima
            if zs[i - 1] < zs[i] and zs[i] > zs[i + 1]:
                r_maxes.append(R)
                z_maxes.append(zs[i])
            # save the local minima
            elif zs[i - 1] > zs[i] and zs[i] < zs[i + 1]:
                r_mins.append(R)
                z_mins.append(zs[i])

        # "use final values from one run as initial conditions for the next to stay near the attractor"
        xs[0], ys[0], zs[0] = xs[i], ys[i], zs[i]

    fig = subplot(figsize=(18,6))
    scatter(r_maxes, z_maxes, color="black", s=1, alpha=0.4)
    scatter(r_mins, z_mins, color="red", s=1, alpha=0.4)
    xlim(0, r_range)
    ylim(0, 300)

    path = '../plot/'+'bifurcation_plot'+'.pdf'
    fig.savefig(path, format='pdf', dpi=1200)
    return


def create_lorenz_with_diff_rho(rho):
  """Creates a Lorenz function with a different rho value.
  Args:
    rho: The rho value.
  Returns:
    A Lorenz function with the specified rho value.
  """
  def lorenz_function(t, u):
    """Lorenz chaotic differential equation: du/dt = f(t, u)

    t: time T to evaluate system
    u: state vector [x, y, z]
    return: new state vector in shape of [3]"""

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
    path = '../plot/'+'LE_diff_rho'+'.pdf'
    fig.savefig(path, format='pdf')

    # save
    np.savetxt('../test_result/expt_'+dyn_sys+'/'+ "LE_diff_rho.csv", lyap_exp, delimiter=",")
    return



# def test_error_diff_rho():
#     ''' func: save csv file that stores test error of different rho and create plot '''

#     return


def plot_traj_lorenz(X, optim_name, time, periodic):
    '''Plot trajectory of lorenz training data'''

    plt.figure(figsize=(40,10))
    plt.plot(X[:, 0], color="C1")
    plt.plot(X[:, 1], color="C2")
    plt.plot(X[:, 2], color="C3")

    plt.title('Trajectory of Training Data')
    plt.xticks()
    plt.yticks()
    plt.legend(["X", "Y", "Z"])
    if periodic == True:
        plt.savefig('expt_lorenz_periodic/' + optim_name + '/' + str(time) + '/' + 'train_data_traj', format='png', dpi=400, bbox_inches ='tight', pad_inches = 0.1)
    else:
        plt.savefig('expt_lorenz/' + optim_name + '/' + str(time) + '/' + 'train_data_traj', format='png', dpi=400, bbox_inches ='tight', pad_inches = 0.1)

    return



def plot_phase_space_lorenz(pred_test, Y_test, optim_name, lr, time, periodic):
    '''plot phase space of lorenz'''

    plt.figure(figsize=(20,15))
    ax = plt.axes(projection='3d')
    ax.grid()
    ax.plot3D(Y_test[:, 0], Y_test[:, 1], Y_test[:, 2], 'gray', linewidth=5)
        
    z = pred_test[:, 2]
    ax.scatter3D(pred_test[:, 0], pred_test[:, 1], z, c=z, cmap='hsv', alpha=0.3, linewidth=0)
    ax.set_title('Phase Space')
    if periodic == True:
        plt.savefig('expt_lorenz_periodic/' + optim_name + '/' + str(time) + '/' + 'Phase Space with ' + 'lr=' + str(lr), format='png', dpi=400, bbox_inches ='tight', pad_inches = 0.1)
    else:
        plt.savefig('expt_lorenz/' + optim_name + '/' + str(time) + '/' + 'Phase Space with ' + 'lr=' + str(lr), format='png', dpi=400, bbox_inches ='tight', pad_inches = 0.1)
    plt.show()
    plt.close("all")

    return



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
    # traj = np.genfromtxt("../test_result/expt_lorenz/AdamW/0.01/pred_traj.csv", delimiter=",", dtype=float)
    # n = len(traj)
    # print(n)
    #plot_3d_space(n, traj, "lorenz", 0.01, "AdamW", False, [0, 180])

    LE_diff_rho(dyn_sys="lorenz", r_range=200, dr=5, time_step=0.01)
