import ctypes
import itertools
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import multiprocessing
import numpy as np
import scipy
from scipy.fft import fft, rfft
from scipy.integrate import odeint
from scipy.signal import argrelextrema
from scipy.signal import correlate
from scipy.stats import wasserstein_distance
from test_metrics import *
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
sys.path.append('..')

from src.NODE_solve import *
from src.NODE import *
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
    7. plot_time_space_lorenz()
    8. correlation_plot()
'''



def plot_3d_space(device, dyn_sys, time_step, optim_name, NODE, integration_time, ALL, tran_state, limit, init_state, model_path, pdf_path):
    ''' func: plot true phase or multi-time step simulated phase for 3D dynamic system, each from different random initial point 
        param:  n = num of time step size
                data = simulated or true trajectory
                dyn_sys = dynamical system (str)
                NODE = plot Neural ODE instead of true traj (bool) '''

    # call h(x) of dynamical system of interest
    dyn_system, dim = define_dyn_sys(dyn_sys)
    ti, tf = integration_time

    # simulate true trajectory
    true_data = simulate(dyn_system, ti, tf, init_state, time_step)
    n = true_data.shape[0]

    # If want to plot either NODE or true trajectory only,
    if ALL == True:

        fig, axs = subplots(2, 3, figsize=(36,18))
        my_range = np.linspace(-1,1,n)
        true_x = true_data[:, 0]
        true_y = true_data[:, 1]
        true_z = true_data[:, 2]

        # Load the saved model
        model = ODE_Lorenz().to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print("Finished Loading model")
 
        # simulate NODE trajectory
        data = simulate(model.double(), ti, tf, init_state.double().to(device), time_step)
        data = data.detach().cpu().numpy()

        # limit = 50000
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]

        cmap = cm.plasma

        axs[0,0].plot(true_x[0], true_y[0], '+', markersize=35, color=cmap.colors[0])
        axs[0,0].scatter(true_x, true_y, c=true_z, s = 6, cmap='plasma', alpha=0.5)
        axs[0,0].set_xlabel("X", fontsize=48)
        axs[0,0].set_ylabel("Y", fontsize=48)
        axs[0,0].tick_params(labelsize=48)

        axs[0,1].plot(true_x[0], true_z[0], '+', markersize=35, color=cmap.colors[0])
        axs[0,1].scatter(true_x, true_z, c=true_z, s = 6,cmap='plasma', alpha=0.5)
        axs[0,1].set_xlabel("X", fontsize=48)
        axs[0,1].set_ylabel("Z", fontsize=48)
        axs[0,1].tick_params(labelsize=48)

        axs[0,2].plot(true_y[0], true_z[0], '+', markersize=15, color=cmap.colors[0])
        axs[0,2].scatter(true_y, true_z, c=true_z, s = 6,cmap='plasma', alpha=0.5)
        axs[0,2].set_xlabel("Y", fontsize=48)
        axs[0,2].set_ylabel("Z", fontsize=48)
        axs[0,2].tick_params(labelsize=48)

        # NODE
        axs[1,0].plot(x[0], y[0], '+', markersize=15, color=cmap.colors[0])
        axs[1,0].scatter(x, y, c=z, s = 8, cmap='plasma', alpha=0.8)
        axs[1,0].set_xlabel("X", fontsize=48)
        axs[1,0].set_ylabel("Y", fontsize=48)
        axs[1,0].tick_params(labelsize=48)

        axs[1,1].plot(x[0], z[0], '+', markersize=15, color=cmap.colors[0])
        axs[1,1].scatter(x, z, c=z, s = 8,cmap='plasma', alpha=0.8)
        axs[1,1].set_xlabel("X", fontsize=48)
        axs[1,1].set_ylabel("Z", fontsize=48)
        axs[1,1].tick_params(labelsize=48)

        axs[1,2].plot(y[0], z[0], '+', markersize=8, color=cmap.colors[0])
        axs[1,2].scatter(y, z, c=z, s = 6,cmap='plasma', alpha=0.8)
        axs[1,2].set_xlabel("Y", fontsize=48)
        axs[1,2].set_ylabel("Z", fontsize=48)
        axs[1,2].tick_params(labelsize=48)

        tight_layout()
        #plt.colorbar(sc)
        fig.savefig(pdf_path, format='png', dpi=1200, bbox_inches ='tight', pad_inches = 0.1)
    
    # If want to plot comparison plot between NODE vs true trajectory
    else:
        print("Plot 2 phase plot!")
        #path = '../plot/Compare_phase_plot_trans_100_2nd' + str(tran_state) +'.pdf'

        # Load the saved model
        model = ODE_Lorenz().to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print("Finished Loading model")
 
        data = simulate(model.double(), ti, tf, init_state.double().to(device), time_step)
        data = data.detach().cpu().numpy()

        # limit = 50000
        x = data[tran_state:limit, 0]
        y = data[tran_state:limit, 1]
        z = data[tran_state:limit, 2]

        x_true = true_data[tran_state:limit, 0]
        y_true = true_data[tran_state:limit, 1]
        z_true = true_data[tran_state:limit, 2]
        t = torch.arange(ti, tf, time_step)[tran_state:limit]

        fig = figure(figsize=(18, 6))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05])  # 3 columns: 2 for subplots and 1 for colorbar

        cmap = cm.winter

        # Create the first subplot
        ax1 = subplot(gs[0])
        ax1.plot(x[0], z[0], marker='P', markersize=30, color="black", linewidth=2)
        ax1.scatter(x, z, c=t, s = 4, cmap='winter', alpha=0.6)
        ax1.set_xlabel("X", fontsize=32)
        ax1.set_ylabel("Z", fontsize=32)
        # ax1.set_title("Phase Portrait of Neural ODE", fontsize=18)
        ax1.tick_params(labelsize=32)

        # Create the second subplot
        ax2 = subplot(gs[1], sharey=ax1)  # sharey ensures equal height
        ax2.plot(x_true[0], z_true[0], marker='P', markersize=30, color="black", linewidth=2)
        sc = ax2.scatter(x_true, z_true, c=t, s = 4,cmap='winter', alpha=0.6)
        ax2.set_xlabel("X", fontsize=32)
        ax2.set_ylabel("Z", fontsize=32)
        # ax2.set_title("Phase Portrait of True ODE", fontsize=18)
        ax2.tick_params(labelsize=32)

        # Add a colorbar to the right of the subplots
        cax = subplot(gs[2])
        cbar = colorbar(sc, cax=cax)
        cbar.ax.tick_params(labelsize=28)
        tight_layout()

        fig.savefig(pdf_path, format='png', dpi=1200, bbox_inches ='tight', pad_inches = 0.1)
    return



def lorenz_system(x, y, z, rho, beta=8/3, sigma=10.0):
    x_dot = sigma * (y - x)
    y_dot = x * (rho - z) - y
    z_dot = x * y - beta * z
    return x_dot, y_dot, z_dot



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
    tf = 100 # from 50 to 100..!
    time_step = 0.001
    t = torch.arange(0, tf, time_step)  # time range # 50
    T = len(t)
    # transition_phase
    trans_t = 40000

    # initialize solution arrays
    # xs, ys, zs = (np.empty(T+ 1) for i in range(3))
    # initial values x0,y0,z0 for the system

    # xs[0], ys[0], zs[0] = torch.rand(3) * 10 #np.random.rand(3)
    # Find solution
    # for i in range(len(t)):
    #     x_dot, y_dot, z_dot = lorenz_system(xs[i], ys[i], zs[i], R)
    #     xs[i + 1] = xs[i] + (x_dot * time_step)
    #     ys[i + 1] = ys[i] + (y_dot * time_step)
    #     zs[i + 1] = zs[i] + (z_dot * time_step)
    init = torch.rand(3) * 10
    res = torchdiffeq.odeint(lorenz, init, t, method='rk4', rtol=1e-8)
    xs, ys, zs = res[:, 0], res[:, 1], res[:, 2]
    # save global maximum
    z_maxes = np.max(zs[trans_t:])
    z_mins = np.min(zs[trans_t:])
    # Time avg for 1 intial condition
    time_avg = np.mean(zs[trans_t:]) 

    res = [z_mins, z_maxes, time_avg]
    return res



def plot_lorenz_bif(dyn_sys, r, z_mins, z_maxes, n):
    ''' func: plot bifurcation plot for 3D system 
              while also checking ergodicity of system
              at certain rho value '''


    fig, ax = subplots(figsize=(36,12))
    ax.scatter(r, z_maxes, color=(0.25, 0.25, 0.25), s=1.5, alpha=0.5)
    ax.scatter(r, z_mins, color="lightgreen", s=1.5, alpha=0.5)

    # Plot the bifurcation plot values for r = 24.74 as a dashed line
    # ax.plot(np.array([24.74, 24.74]), np.array([0, 150]), linestyle='--', color="lightgray")
    ax.xaxis.set_tick_params(labelsize=24)
    ax.yaxis.set_tick_params(labelsize=24)
    xlim(0, r_range)
    ylim(0, 300)

    path = '../plot/'+'bifurcation_plot_new_'+str(n)+'.svg'
    fig.savefig(path, format='svg', dpi=400)
    return




def plot_3d_trajectory(dyn_sys, integration_time, model_name, time_step, init_state, device, comparison=False):

    # call h(x) of dynamical system of interest
    dyn_system, dim = define_dyn_sys(dyn_sys)
    ti, tf = integration_time

    model_path = "../test_result/expt_lorenz/AdamW/"+str(time_step)+'/'+str(model_name)+'/model.pt'
    pdf_path = '../plot/dist_'+str(model_name)+'_'+str(init_state.tolist())+'.pdf'

    # simulate true trajectory
    # true_data = simulate(dyn_system, ti, tf, init_state, time_step)

    # Load the saved model
    model = ODE_Lorenz().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("Finished Loading model")
    # simulate node trajectory
    node_data = simulate(model, ti, tf, init_state.to(device), time_step).detach().cpu()

    # Plot 
    figure(figsize=(20, 15))
    ax = axes(projection='3d')

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False


    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.grid()

    # true_data = true_data.detach().cpu().numpy()
    # ax.plot3D(true_data[:, 0], true_data[:, 1], true_data[:, 2], 'gray', linewidth=4)
    path = '../plot/3d'+str(model_name)+ '.pdf'

    if comparison == True:
        z = node_data[:, 2]
        ax.scatter3D(node_data[:, 0], node_data[:, 1], z, c=z, cmap='hsv', alpha=0.5, s=5, linewidth=4) #scatter3d
        # ax.set_title(f"Iteration {iter+1}")
        ax.xaxis.set_tick_params(labelsize=24)
        ax.yaxis.set_tick_params(labelsize=24)
        ax.zaxis.set_tick_params(labelsize=24)
        savefig(path, format='pdf', dpi=600, bbox_inches ='tight')
        close("all")
    else:
        # ax.set_title(f"Iteration {iter+1}")
        ax.xaxis.set_tick_params(labelsize=36)
        ax.yaxis.set_tick_params(labelsize=36)
        savefig(path, format='pdf', dpi=600, bbox_inches ='tight')
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





def plot_loss(MSE_train, MSE_test, J_train, J_test, MSE_train_5, MSE_test_5, J_train_5, J_test_5):
    fig, axs = subplots(1, 2, figsize=(24, 8)) #, sharey=True
    # fig.suptitle("Loss Behavior of Jacobian Loss Compared to MSE Loss", fontsize=24)
    
    colors = cm.turbo(np.linspace(0, 1, 20))

    # Training Loss
    x = np.arange(0, MSE_train.shape[0])

    axs[0].plot(MSE_train[0:500], c=colors[16], label='MSE_0', alpha=0.9, linewidth=4, marker='o', markersize=18, markevery=50)
    axs[0].plot(J_train[0:500], c=colors[5], label='JAC_0', alpha=0.9, linewidth=4, marker='s', markersize=18, markevery=50)
    axs[0].plot(MSE_train_5[0:500], c=colors[8], label='MSE_5', alpha=0.9, linewidth=4, marker='8', markersize=18, markevery=50)
    axs[0].plot(J_train_5[0:500], c=colors[1], label='JAC_5', alpha=0.9, linewidth=4, marker='*', markersize=18, markevery=50)
    axs[0].grid(True)
    axs[0].legend(loc='best', fontsize=38)
    axs[0].set_ylabel('Train Loss', fontsize=40)
    axs[0].set_xlabel('Epoch', fontsize=40)
    axs[0].tick_params(labelsize=40)

    # Test Loss
    axs[1].plot(x[0:500], MSE_test[0:500], c=colors[16], label='MSE_0 in MSE', alpha=0.9, linewidth=4, marker='o', markersize=18, markevery=50)
    axs[1].plot(x[0:500], J_test[0:500], c=colors[5],label='JAC_0 in MSE', alpha=0.9, linewidth=4, marker='s', markersize=18, markevery=50)
    axs[1].plot(x[0:500], MSE_test_5[0:500], c=colors[8], label='MSE_5 in MSE', alpha=0.9, linewidth=4, marker='8', markersize=18, markevery=50)
    axs[1].plot(x[0:500], J_test_5[0:500], c=colors[1],label='JAC_5 in MSE', alpha=0.9, linewidth=4, marker='*', markersize=18, markevery=50)
    axs[1].grid(True)
    axs[1].legend(loc='best', fontsize=38)
    axs[1].tick_params(labelsize=40)
    axs[1].set_ylabel('Test Loss', fontsize=40)
    axs[1].set_xlabel('Epoch', fontsize=40)

    tight_layout()
    savefig('../plot/loss_behavior.png', format='png', dpi=800, bbox_inches ='tight', pad_inches = 0.1)

    return


def plot_loss_MSE(MSE_train, MSE_test, model_name):
    fig, axs = subplots(1, figsize=(24, 12)) #, sharey=True
    fig.suptitle("Loss Behavior of MSE Loss", fontsize=24)
    
    colors = cm.tab20b(np.linspace(0, 1, 20))

    # Training Loss
    x = np.arange(0, MSE_train.shape[0])

    axs.plot(x[500:], MSE_train[500:], c=colors[15], label='Train Loss', alpha=0.9, linewidth=5)
    axs.plot(x[500:], MSE_test[500:], c=colors[1], label='Test Loss', alpha=0.9, linewidth=5)
    axs.grid(True)
    axs.legend(loc='best', fontsize=48)
    # axs.set_ylabel(r'$\mathcal{L}$', fontsize=24)
    axs.set_xlabel('Epochs', fontsize=48)
    axs.tick_params(labelsize=48)

    tight_layout()
    savefig('../plot/loss_behavior'+str(model_name)+ '.pdf', format='pdf', dpi=600, bbox_inches ='tight', pad_inches = 0.1)

    return


def plot_distribution(dyn_sys, dyn_sys_func, integration_time, model_name, init_state, time_step, model_path, neural_model_path):
    # call h(x) of dynamical system of interest
    dyn_system, dim = define_dyn_sys(dyn_sys)
    ti, tf = integration_time

    pdf_path = '../plot/dist_'+str(dyn_sys)+'all'+'_'+str(init_state.tolist())+'.jpg'
    is_copy1 = init_state
    is_copy2 = init_state
    # simulate true trajectory
    # Generate Training/Test/Multi-Step Prediction Data
    if (str(dyn_sys) == "henon") or (str(dyn_sys) == "baker") or (str(dyn_sys) == "tent_map"):
        
        true_data = torch.zeros(tf*int(1/time_step), dim)
        node_data = torch.zeros(tf*int(1/time_step), dim)
        jac_data = torch.zeros(tf*int(1/time_step), dim)
        
        for i in range(tf*int(1/time_step)):
            next_x = dyn_sys_func(init_state)
            true_data[i] = next_x
            init_state = next_x

        # Load the saved model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = create_NODE(device, dyn_sys= dyn_sys, n_nodes=dim,  n_hidden=64, T=time_step).double()
        model.load_state_dict(torch.load(model_path))
        model.eval()

        m = create_NODE(device, dyn_sys= dyn_sys, n_nodes=dim,  n_hidden=64, T=time_step).double()
        m.load_state_dict(torch.load(neural_model_path))
        m.eval()
        print("Finished Loading model")
        # simulate node trajectory

        for i in range(tf*int(1/time_step)):
            next_x = model(is_copy1.to(device).double())
            node_data[i] = next_x
            is_copy1 = next_x

        for i in range(tf*int(1/time_step)):
            nx = m(is_copy2.to(device).double())
            jac_data[i] = nx
            is_copy2 = nx

    else:
        true_data = simulate(dyn_system, 0, tf+1, init_state, time_step) # last 100 points are for testing
    
        # Load the saved model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = create_NODE(device, dyn_sys= dyn_sys, n_nodes=dim,  n_hidden=64, T=time_step).double()
        model.load_state_dict(torch.load(model_path))
        model.eval()

        m = create_NODE(device, dyn_sys= dyn_sys, n_nodes=dim,  n_hidden=64, T=time_step).double()
        m.load_state_dict(torch.load(neural_model_path))
        m.eval()
        print("Finished Loading model")
        # simulate node trajectory
        node_data = simulate(model, 0, tf+1, init_state.to(device).double(), time_step).detach().cpu()
        jac_data = simulate(m, 0, tf+1, init_state.to(device).double(), time_step).detach().cpu()

        fig, (ax1, ax2, ax3) = subplots(1,3, figsize=(24,14))

        ax2.scatter(node_data[:10000, 0], node_data[:10000, 2], color="turquoise", s=20, alpha=0.8)
        ax1.scatter(true_data[:10000, 0], true_data[:10000, 2], color="salmon", s=20, alpha=0.8)
        ax3.scatter(jac_data[:10000, 0], jac_data[:10000, 2], color="slateblue", s=20, alpha=0.8)
        
        ax1.xaxis.set_tick_params(labelsize=44)
        ax1.yaxis.set_tick_params(labelsize=44)
        ax2.xaxis.set_tick_params(labelsize=44)
        ax2.yaxis.set_tick_params(labelsize=44)
        ax3.xaxis.set_tick_params(labelsize=44)
        ax3.yaxis.set_tick_params(labelsize=44)
        # ax1.set_xlabel(r"$x_1$", fontsize=44)
        # ax1.set_ylabel(r"$x_4$", fontsize=44)
        # ax2.set_xlabel(r"$x_2$", fontsize=44)
        # ax2.set_ylabel(r"$x_4$", fontsize=44)
        # ax3.set_xlabel(r"$x_2$", fontsize=44)
        # ax3.set_ylabel(r"$x_4$", fontsize=44)
        tight_layout()
        path = '../plot/'+str(dyn_sys)+'_all'+'.jpg'
        fig.savefig(path, format='jpg', dpi=400)

    print(true_data[:10], true_data[:-10])
    print("mid", true_data[100])
    print(node_data[:10], node_data[:-10])
    print(jac_data[:10], jac_data[:-10])


    # plot

    fig, ax1 = subplots(1,figsize=(16,8)) #, sharey=True
    sns.set_theme()
    # ax1.histplot(true_data[:, 0], kde=True, stat="density")
    # ax1.histplot(node_data[:, 0], kde=True, stat="density")
    # ax1.hist(true_data[:1000, 3], bins=100, alpha=0.7, color=[0.25, 0.25, 0.25]) #density=True, 
    # ax2.hist(node_data[:1000, 3], bins=100, alpha=0.7, color="turquoise") #density=True, 
    # ax3.hist(jac_data[:1000, 3], bins=100, alpha=0.7, color="turquoise") #density=True, 


    # tent_map
    # ax1.hist(true_data[:100, 0], bins=50, alpha=0.5, color="salmon") #density=True, 
    # ax1.hist(node_data[:100, 0].detach().cpu(), bins=50, alpha=0.5, color="turquoise") #density=True, 
    # ax1.hist(jac_data[:100, 0].detach().cpu(), bins=50, alpha=0.5, color="slateblue") #density=True,


    # baker
    # ax1.hist(true_data[:80, 1], bins=50, alpha=0.5, color="salmon") #density=True, 
    # ax1.hist(node_data[:80, 1].detach().cpu(), bins=50, alpha=0.5, color="turquoise") #density=True, 
    # ax1.hist(jac_data[:80, 1].detach().cpu(), bins=50, alpha=0.5, color="slateblue") #density=True,


    # lorenz
    # ax1.hist(true_data[:2000, 2], bins=50, alpha=0.5, color="salmon") #density=True, 
    # ax1.hist(node_data[:2000, 2], bins=50, alpha=0.5, color="turquoise") #density=True, 
    # ax1.hist(jac_data[:2000, 2], bins=50, alpha=0.5, color="slateblue") #density=True, 


    # # rossler
    ax1.hist(true_data[:2000, 2], bins=50, alpha=0.5, range=[0.02, 0.05], color="salmon") #density=True, 
    ax1.hist(node_data[:2000, 2], bins=50, alpha=0.5, range=[0.02, 0.05], color="turquoise") #density=True, 
    ax1.hist(jac_data[:2000, 2], bins=50, alpha=0.5, range=[0.02, 0.05], color="slateblue") #density=True, 

    # # hyperchaos range=[-15, 5], 
    # print(true_data[100:110])
    # ax1.hist(np.log(true_data[:2000, 0]), bins=100, alpha=0.5, range=[-400, 65], density=True, color="salmon") #density=True, 
    # ax1.hist(np.log(node_data[:2000, 0]), bins=100, alpha=0.5, range=[-400, 65], density=True, color="turquoise") #density=True, 
    # ax1.hist(np.log(jac_data[:2000, 0]), bins=100, alpha=0.5, range=[-400, 65], density=True, color="slateblue") #density=True, 


    ax1.grid(True)
    ax1.legend(['rk4', 'MSE', 'JAC'], fontsize=30)
    ax1.xaxis.set_tick_params(labelsize=34)
    ax1.yaxis.set_tick_params(labelsize=34)
    # ax2.grid(True)
    # ax2.legend(['MSE'], fontsize=30)
    # ax2.xaxis.set_tick_params(labelsize=34)
    # ax2.yaxis.set_tick_params(labelsize=34)
    # ax3.grid(True)
    # ax3.legend(['JAC'], fontsize=30)
    # ax3.xaxis.set_tick_params(labelsize=34)
    # ax3.yaxis.set_tick_params(labelsize=34)
    tight_layout()
    savefig(pdf_path, format='jpg', dpi=400, bbox_inches ='tight', pad_inches = 0.1)
    return





if __name__ == '__main__':
    
    init_state=torch.tensor([0.1, 0.1, 0.1])
    dyn_system=rossler
    time_step=0.01
    tau = 10
    true_data = simulate(dyn_system, 0, 5, init_state, time_step) # last 100 points are for testing
    plot_data = np.zeros((true_data.shape[0]-tau, 1))

    for i in range(true_data.shape[0] - tau):
        plot_data[i] = np.correlate(true_data[0:tau,0], true_data[i:i+tau,0])/true_data.shape[0] - np.mean(true_data[:,0])**2

    figure(figsize=(20,10))
    plot(plot_data)
    # plot_acf(true_data[:, 0], lags=10,use_vlines=False) 
    # acorr(true_data[:,0], usevlines=False, normed=True, maxlags=50, lw=2)

    savefig("../plot/autocorr.jpg", format='jpg', dpi=400, bbox_inches ='tight', pad_inches = 0.1)
    
    
    '''device = "cuda" if torch.cuda.is_available() else "cpu"
    plot_3d_trajectory("lorenz", [0,200], "JAC_0", 0.01, torch.tensor([1.0, 1.0, -1.0]).to(device), device, comparison=True)'''

    '''loss_JAC = np.genfromtxt("../test_result/expt_lorenz/AdamW/0.01/loss_100_Jacobian.csv", delimiter=",", dtype=float)
    loss_MSE = np.genfromtxt("../test_result/expt_lorenz/AdamW/0.01/loss_100_MSE.csv", delimiter=",", dtype=float)
    print(np.mean(loss_JAC), np.mean(loss_MSE))''' # 0.0467645374708809 0.8517619795713108
    
    '''# init_state = torch.tensor([1.,1.,-1.])
    # init_state = torch.tensor([-8.6445e-01,-1.19299e+00,1.4918e+01])
    init_state = torch.tensor([0.1, 0.1, 0.1])
    # init_state= torch.tensor([0.1033, 0.1211, 0.0990, 0.0844])
    dyn_sys= "rossler"
    time_step= 0.01
    model_path = "../test_result/expt_"+str(dyn_sys)+"/AdamW/"+str(time_step)+'/'+'Rossler_MSE/model.pt'
    neural_model_path = "../test_result/expt_"+str(dyn_sys)+"/AdamW/"+str(time_step)+'/'+'Rossler_JAC/model.pt'
    plot_distribution(dyn_sys, baker, [0, 20], "MSE", init_state, 0.01, model_path, neural_model_path)'''

    # MSE_train =  np.genfromtxt("../test_result/expt_lorenz/AdamW/0.01/MSE_0/training_loss.csv", delimiter=",", dtype=float)
    # MSE_test =  np.genfromtxt("../test_result/expt_lorenz/AdamW/0.01/MSE_0/test_loss.csv", delimiter=",", dtype=float)
    # J_train =  np.genfromtxt("../test_result/expt_lorenz/AdamW/0.01/JAC_0/training_loss.csv", delimiter=",", dtype=float)
    # J_test =  np.genfromtxt("../test_result/expt_lorenz/AdamW/0.01/JAC_0/test_loss.csv", delimiter=",", dtype=float)
    # MSE_train_5 =  np.genfromtxt("../test_result/expt_lorenz/AdamW/0.01/MSE_5/training_loss.csv", delimiter=",", dtype=float)
    # MSE_test_5 =  np.genfromtxt("../test_result/expt_lorenz/AdamW/0.01/MSE_5/test_loss.csv", delimiter=",", dtype=float)
    # J_train_5 =  np.genfromtxt("../test_result/expt_lorenz/AdamW/0.01/JAC_5/training_loss.csv", delimiter=",", dtype=float)
    # J_test_5 =  np.genfromtxt("../test_result/expt_lorenz/AdamW/0.01/JAC_5/test_loss.csv", delimiter=",", dtype=float)
    
    # plot_loss(MSE_train, MSE_test, J_train, J_test, MSE_train_5, MSE_test_5, J_train_5, J_test_5)


    #----- test plot_3d_space() -----#
    '''time_step = 0.01
    model = "MSE_0"

    # in attractor/ call training point: load csv file, 90*int(1/time_step)
    traj = np.loadtxt("../test_result/expt_lorenz/AdamW/"+str(time_step)+'/'+str(model)+'/whole_traj.csv', delimiter=",", dtype=float)
    init_state = torch.tensor(traj[90*int(1/time_step), :]).double()
    print(init_state)

    # out of attractor
    # init_state = torch.tensor([1.,1.,-1.])
    # init_state = torch.tensor([9.39,9.51,28.06])

    # call model
    model_path = "../test_result/expt_lorenz/AdamW/"+str(time_step)+'/'+str(model)+'/model.pt'
    pdf_path = '../plot/three_phase_plot_'+str(model)+'_'+str(init_state.tolist())+'.png'

    device = "cuda" if torch.cuda.is_available() else "cpu"
    plot_3d_space(device, "lorenz", 0.01, "AdamW", True, [0, 100], False, 0, 50000, init_state, model_path, pdf_path)'''

    #LE_diff_rho(dyn_sys="lorenz", r_range=200, dr=5, time_step=0.01)


    #----- test bifurcation plot -----#
    '''# 1. initialize
    r_range=200
    r = np.arange(0, r_range, 0.1) # range of parameter rho
    n = range(100) # num of new initial condition
    param_list = list(itertools.product(r, n))

    # 2. run parallel
    with multiprocessing.Pool(processes=500) as pool:
        res = pool.map(compute_lorenz_bif, param_list)
        res = np.array(res)
        z_mins, z_maxes, time_avg = res[:, 0], res[:, 1], res[:, 2]
    
    # 3. create plot
    print("creating plot... ")
    print(time_avg.shape)
    
    r_axis = [el for el in r for i in range(len(n))]
    r_axis, time_avg = np.array(r_axis), np.array(time_avg)
    print(len(n))
    plot_lorenz_bif("lorenz", r_axis, z_mins, z_maxes, len(n))
    plot_lorenz_time_avg("lorenz", r_axis, time_avg, len(n))

    # 4. Check property
    fixed_point = []
    ergodic_point = []
    for i in range(0, len(r_axis), 100):
        if (np.abs(z_maxes[i] - z_mins[i])) < 1e-7:
            print("fixed point at rho=", r_axis[i])
            fixed_point.append([r_axis[i], z_maxes[i]])
        if (np.abs(z_maxes[i+10] - z_maxes[i+90])) < 1e-7:
            print("ergodic at rho=", r_axis[i])
            ergodic_point.append([r_axis[i], z_maxes[i+10]])
    np.savetxt('../test_result/expt_lorenz/'+ "fixed_point.csv", np.asarray(fixed_point), delimiter=",")
    np.savetxt('../test_result/expt_lorenz/'+ "ergodic_point.csv", np.asarray(ergodic_point), delimiter=",")'''