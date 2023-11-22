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
from test_metrics import *

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





def corr_plot_rk4(args):

    i, t, dt, tau, init = args

    # ----- rk4 ----- #

    # Generate traj(0 ~ 50+tau)
    correlation = []
    time = torch.arange(0, t+tau[i]+1, dt)
    num_t = time.shape[0]
    traj = torchdiffeq.odeint(lorenz, init, time, method='rk4', rtol=1e-8)

    print(tau[i], "time:", time.shape)

    #z(t+Tau)
    z_tau = traj[(tau[i]+1)*int(1/dt):, 1]
    #x(t)
    x = traj[:t*int(1/dt), 0]
    mean_xz = torch.inner(x,z_tau) / (x.shape[0])
    mean_xz = mean_xz - torch.mean(x)*torch.mean(z_tau)
    #print(i, "before abs:", mean_xz)
    #mean_xz = torch.abs(mean_xz)
    correlation.append(mean_xz)

    return correlation



def corr_plot_node(args):
    # ----- node ----- #
    device, model, i, t, dt, tau, init = args
    node_correlation = []
    time = torch.arange(0, t+tau[i]+1, dt).to(device)
    t_eval_point = torch.linspace(0, dt, 2).to(device)
    num_t = time.shape[0]

    x = init.to(device)
    temp = torchdiffeq.odeint(model, x, time, method='rk4', rtol=1e-8)

    # Compute x*z
    print("temp", temp.shape)
    node_z = temp[(tau[i]+1)*int(1/dt):, 1]
    node_x = temp[:t*int(1/dt), 0]

    node_mean_xz = torch.inner(node_x,node_z) / (node_x.shape[0])
    node_mean_xz = node_mean_xz - torch.mean(node_x)*torch.mean(node_z)
    print(i, "before node abs:", node_mean_xz)
    #node_mean_xz[node_mean_xz < 0] = 0.
    #node_mean_xz = torch.abs(node_mean_xz)
    node_correlation.append(node_mean_xz.detach().cpu())

    return node_correlation



def plot_correlation(dyn_sys, tau, val, node_val, t):
    fig, ax = subplots(figsize=(36,12))
    ax.semilogy(tau, val, color=(0.25, 0.25, 0.25), marker='o', linewidth=4, alpha=0.8)
    ax.semilogy(tau, node_val, color="slateblue", marker='o', linewidth=4, alpha=0.8)

    ax.grid(True)
    ax.set_xlabel(r"$\tau$", fontsize=24)
    ax.set_ylabel(r"$C_{x,z}(\tau)$", fontsize=24)
    ax.tick_params(labelsize=24)
    ax.legend(["rk4", "Neural ODE"], fontsize=24)

    path = '../plot/'+'correlation_'+str(t)+'.svg'
    fig.savefig(path, format='svg', dpi=400)
    return




if __name__ == '__main__':

    dt = 0.01
    integration = 100
    len_integration = integration*int(1/dt)
    tau = 500
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #init = torch.tensor([1., 1., -1.]).to(device)
    init = torch.tensor([-8.6445e-01,-1.19299e+00,1.4918e+01]).to(device)

    trans = 0
    len_trans = trans*int(1/dt)

    time = torch.arange(0, trans+integration+tau+1, dt)
    corr_list, node_corr_list = [], []

    # simulate true
    traj = torchdiffeq.odeint(lorenz, init, time, method='rk4', rtol=1e-8)[len_trans:].cpu()

    # simulate Neural ODE
    model_name = "JAC_0"
    model_path = "../test_result/expt_lorenz/AdamW/"+str(dt)+'/'+str(model_name)+'/model.pt'
    pdf_path = '../plot/corr_'+str(model_name)+'_'+str(torch.round(init, decimals=4).tolist())+'.pdf'
    # Load the saved model
    model = ODE_Lorenz().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("Finished Loading model")
    # simulate NODE trajectory
    data = torchdiffeq.odeint(model, init, time, method='rk4', rtol=1e-8)[len_trans:].detach().cpu()

    # x(0 : t)
    base_traj_x = np.array(traj[:len_integration, 0])
    node_base_traj_x = np.array(data[:len_integration, 0])
    
    # Iterate over from 0 ... tau-1
    for i in range(tau*int(1/dt)):

        # # z(0 + Tau: t + Tau)
        len_tau = i #i*int(1/dt)
        tau_traj_z = np.array(traj[len_tau: len_tau+len_integration, 2])

        # # compute corr between
        corr = np.correlate(tau_traj_z, base_traj_x)/len_integration - np.mean(np.array(traj[:, 0]))*np.mean(np.array(traj[:, 2]))
        corr_list.append(corr)
        print(i, corr)

    # Iterate over from 0 ... tau-1
    for i in range(tau*int(1/dt)):

        # # z(0 + Tau: t + Tau)
        len_tau = i #i*int(1/dt)
        tau_traj_z = np.array(data[len_tau: len_tau+len_integration, 2])

        # # compute corr between
        node_corr = np.correlate(tau_traj_z, node_base_traj_x)/len_integration - np.mean(np.array(data[:, 0]))*np.mean(np.array(data[:, 2]))
        node_corr_list.append(node_corr)
        print(i, node_corr)


    tau_x = np.linspace(0, tau, tau*int(1/dt))
    corr_list = np.array(corr_list)
    node_corr_list = np.array(node_corr_list)

    # savefig
    fig, ax = subplots(figsize=(36,12))
    ax.plot(tau_x, corr_list, color=(0.25, 0.25, 0.25), marker='o', linewidth=4, alpha=0.8)
    ax.plot(tau_x, node_corr_list, color="slateblue", marker='o', linewidth=4, alpha=0.8)

    ax.grid(True)
    ax.set_xlabel(r"$\tau$", fontsize=24)
    ax.set_ylabel(r"$C_{x,x}(\tau)$", fontsize=24)
    ax.tick_params(labelsize=24)
    ax.legend(["rk4", "Neural ODE"], fontsize=24)

    fig.savefig(pdf_path, format='pdf', dpi=400)


    # Compute Fourier
    rk4_fourier = scipy.fft.rfft(corr_list)
    node_fourier = scipy.fft.rfft(node_corr_list)

    N = len(rk4_fourier)
    sampling_rate = 1/(2*np.pi)
    T = N/sampling_rate
    freq = np.arange(N)/T

    # get the one sided spectrum
    n_oneside = N //2
    # get one side frequency
    f_oneside = freq[:n_oneside]
    
    # 5. plot Fourier
    fig, ax = subplots(figsize=(36,12))
    ax.plot(f_oneside, np.abs(rk4_fourier[:n_oneside]), color=(0.25, 0.25, 0.25), marker='o', linewidth=4, alpha=1)
    ax.plot(f_oneside, np.abs(node_fourier[:n_oneside]), color="slateblue", marker='o', linewidth=4, alpha=1)

    ax.grid(True)
    #ax.set_xlabel(r"$Frequency$", fontsize=24)
    ax.set_ylabel(r"$F(C_{x,z}(\tau))$", fontsize=24)
    ax.tick_params(labelsize=24)
    ax.legend(["rk4", "Neural ODE"], fontsize=24)

    # path = '../plot/'+'Fourier.png'
    pdf_path = '../plot/fourier_'+str(model_name)+'_'+str(torch.round(init, decimals=4).tolist())+'.pdf'
    fig.savefig(pdf_path, format='pdf', dpi=400)



    #len_tau = torch.linspace(0, tau, tau.shape[0])
    #plot_correlation("lorenz", len_tau, val, node_val, t)

        # corr = scipy.signal.correlate( base_traj_x, tau_traj_z, mode="same") #/ len_tau #- np.mean(tau_traj_z)*np.mean(base_traj_x)

        # lags = scipy.signal.correlation_lags(base_traj_x.shape[0], tau_traj_z.shape[0])

        # print(i, np.abs(np.mean(corr))- np.mean(tau_traj_z)*np.mean(base_traj_x))
        # lag = lags[np.argmax(corr)]
        #print(i, corr)
        # print(np.log(np.abs(lag / len_tau)), "\n")

    


    # # ----- correlation plot ----- #

    
    # #1. initialize
    # t= 100
    # dt= 0.01
    # tf = 1000
    # tau = torch.arange(0, tf, 100)
    # init = torch.rand(3)
    # num_processes = 5
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # multiprocessing.set_start_method('spawn')
    
    # #1-1. Load the saved model
    # model = ODE_Lorenz().to(device)
    # path = "../test_result/expt_lorenz/AdamW/"+str(dt)+'/'+'model_J_0.pt'
    # model.load_state_dict(torch.load(path))
    # model.eval()
    # print("Finished Loading model")

    # # 2. run parallel
    # with multiprocessing.Pool(processes=num_processes) as pool:
    #     res = pool.map(corr_plot_rk4, [(i, t, dt, tau, init) for i in range(len(tau))])
    #     node_res = pool.map(corr_plot_node, [(device, model, i, t, dt, tau, init) for i in range(len(tau))]) #starmap

    #     rk4_val = np.array(res)
    #     node_val = np.array(node_res)

    