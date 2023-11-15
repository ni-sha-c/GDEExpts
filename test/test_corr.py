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
    tau = 600
    device = "cuda" if torch.cuda.is_available() else "cpu"

    init = torch.rand(3).to(device)
    #init = torch.tensor([9.390855789184570312e+00,9.506474494934082031e+00,2.806442070007324219e+01]).to(device)

    trans = 80
    len_trans = trans*int(1/dt)

    time = torch.arange(0, trans+integration+tau+1, dt)
    traj = torchdiffeq.odeint(lorenz, init, time, method='rk4', rtol=1e-8)[len_trans:].cpu()

    # x(0 : t)
    base_traj_x = np.array(traj[:len_integration, 0])
    #base_traj_x = np.array(traj[:, 0])

    # Iterate over from 0 ... tau-1
    for i in range(tau):
        # z(0 + Tau: t + Tau)
        len_tau = i*int(1/dt)
        tau_traj_z = np.array(traj[len_tau: len_tau+len_integration, 2])

        # compute corr between
        corr = np.abs(np.dot(tau_traj_z, base_traj_x))/len_integration #- np.mean(tau_traj_z)*np.mean(base_traj_x)
        print(i, corr)

        # corr = scipy.signal.correlate( base_traj_x, tau_traj_z, mode="same") / len_tau - np.mean(tau_traj_z)*np.mean(base_traj_x)

        # lags = scipy.signal.correlation_lags(base_traj_x.shape[0], tau_traj_z.shape[0])
        # print(i, corr, lags)
        # lag = lags[np.argmax(corr)]
        # #print(i, corr)
        # print(np.log(np.abs(lag / len_tau)), "\n")

    


    # ----- correlation plot ----- #
    
    #1. initialize
    t= 100
    dt= 0.01
    tf = 1000
    tau = torch.arange(0, tf, 100)
    init = torch.rand(3)
    num_processes = 5
    device = "cuda" if torch.cuda.is_available() else "cpu"
    multiprocessing.set_start_method('spawn')
    
    #1-1. Load the saved model
    model = ODE_Lorenz().to(device)
    path = "../test_result/expt_lorenz/AdamW/"+str(dt)+'/'+'model_J_0.pt'
    model.load_state_dict(torch.load(path))
    model.eval()
    print("Finished Loading model")

    # 2. run parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        res = pool.map(corr_plot_rk4, [(i, t, dt, tau, init) for i in range(len(tau))])
        node_res = pool.map(corr_plot_node, [(device, model, i, t, dt, tau, init) for i in range(len(tau))]) #starmap

        rk4_val = np.array(res)
        node_val = np.array(node_res)

    # 3. compute Fourier
    rk4_fourier = scipy.fft.rfft(rk4_val)
    node_fourier = scipy.fft.rfft(node_val)
    normalize = rk4_val.shape[0] / 2
    print(normalize)

    freq_axis = scipy.fft.rfftfreq(rk4_val.shape[0], 1/0.01)
        
    # 4. plot correlation
    print("initial point:", init)
    print(rk4_val)
    print(node_val)
    len_tau = torch.linspace(0, tf, tau.shape[0])
    plot_correlation("lorenz", len_tau, np.abs(rk4_val), np.abs(node_val), t)
    
    # 5. plot Fourier
    fig, ax = subplots(figsize=(36,12))
    ax.plot(np.abs(rk4_fourier/normalize), color=(0.25, 0.25, 0.25), marker='o', linewidth=4, alpha=1)
    ax.plot(np.abs(node_fourier/normalize), color="slateblue", marker='o', linewidth=4, alpha=1)

    ax.grid(True)
    #ax.set_xlabel(r"$Frequency$", fontsize=24)
    ax.set_ylabel(r"$F(C_{x,z}(\tau))$", fontsize=24)
    ax.tick_params(labelsize=24)
    ax.legend(["rk4", "Neural ODE"], fontsize=24)

    path = '../plot/'+'Fourier.png'
    fig.savefig(path, format='png', dpi=400)

