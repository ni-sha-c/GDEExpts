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




if __name__ == '__main__':

    dt = 0.01
    integration = 300
    len_integration = integration*int(1/dt)
    tau = 500
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # init = torch.tensor([1., 1., -1.]).to(device)
    init = torch.tensor([14.9440, 13.9801, 36.6756]).to(device)

    trans = 0
    len_trans = trans*int(1/dt)

    time = torch.arange(0, trans+integration+tau+1, dt)
    corr_list = []

    # simulate true
    traj = torchdiffeq.odeint(lorenz, init, time, method='rk4', rtol=1e-8)[len_trans:].cpu()

    # savefig
    fig, ax = subplots(figsize=(36,24))
    pdf_path = '../plot/corr_all'+'_'+str(torch.round(init, decimals=4).tolist())+'.png'
    colors = cm.rainbow(np.linspace(0, 1, 5))
    c = 0
    tau_x = np.linspace(0, tau, tau*int(1/dt))
    node_corr_list = np.zeros((4,tau*int(1/dt)))

    for model_name in ["MSE_0", "MSE_5", "JAC_0", "JAC_5"]:
        print(model_name)
        # simulate Neural ODE
        model_path = "../test_result/expt_lorenz/AdamW/"+str(dt)+'/'+str(model_name)+'/model.pt'

        # Load the saved model
        model = ODE_Lorenz().to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print("Finished Loading model")

        # simulate NODE trajectory
        data = torchdiffeq.odeint(model, init, time, method='rk4', rtol=1e-8)[len_trans:].detach().cpu()
        print("d", data[:10])

        # x(0 : t)
        base_traj_x = np.array(traj[:len_integration, 0])
        node_base_traj_x = np.array(data[:len_integration, 0])
    
        # Iterate over from 0 ... tau-1
        for i in range(tau*int(1/dt)):

            # # z(0 + Tau: t + Tau)
            len_tau = i #i*int(1/dt)
            tau_traj_z = np.array(data[len_tau: len_tau+len_integration, 0])

            # # compute corr between
            node_corr = np.correlate(tau_traj_z, node_base_traj_x)/len_integration - np.mean(np.array(data[:, 0]))*np.mean(np.array(data[:, 0]))
            node_corr_list[c, i] = node_corr[0]
            if i % 1000 ==0:
                print(i, node_corr)

        c += 1
        print("c", c)


    # Iterate over from 0 ... tau-1
    for i in range(tau*int(1/dt)):

        # # z(0 + Tau: t + Tau)
        len_tau = i #i*int(1/dt)
        tau_traj_z = np.array(traj[len_tau: len_tau+len_integration, 0])

        # # compute corr between
        corr = np.correlate(tau_traj_z, base_traj_x)/len_integration - np.mean(np.array(traj[:, 0]))*np.mean(np.array(traj[:, 0]))
        corr_list.append(corr[0])
        if i % 1000 == 0:
            print(i, corr)

    node_corr_list = np.array(node_corr_list)
    corr_list = np.array(corr_list)
    ax.plot(tau_x[:1000], node_corr_list[0, :1000], color=colors[0], marker='o', linewidth=6, markersize=28, markevery=50, alpha=0.8, label='MSE_0')
    ax.fill_between(tau_x[:1000], node_corr_list[0, :1000] - node_corr_list[0, :1000].std(), node_corr_list[0, :1000] + node_corr_list[0, :1000].std(), color=colors[0], alpha=0.3)

    ax.plot(tau_x[:1000], node_corr_list[1, :1000], color=colors[1], marker='*', linewidth=6, markersize=28, markevery=50, alpha=0.8, label='MSE_5')
    ax.fill_between(tau_x[:1000], node_corr_list[1, :1000] - node_corr_list[1, :1000].std(), node_corr_list[1, :1000] + node_corr_list[1, :1000].std(), color=colors[1], alpha=0.3)

    ax.plot(tau_x[:1500], node_corr_list[2, :1500], color=colors[2], marker='s', linewidth=6, markersize=28, markevery=50, alpha=0.8, label='JAC_0')
    ax.fill_between(tau_x[:1500], node_corr_list[2, :1500] - node_corr_list[2, :1500].std(), node_corr_list[2, :1500] + node_corr_list[2, :1500].std(), color=colors[2], alpha=0.3)

    ax.plot(tau_x[:1500], node_corr_list[3, :1500], color=colors[3], marker='8', linewidth=6, markersize=28, markevery=50, alpha=0.8, label='JAC_5')
    ax.fill_between(tau_x[:1500], node_corr_list[3, :1500] - node_corr_list[3, :1500].std(), node_corr_list[3, :1500] + node_corr_list[3, :1500].std(), color=colors[3], alpha=0.3)

    ax.plot(tau_x[:1500], corr_list[:1500], color=colors[c], marker='>', linewidth=6, markersize=28, markevery=50, alpha=0.8, label='rk4')
    ax.fill_between(tau_x[:1500], corr_list[:1500] - corr_list[:1500].std(), corr_list[:1500] + corr_list[:1500].std(), color=colors[c], alpha=0.3)

    ax.grid(True)
    ax.set_xlabel(r"$\tau$", fontsize=40)
    ax.set_ylabel(r"$C_{x,x}(\tau)$", fontsize=40)
    ax.tick_params(labelsize=38)
    ax.legend(loc='best', fontsize=38)
    tight_layout()

    fig.savefig(pdf_path, format='png', dpi=400)


    '''# Compute Fourier
    rk4_fourier = scipy.fft.rfft(corr_list)
    node_fourier = scipy.fft.rfft(node_corr_list)

    N = len(rk4_fourier)
    sampling_rate = 1/(dt)
    T = N/sampling_rate
    freq = np.arange(N)/T

    # get the one sided spectrum
    n_oneside = N //2
    # get one side frequency
    f_oneside = freq[:n_oneside]
    
    # 5. plot Fourier
    fig, ax = subplots(figsize=(36,12))
    ax.plot(f_oneside, np.log(np.abs(rk4_fourier[:n_oneside])), color=(0.25, 0.25, 0.25), marker='o', linewidth=4, alpha=1)
    ax.plot(f_oneside, np.log(np.abs(node_fourier[:n_oneside])), color="slateblue", marker='o', linewidth=4, alpha=1)

    ax.grid(True)
    #ax.set_xlabel(r"$Frequency$", fontsize=24)
    ax.set_ylabel(r"$F(C_{x,z}(\tau))$", fontsize=24)
    ax.tick_params(labelsize=24)
    ax.legend(["rk4", "Neural ODE"], fontsize=24)

    # path = '../plot/'+'Fourier.png'
    pdf_path = '../plot/updated_fourier_'+str(model_name)+'_'+str(torch.round(init, decimals=4).tolist())+'.pdf'
    fig.savefig(pdf_path, format='pdf', dpi=400)



    #len_tau = torch.linspace(0, tau, tau.shape[0])
    #plot_correlation("lorenz", len_tau, val, node_val, t)

        # corr = scipy.signal.correlate( base_traj_x, tau_traj_z, mode="same") #/ len_tau #- np.mean(tau_traj_z)*np.mean(base_traj_x)

        # lags = scipy.signal.correlation_lags(base_traj_x.shape[0], tau_traj_z.shape[0])

        # print(i, np.abs(np.mean(corr))- np.mean(tau_traj_z)*np.mean(base_traj_x))
        # lag = lags[np.argmax(corr)]
        #print(i, corr)
        # print(np.log(np.abs(lag / len_tau)), "\n")'''

    


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

    