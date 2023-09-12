import torch
import numpy as np
import torchdiffeq
from matplotlib.pyplot import * 
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
sys.path.append('..')

from examples.Lorenz import lorenz
from src import NODE_solve_Lorenz as sol


def plot_time_average(time_step, optim_name, tau):
    '''plot |avg(z_tau) - avg(z_t)|'''

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # create z_n trajectory
    init = torch.Tensor([ -8., 7., 27.])
    t_eval_point = torch.arange(0, tau, time_step)
    sol_tau = torchdiffeq.odeint(lorenz, init, t_eval_point, method='rk4', rtol=1e-8) 
    time_avg_tau = torch.mean(sol_tau[1:, 2])
    print("mean: ", time_avg_tau)

    # initialize
    n_val = torch.arange(1, tau+1)
    diff_time_avg = torch.zeros(n_val.shape)


    with torch.no_grad():
        # load the saved model
        model = sol.create_NODE(device, n_nodes=3, T=time_step).double()
        path = "expt_lorenz/"+optim_name+"/"+str(time_step)+'/'+'model.pt'
        model.load_state_dict(torch.load(path))
        model.eval()

        for i, n in enumerate(n_val):
            # create data
            t = torch.arange(0, n, time_step)
            sol_n = torchdiffeq.odeint(lorenz, init, t, method='rk4', rtol=1e-8) 

            # compute y
            X = sol_n.to(device).double()
            y = model(X)

            # calculate average and difference
            y = y.detach()
            print("i: ", i, "mean: ", torch.mean(y[:, 2]))
            diff_time_avg[i] = torch.abs(torch.mean(y[:, 2]) - time_avg_tau)
            print("i: ", i, "diff: ", diff_time_avg[i], "\n")
            torch.cuda.empty_cache()


    fig, ax = subplots()
    ax.semilogy(diff_time_avg.detach().numpy(), ".", ms = 10.0)
    ax.grid(True)
    ax.set_xlabel(r"$n \times \delta t$", fontsize=20)
    ax.set_ylabel(r"log $|\bar z(\tau) - \bar z|$", fontsize=20)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    tight_layout()
    fig.savefig("time_avg_convergence.png")


#def plot_lyap_expts():


##### test run #####
plot_time_average(time_step=0.01, optim_name='AdamW', tau=100)