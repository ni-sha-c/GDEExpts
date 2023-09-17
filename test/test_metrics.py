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


def relative_error(optim_name, time_step):
    # Load training_error csv file
    training_err = torch.from_numpy(np.genfromtxt("expt_lorenz/"+optim_name+"/"+str(time_step)+"/"+"training_loss.csv", delimiter=","))
    test_err = torch.from_numpy(np.genfromtxt("expt_lorenz/"+optim_name+"/"+str(time_step)+"/"+"test_loss.csv", delimiter=","))

    # Compute supereme x
    t_n = 100
    tran = 100
    X, Y, X_test, Y_test = sol.create_data(0, 120, 
                                torch.Tensor([ -8., 7., 27.]), 12001, 
                                n_train=10000, n_test=1800, n_nodes=3, n_trans=tran)
    x_norm = torch.zeros(X.shape[0])
    for i in range(X.shape[0]):
        x_norm[i] = torch.linalg.norm(X[i])
    
    x_norm_sup = torch.max(x_norm)

    # Compute relative error
    train_err_plot = training_err/x_norm_sup
    test_err_plot = test_err/x_norm_sup

    # Plot relative error
    fig, ax = subplots()
    ax.semilogy(train_err_plot.detach().numpy(), ".", ms = 8.0)
    ax.semilogy(test_err.detach().numpy(), ".", ms = 8.0)
    ax.grid(True)
    ax.set_xlabel("epoch", fontsize=20)
    ax.set_ylabel(r"$\frac{L(x)}{sup_{x}\|x\|^2}$", fontsize=20)
    ax.legend(['Relative training error', 'Relative test error'])
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    tight_layout()
    fig.savefig("relative_error.png")

    return


def plot_time_average(init, time_step, optim_name, tau, component):
    '''plot |avg(z_tau) - avg(z_t)|'''

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transition_phase = 0

    # Initialize integration times
    n_val = torch.arange(1, tau+1)
    diff_time_avg_rk4 = torch.zeros(n_val.shape)
    diff_time_avg_node = torch.zeros(n_val.shape)

    # Create phi_rk4[:, component]
    t_eval_point = torch.arange(0, tau, time_step)
    sol_tau = torchdiffeq.odeint(lorenz, init, t_eval_point, method='rk4', rtol=1e-8) 
    time_avg_tau_rk4 = torch.mean(sol_tau[transition_phase:, component])
    print("mean rk4: ", time_avg_tau_rk4)

    for i, n in enumerate(n_val):
        print(i)
        # create data
        t = torch.arange(0, n, time_step)
        sol_n = torchdiffeq.odeint(lorenz, init, t, method='rk4', rtol=1e-8) 

        # compute time average at integration time = n
        time_avg_n = torch.mean(sol_n[transition_phase:, component])

        # calculate average and difference
        diff_time_avg_rk4[i] = torch.abs(time_avg_n - time_avg_tau_rk4)



    # Load the saved model
    model = sol.create_NODE(device, n_nodes=3, T=time_step).double()
    path = "expt_lorenz/"+optim_name+"/"+str(time_step)+'/'+'model.pt'
    model.load_state_dict(torch.load(path))
    model.eval()

    # Create phi_NODE[:, component]
    x = init
    x = x.to(device)
    phi_tau = torch.zeros(tau, 3).to(device)

    for i in range(tau):
        phi_tau[i] = x # shape [3]
        cur_pred = model(x.double())
        x = cur_pred

    time_avg_tau_node = torch.mean(phi_tau[transition_phase:, component])
    print("model mean: ", time_avg_tau_node)

    for i, n in enumerate(n_val):
        print(i)
        temp = torch.zeros(n, 3)
        # create data
        for j in range(n):
            temp[j] = x # shape [3]
            cur_pred = model(x.double())
            x = cur_pred

        # compute time average at integration time = n
        time_avg_n = torch.mean(temp[transition_phase:, component])

        # calculate average and difference
        diff_time_avg_node[i] = torch.abs(time_avg_n - time_avg_tau_node)


    fig, ax = subplots()
    ax.semilogy(diff_time_avg_node.detach().numpy()[transition_phase:], ".", ms = 10.0)
    ax.semilogy(diff_time_avg_rk4.detach().numpy()[transition_phase:], ".", ms = 10.0)
    ax.grid(True)
    ax.set_xlabel(r"$n \times \delta t$", fontsize=20)
    ax.set_ylabel(r"log $|\bar z(\tau) - \bar z|$", fontsize=20)
    ax.legend(['NODE', 'rk4'])
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    tight_layout()
    fig.savefig("time_avg_convergence.png")

    return



def multi_step_pred_err(x, optim_name, time_step, integration_time, component):
    ''' Generate Plot for |φ_TRUE^t(x0) - φ_NODE^t(x0)| '''
    # Initialize Tensor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pred_traj = torch.zeros(integration_time, 3).to(device)
    err = torch.zeros(integration_time, 3).to(device)
    x = x.to(device)

    # Load the model
    model = sol.create_NODE(device, n_nodes=3, T=time_step).double()
    path = "expt_lorenz/"+optim_name+"/"+str(time_step)+'/'+'model.pt'
    model.load_state_dict(torch.load(path))
    model.eval()

    # Generate pred_traj, φ_NODE^t(x0) 
    for i in range(integration_time):
        pred_traj[i] = x # shape [3]
        cur_pred = model(x.double())
        x = cur_pred

    # Compute true_traj, φ_TRUE^t(x0)
    t_eval_point = torch.arange(0, integration_time, step=time_step).to(device)
    true_traj = torchdiffeq.odeint(lorenz, x, t_eval_point, method='rk4', rtol=1e-9) 
    
    # Compute difference |φ_TRUE^t(x0) - φ_NODE^t(x0)|
    for i in range(integration_time):
        err[i] = torch.abs(pred_traj[i, component] - true_traj[i, component])

    fig, ax = subplots()
    plot_x = torch.linspace(0, integration_time * time_step, integration_time)
    ax.semilogy(plot_x[100:], err.detach().cpu().numpy()[100:], ".", ms = 10.0)
    ax.grid(True)
    ax.set_xlabel(r"$n * \delta t$", fontsize=20)
    ax.set_ylabel(r"$|\phi_{TRUE}^t(x) - \phi_{NODE}^t(x)|$", fontsize=20)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    tight_layout()
    fig.savefig("multi_step_error_"+str(time_step) +".png")

    return



def perturbed_multi_step_error(method, x, eps, optim_name, time_step, integration_time):
    ''' Generate plot for ∥φt(x0) - φt(x0 + εv0)∥ 
        params: method = {'NODE', 'rk4'} 
                x = double (initial point)
        returns: plot '''

    # Initialize Tensor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    unperturbed = torch.zeros(integration_time, 3).to(device)
    perturbed = torch.zeros(integration_time, 3).to(device)
    diff = torch.zeros(integration_time).to(device)

    # Generate random norm 1 vector
    random_vec = torch.randn(3)
    norm = torch.linalg.norm(random_vec)
    v0 = random_vec / norm
    x_perturbed = x + eps*v0
    x = x.to(device)
    x_perturbed = x_perturbed.to(device)

    if method == "NODE":
        # Load the model
        model = sol.create_NODE(device, n_nodes=3, T=time_step).double()
        path = "expt_lorenz/"+optim_name+"/"+str(time_step)+'/'+'model.pt'
        model.load_state_dict(torch.load(path))
        model.eval()

        # Generate pred_traj, φt(x0) 
        for i in range(integration_time):
            unperturbed[i] = x # shape [3]
            cur_pred = model(x.double())
            x = cur_pred

        # Compute perturbed trajectory, φt(x0 + εv0) 
        for i in range(integration_time):
            perturbed[i] = x_perturbed # shape [3]
            cur_pred = model(x_perturbed.double())
            x_perturbed = cur_pred
        
        # Compute difference ∥φt(x0) - φt(x0 + εv0)∥
        for i in range(integration_time):
            diff[i] = torch.linalg.norm(unperturbed[i] - perturbed[i])

    else:
        t_eval_point = torch.arange(0, integration_time, step=time_step).to(device)

        # Generate pred_traj, φt(x0)
        unperturbed = torchdiffeq.odeint(lorenz, x, t_eval_point, method=method, rtol=1e-9) 

        # Compute perturbed trajectory, φt(x0 + εv0)
        perturbed = torchdiffeq.odeint(lorenz, x_perturbed, t_eval_point, method=method, rtol=1e-9) 

        # Compute difference ∥φt(x0) - φt(x0 + εv0)∥
        for i in range(integration_time):
            diff_i = unperturbed[i] - perturbed[i]
            diff[i] = torch.linalg.norm(diff_i)

    # plot after transition time (real time = 1)
    fig, ax = subplots()
    plot_x = torch.linspace(0, integration_time * time_step, integration_time)
    ax.semilogy(plot_x[100:], diff.detach().cpu().numpy()[100:], ".", ms = 10.0)
    ax.grid(True)
    ax.set_xlabel(r"$n * \delta t$", fontsize=20)
    ax.set_ylabel(r"$\| \phi^t(x) - \phi^t(x + \epsilon v) \|$", fontsize=20)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    tight_layout()
    fig.savefig(str(method) + "_perturbed_multi_step_error.png")

    return

#def plot_lyap_expts():


##### test run #####

torch.set_printoptions(sci_mode=True, precision=10)
x = torch.randn(3)
eps = 1e-6
relative_error(optim_name="AdamW", time_step=0.01)

#plot_time_average(x, time_step=0.01, optim_name='AdamW', tau=300, component=2)

#perturbed_multi_step_error("rk4", x, eps, optim_name="AdamW", time_step=0.01, integration_time=1500)
#perturbed_multi_step_error("NODE", x, eps, optim_name="AdamW", time_step=0.01, integration_time=1500)

# multi_step_pred_err(x, optim_name="AdamW", time_step=0.01, integration_time=1000, component=0)