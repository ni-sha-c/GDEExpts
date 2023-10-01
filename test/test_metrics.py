import torch
import numpy as np
import torchdiffeq
from matplotlib.pyplot import * 
from scipy import stats
import torch.autograd.functional as F
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
sys.path.append('..')

from examples.Brusselator import *
from examples.Lorenz import *
from examples.Lorenz_periodic import *
from examples.Sin import *
from examples.Tent_map import *
from src import NODE_solve as sol

# TODO: Make sure that all of function starts from calling saved model. So that # it can also be used outside of training loop


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
    fig.savefig("../plot/relative_error.png")

    return



def plot_time_average(init, dyn_sys, time_step, optim_name, tau, component):
    '''plot |avg(z_tau) - avg(z_t)|'''

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transition_phase = 0

    # Initialize integration times
    n_val = torch.arange(1, tau+1)
    diff_time_avg_rk4 = torch.zeros(n_val.shape)
    diff_time_avg_node = torch.zeros(n_val.shape)

    # Load the saved model
    model = sol.create_NODE(device, dyn_sys=dyn_sys, n_nodes=3, n_hidden=64, T=time_step).double()
    path = "expt_"+str(dyn_sys)+"/"+optim_name+"/"+str(time_step)+'/'+'model.pt'
    model.load_state_dict(torch.load(path))
    model.eval()

    #----- Create phi_rk4[:, component] -----#
    t_eval_point = torch.arange(0, tau, time_step)
    iters = t_eval_point.shape[0]
    sol_tau = torchdiffeq.odeint(lorenz, init, t_eval_point, method='rk4', rtol=1e-8) 
    time_avg_tau_rk4 = torch.mean(sol_tau[:, component])

    for i, n in enumerate(n_val):
        # create data
        t = torch.arange(0, n, time_step)
        sol_n = torchdiffeq.odeint(lorenz, init, t, method='rk4', rtol=1e-8) 

        # compute time average at integration time = n
        time_avg_rk4 = torch.mean(sol_n[:, component])

        # calculate average and difference
        diff_time_avg_rk4[i] = torch.abs(time_avg_rk4 - time_avg_tau_rk4)

    print("mean rk4: ", time_avg_tau_rk4)
    print("sanity check rk4: ", diff_time_avg_rk4[-1])


    #----- Create phi_NODE[:, component] -----#
    x = init
    x = x.to(device)
    phi_tau = torch.zeros(tau, 3).to(device)

    for i in range(tau):
        phi_tau[i] = x # shape [3]
        cur_pred = model(x.double())
        x = cur_pred
    time_avg_tau_node = torch.mean(phi_tau[:, component])

    for i, n in enumerate(n_val):
        temp = torch.zeros(n, 3).to(device)
        x = init.to(device)
        # create data
        for j in range(n):
            #print(i, j, x)
            temp[j] = x # shape [3]
            #print(temp[j], "\n")
            cur_pred = model(x.double())
            x = cur_pred

        # compute time average at integration time = n
        print("temp: ", temp[:, component])
        temp_sum = torch.sum(temp[:, component])
        print("sum: ", temp_sum)
        time_avg_n = temp_sum / int(n)
        print("time avg: ", time_avg_n)
        #time_avg_n = torch.mean(temp[:, component])
        #print("temp mean: ", time_avg_n)

        # calculate average and difference
        print("diff: ", i, torch.abs(time_avg_n - time_avg_tau_node), "\n")
        diff_time_avg_node[i] = torch.abs(time_avg_n - time_avg_tau_node)
    print("model mean: ", time_avg_tau_node)
    print("sanity check node: ", diff_time_avg_node[-1], "\n")


    fig, ax = subplots()
    n_val_log = np.log(n_val.numpy())
    y1 = diff_time_avg_node.detach().numpy()
    y2 = diff_time_avg_rk4.detach().numpy()
    ax.plot(n_val_log, np.log(y1, out=np.zeros_like(y1), where=(y1!=0)), ".", ms = 10.0)
    ax.plot(n_val_log, np.log(y2, out=np.zeros_like(y2), where=(y2!=0)), ".", ms = 10.0)
    ax.grid(True)
    ax.set_xlabel(r"$n \times \delta t$", fontsize=20)
    ax.set_ylabel(r"log $|\bar z(\tau) - \bar z|$", fontsize=20)
    ax.set_title(r"Log-log Plot for Time Average Convergence")
    ax.legend(['NODE', 'rk4'])
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    tight_layout()
    fig.savefig("../plot/time_avg_convergence.png")

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
    fig.savefig("../plot/"+ str(method) + "_perturbed_multi_step_error.png")

    return



def lyap_exps(dyn_sys, dyn_sys_info, true_traj, iters, time_step, optim_name, method):
    ''' Compute Lyapunov Exponents '''

    # Initialize parameter
    dyn_sys_func = dyn_sys_info[0]
    dim = dyn_sys_info[1]

    # QR Method where U = tangent vector, V = regular system
    U = torch.eye(dim).double()
    lyap_exp = [] #empty list to store the lengths of the orthogonal axes

    real_time = iters * time_step
    t_eval_point = torch.linspace(0, time_step, 2)
    tran = 0

    if method == "NODE":
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # load the saved model
        model = sol.create_NODE(device, dyn_sys= dyn_sys, n_nodes=dim,  n_hidden=64, T=time_step).double()
        path = "../test_result/expt_"+str(dyn_sys)+"/"+optim_name+"/"+str(time_step)+'/'+'model.pt'
        model.load_state_dict(torch.load(path), strict=False)
        model.eval()

        for i in range(0, iters):

            #update x0
            x0 = true_traj[i].to(device).double()

            cur_J = torch.squeeze(F.jacobian(model, x0)).clone().detach()
            J = torch.matmul(cur_J.to("cpu"), U.to("cpu").double())
            if i % 10000 == 0:
                print("jacobian_node", J)

            # QR Decomposition for J
            Q, R = np.linalg.qr(J.clone().detach().numpy())

            lyap_exp.append(np.log(abs(R.diagonal())))
            U = torch.tensor(Q) #new axes after iteration

        LE = [sum([lyap_exp[i][j] for i in range(iters)]) / (real_time) for j in range(dim)]

    else:
        for i in range(0, iters):

            #update x0
            x0 = true_traj[i].double()

            cur_J = F.jacobian(lambda x: torchdiffeq.odeint(dyn_sys_func, x, t_eval_point, method=method), x0)[1]
            J = torch.matmul(cur_J, U)
            if i % 10000 == 0:
                print("jacobian_rk4", J)

            # QR Decomposition for J
            Q, R = np.linalg.qr(J.clone().detach().numpy())

            lyap_exp.append(np.log(abs(R.diagonal())))
            U = torch.tensor(Q).double() #new axes after iteration

        LE = [sum([lyap_exp[i][j] for i in range(iters)]) / (real_time) for j in range(dim)]
    
    return torch.tensor(LE)



def long_time_avg(pred_result, true_result):
    ''' For Calculating Long Time Average of System.
        Example call: long_time_avg('./expt_lorenz/AdamW/0.01/pred_traj.csv', './expt_lorenz/AdamW/0.01/true_traj.csv') '''
    
    pred = np.loadtxt(pred_result, delimiter=",", dtype=float)
    true = np.loadtxt(true_result, delimiter=",", dtype=float)

    print(pred[0:5, 2])
    print(pred[1000:1005, 2])

    print(np.mean(pred[:,2]))
    print(np.mean(true[:,2]))

    return





##### ----- test run ----- #####
if __name__ == '__main__':
    torch.set_printoptions(sci_mode=True, precision=10)
    x = torch.randn(3)
    eps = 1e-6

    # compute lyapunov exponent

    t_eval_point = torch.arange(0,500,1e-2)
    true_traj = torchdiffeq.odeint(lorenz, x, t_eval_point, method='rk4', rtol=1e-8) 

    # dyn_sys, dyn_sys_info, true_traj, iters, x0, time_step, optim_name, method
    # LE_node = lyap_exps("lorenz_periodic", [lorenz_periodic, 3], true_traj = true_traj, iters=10**4, time_step= 1e-2, optim_name="AdamW", x0 = x, method="NODE")
    # print("NODE: ", LE_node)
    
    LE_rk4 = lyap_exps("lorenz", [lorenz, 3], true_traj, iters=5*(10**4), time_step= 1e-2, optim_name="AdamW", method="rk4")
    print("rk4 LE: ", LE_rk4)

    # relative_error(optim_name="AdamW", time_step=0.01)
    # fixed_x = torch.tensor([0.01, 0.01, 0.01], requires_grad=True)
    # plot_time_average(x, dyn_sys="lorenz_periodic", time_step=0.01, optim_name='AdamW', tau=100, component=2)

    #perturbed_multi_step_error("rk4", x, eps, optim_name="AdamW", time_step=0.01, integration_time=1500)
    #perturbed_multi_step_error("NODE", x, eps, optim_name="AdamW", time_step=0.01, integration_time=1500)

    # multi_step_pred_err(fixed_x, optim_name="AdamW", time_step=0.01, integration_time=100, component=0)