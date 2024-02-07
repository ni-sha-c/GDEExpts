import torch
import numpy as np
import torchdiffeq
from matplotlib.pyplot import * 
import scipy
from scipy import stats
from scipy.stats import ortho_group
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
from examples.Coupled_Brusselator import *
from src.NODE_solve import *
from src.NODE import *

# TODO: Make sure that all of function starts from calling saved model. So that # it can also be used outside of training loop

''' List of functions included in test_metrics.py:

    1. relative_error()
    2. plot_time_average()
    3. perturbed_multi_step_error()
    4. lyap_exps()
    5. test_jacobian()
    6. test_stability()
    '''


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



def plot_time_average(init, dyn_sys, time_step, optim_name, tau, component, model_name):
    ''' func: plot |avg(z_tau) - avg(z_t)| '''

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transition_phase = 0

    # Initialize integration times
    taus = torch.arange(1, tau+1) # 1 ~ 100
    len_taus = taus.shape[0]
    diff_time_avg_rk4 = torch.zeros(len_taus*int(1/time_step))
    diff_time_avg_node = torch.zeros(len_taus*int(1/time_step))

    # Load the saved model
    model = sol.create_NODE(device, dyn_sys=dyn_sys, n_nodes=3, n_hidden=64, T=time_step)
    path = "../test_result/expt_"+str(dyn_sys)+"/"+optim_name+"/"+str(time_step)+'/'+str(model_name)+'/model.pt'
    model.load_state_dict(torch.load(path))
    model.eval()

    # ----- compute node_tau, rk4_tau ----- #
    t_eval_point = torch.arange(0, tau, time_step)
    iters = t_eval_point.shape[0]
    x = init.to(device)
    init_2 = torch.tensor([1., 0., 0.])

    phi_tau = torch.zeros(tau, 3).to(device)
    
    # rk4_tau
    sol_tau = torchdiffeq.odeint(lorenz, init, t_eval_point, method='rk4', rtol=1e-8) 
    time_avg_tau_rk4 = torch.mean(sol_tau[:, component])

    sol_tau_2 = torchdiffeq.odeint(lorenz, init_2, t_eval_point, method='rk4', rtol=1e-8) 
    time_avg_tau_rk4 = torch.mean(sol_tau[:, component])

    # node_tau
    phi_tau = torchdiffeq.odeint(model, init, t_eval_point, method='rk4', rtol=1e-8) 
    time_avg_tau_node = torch.mean(phi_tau[:, component])

    #----- Create phi_rk4[:, component] -----#
    for i in range(len_taus*int(1/time_step)):

        # compute time average at integration time = n
        time_avg_rk4 = torch.mean(sol_tau[:i, component])
        # calculate average and difference
        diff_time_avg_rk4[i] = time_avg_rk4 - time_avg_tau_rk4

    print("mean rk4: ", time_avg_tau_rk4)
    print("sanity check rk4: ", diff_time_avg_rk4[-1])


    #----- Create phi_NODE[:, component] -----#
    x = init.to(device)
    for i in range(len_taus*int(1/time_step)):

        # compute time average at integration time = n
        time_avg_n = torch.mean(phi_tau[:i, component])
        # calculate average and difference
        diff_time_avg_node[i] = time_avg_n - time_avg_tau_rk4 #time_avg_tau_node
         #

        print("diff: ", i, diff_time_avg_node[i], "\n")

    print("model mean: ", time_avg_tau_node)
    print("sanity check node: ", diff_time_avg_node[-1], "\n")


    fig, ax = subplots(figsize=(20,10))
    # taus_log = np.log(taus.numpy())
    taus_x = np.arange(0, tau, time_step)
    rk4 = np.abs(diff_time_avg_rk4.detach().numpy())
    node = np.abs(diff_time_avg_node.detach().numpy())
    rk4[-1], node[-1] = 1e-3, 1e-3


    ax.semilogy(np.log10(taus_x), rk4, color=(0.25, 0.25, 0.25), marker='o', linewidth=4, alpha=1)
    ax.semilogy(np.log10(taus_x), node, color="slateblue", marker='o', linewidth=4, alpha=1)

    ax.grid(True)
    ax.set_xlabel(r"$log(n \times \delta t)$", fontsize=24)
    ax.set_ylabel(r"log $|\bar z(\tau) - \bar z|$", fontsize=24)
    ax.set_title(r"Log-log Plot for Time Average Convergence", fontsize=24)
    ax.legend(['rk4', 'NODE'], fontsize=24)
    ax.xaxis.set_tick_params(labelsize=24)
    ax.yaxis.set_tick_params(labelsize=24)
    tight_layout()

    pdf_path = '../plot/timeavg_'+str(model_name)+'_'+str(torch.round(init, decimals=4).tolist())+'.pdf'
    fig.savefig(pdf_path, format='pdf', dpi=400)

    return


def plot_time_average_multi(init, dyn_sys, time_step, optim_name, tau, component, model_name):
    ''' func: plot |avg(z_tau) - avg(z_t)| '''

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transition_phase = 0

    # Initialize integration times
    taus = torch.arange(1, tau+1) # 1 ~ 100
    len_taus = taus.shape[0]
    diff_time_avg_rk4 = torch.zeros(len_taus*int(1/time_step))
    diff_time_avg_node = torch.zeros(len_taus*int(1/time_step))

    fig, ax = subplots(figsize=(20,10))
    # taus_log = np.log(taus.numpy())
    taus_x = np.arange(0, tau, time_step)

    # Load the saved model
    model = sol.create_NODE(device, dyn_sys=dyn_sys, n_nodes=3, n_hidden=64, T=time_step)
    path = "../test_result/expt_"+str(dyn_sys)+"/"+optim_name+"/"+str(time_step)+'/'+str(model_name)+'/model.pt'
    model.load_state_dict(torch.load(path))
    model.eval()

    # ----- compute node_tau, rk4_tau ----- #
    t_eval_point = torch.arange(0, tau, time_step)
    iters = t_eval_point.shape[0]
    x = init.to(device)
    init_2 = torch.tensor([1., 0., 0.])

    phi_tau = torch.zeros(tau, 3).to(device)
    
    # rk4_tau
    sol_tau = torchdiffeq.odeint(lorenz, init, t_eval_point, method='rk4', rtol=1e-8) 
    time_avg_tau_rk4 = torch.mean(sol_tau[:, component])


    # node_tau
    phi_tau = torchdiffeq.odeint(model, init, t_eval_point, method='rk4', rtol=1e-8) 
    time_avg_tau_node = torch.mean(phi_tau[:, component])

    #----- Create phi_rk4[:, component] -----#
    for i in range(len_taus*int(1/time_step)):

        # compute time average at integration time = n
        time_avg_rk4 = torch.mean(sol_tau[:i, component])
        # calculate average and difference
        diff_time_avg_rk4[i] = time_avg_rk4 - time_avg_tau_rk4

    print("mean rk4: ", time_avg_tau_rk4)
    print("sanity check rk4: ", diff_time_avg_rk4[-1])


    #----- Create phi_NODE[:, component] -----#
    x = init.to(device)
    for i in range(len_taus*int(1/time_step)):

        # compute time average at integration time = n
        time_avg_n = torch.mean(phi_tau[:i, component])
        # calculate average and difference
        diff_time_avg_node[i] = time_avg_n - time_avg_tau_rk4 #time_avg_tau_node
         #

        print("diff: ", i, diff_time_avg_node[i], "\n")

    print("model mean: ", time_avg_tau_node)
    print("sanity check node: ", diff_time_avg_node[-1], "\n")



    rk4 = np.abs(diff_time_avg_rk4.detach().numpy())
    node = np.abs(diff_time_avg_node.detach().numpy())
    rk4[-1], node[-1] = 1e-3, 1e-3


    ax.semilogy(np.log10(taus_x), rk4, color=(0.25, 0.25, 0.25), marker='o', linewidth=4, alpha=1)
    ax.semilogy(np.log10(taus_x), node, color="slateblue", marker='o', linewidth=4, alpha=1)

    ax.grid(True)
    ax.set_xlabel(r"$log(n \times \delta t)$", fontsize=24)
    ax.set_ylabel(r"log $|\bar z(\tau) - \bar z|$", fontsize=24)
    ax.set_title(r"Log-log Plot for Time Average Convergence", fontsize=24)
    ax.legend(['rk4', 'NODE'], fontsize=24)
    ax.xaxis.set_tick_params(labelsize=24)
    ax.yaxis.set_tick_params(labelsize=24)
    tight_layout()

    pdf_path = '../plot/timeavg_'+str(model_name)+'_'+str(torch.round(init, decimals=4).tolist())+'.pdf'
    fig.savefig(pdf_path, format='pdf', dpi=400)

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


def lyap_exps(dyn_sys, dyn_sys_info, true_traj, iters, time_step, optim_name, method, path):
    ''' Compute Lyapunov Exponents 
        args: path = path to model '''

    # Initialize parameter
    dyn_sys_func, dyn_sys_name, dim = dyn_sys_info

    # QR Method where U = tangent vector, V = regular system
    U = torch.eye(dim).double()
    lyap_exp = [] #empty list to store the lengths of the orthogonal axes

    real_time = iters * time_step
    if (dyn_sys_name == "henon") or (dyn_sys_name == "baker"):
        assert time_step == 1
    t_eval_point = torch.linspace(0, time_step, 2)
    tran = 0

    if method == "NODE":
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        t_eval_point = t_eval_point.to(device)

        # load the saved model
        model = create_NODE(device, dyn_sys= dyn_sys, n_nodes=dim,  n_hidden=64, T=time_step).double()
        model.load_state_dict(torch.load(path), strict=False)
        model.eval()

        for i in range(iters):
            if i % 1000 == 0:
                print(i)

            #update x0
            x0 = true_traj[i].to(device).double()
            # cur_J = model(x0).clone().detach()
            if (dyn_sys_name =="henon") or (dyn_sys_name == "baker"):
                cur_J = F.jacobian(model, x0)
            else:
                cur_J = F.jacobian(lambda x: torchdiffeq.odeint(model, x, t_eval_point, method="rk4"), x0)[1]
            #print(cur_J)
            J = torch.matmul(cur_J.to("cpu"), U.to("cpu").double())

            # QR Decomposition for J
            Q, R = torch.linalg.qr(J)

            lyap_exp.append(torch.log(abs(R.diagonal())))
            U = Q #new axes after iteration

        LE = [sum([lyap_exp[i][j] for i in range(iters)]) / (real_time) for j in range(dim)]

    else:
        for i in range(iters):

            #update x0
            x0 = true_traj[i].double()
            if (dyn_sys_name =="henon") or (dyn_sys_name =="baker"):
                cur_J = F.jacobian(dyn_sys_func, x0)
            else:
                cur_J = F.jacobian(lambda x: torchdiffeq.odeint(dyn_sys_func, x, t_eval_point, method=method), x0)[1]
            #print(cur_J)
         
            J = torch.matmul(cur_J, U)

            # QR Decomposition for J
            Q, R = torch.linalg.qr(J)

            lyap_exp.append(torch.log(abs(R.diagonal())))
            U = Q.double() #new axes after iteration

        LE = [sum([lyap_exp[i][j] for i in range(iters)]) / (real_time) for j in range(dim)]

    return torch.tensor(LE)



def lyap_exps_ks(dyn_sys, dyn_sys_info, true_traj, iters, u_list, dx, L, c, T, dt, time_step, optim_name, method, path):
    ''' Compute Lyapunov Exponents 
        args: path = path to model '''

    # Initialize parameter
    dyn_sys_func, dyn_sys_name, org_dim = dyn_sys_info

    # reorthonormalization
    epsilon = 1e-6
    dim = 15
    N = 100
    print("d", dim)

    # QR Method where U = tangent vector, V = regular system
    # CHANGE IT TO DIM X M -> THEN IT WILL COMPUTE M LYAPUNOV EXPONENT.!
    U = torch.eye(*(org_dim, dim)).double()
    print("U", U)
    lyap_exp = [] #empty list to store the lengths of the orthogonal axes
    

    real_time = iters * time_step
    t_eval_point = torch.linspace(0, time_step, 2)
    tran = 0
    print("true traj ks", true_traj.shape)

    if method == "NODE":
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        t_eval_point = t_eval_point.to(device)

        # load the saved model
        model = create_NODE(device, dyn_sys= dyn_sys, n_nodes=org_dim,  n_hidden=64, T=time_step).double()
        model.load_state_dict(torch.load(path), strict=False)
        model.eval()

        for i in range(iters):
            if i % 1000 == 0:
                print(i)

            #update x0
            x0 = true_traj[i].to(device).double()
            # cur_J = model(x0).clone().detach()
            if (dyn_sys_name =="henon") or (dyn_sys_name == "baker"):
                cur_J = F.jacobian(model, x0)
            else:
                cur_J = F.jacobian(lambda x: torchdiffeq.odeint(model, x, t_eval_point, method="rk4"), x0)[1]
            #print(cur_J)
            J = torch.matmul(cur_J.to("cpu"), U.to("cpu").double())

            # QR Decomposition for J
            Q, R = torch.linalg.qr(J)

            lyap_exp.append(torch.log(abs(R.diagonal())))
            U = Q #new axes after iteration

        LE = [sum([lyap_exp[i][j] for i in range(iters)]) / (real_time) for j in range(dim)]

        # for i in range(iters):
        #     if i % 1000 == 0:
        #         print(i)

        #     #update x0
        #     x0 = true_traj[i].to(device).double()
        #     cur_J = F.jacobian(lambda x: torchdiffeq.odeint(model, x, t_eval_point, method="rk4"), x0)[-1]
        #     perturbed_J = torch.zeros(cur_J.shape)

        #     # Reorthogonalization steps
        #     for n in range(dim):
        #         # Perturb the system
        #         perturbation = epsilon * U[:, n]
        #         new_input = x0 + perturbation.to(device)
        #         # perturbed_traj = torchdiffeq.odeint(model, new_input.to(device), t_eval_point, method="rk4")[-1]
        #         perturbed_J = F.jacobian(lambda x: torchdiffeq.odeint(model, x, t_eval_point, method="rk4"), new_input)[-1]


        #     J = (cur_J - perturbed_J) / epsilon
        #     print("J shape", J.shape)
        #     # QR Decomposition for J
        #     Q, R = torch.linalg.qr(J)

        #     lyap_exp.append(torch.log(abs(R.diagonal())))
        #     U = Q  # New axes after iteration


        LE = [sum([lyap_exp[i][j] for i in range(iters)]) / (real_time) for j in range(dim)]

    else:
        
        for i in range(iters):
            if i % 1000 == 0:
                print("rk4", i) 
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            #update x0
            x0 = true_traj[i].requires_grad_(True)
            
            dx = 1 # 0.25
            dt = 0.25
            c = 0.4

            cur_J = F.jacobian(lambda x: run_KS(x, c, dx, dt, dt*2, False, device), x0, vectorize=True)[-1]

            J = torch.matmul(cur_J.to(device).double(), U.to(device).double())

            # QR Decomposition for J
            Q, R = torch.linalg.qr(J)

            lyap_exp.append(torch.log(abs(R.diagonal())))
            U = Q.double() #new axes after iteration

        lyap_exp = torch.stack(lyap_exp).detach().cpu().numpy()

        LE = [sum([lyap_exp[i][j] for i in range(iters)]) / (real_time) for j in range(dim)]

    return torch.tensor(LE)



def test_jacobian(device, x0, method, time_step, optim_name, dyn_sys_info, dyn_sys):
    ''' Compute Jacobian Matrix of rk4 or Neural ODE 
            args:   x0 = 3D tensor of initial point 
            method: "NODE" or any other time integrator '''

    x0 = x0.to(device).double()
    print("initial point: ", x0)
    t_eval_point = torch.linspace(0, time_step, 2).to(device)
    dyn_sys_func, dim = dyn_sys_info


    # jacobian_node
    if method == "NODE":
        # Load the model
        model = create_NODE(device, dyn_sys=dyn_sys, n_nodes=dim,  n_hidden=64, T=time_step).double()
        path = "../test_result/expt_"+dyn_sys+"/"+optim_name+"/"+str(time_step)+'/'+'model.pt'
        model.load_state_dict(torch.load(path))
        model.eval()
        #node_fixed_point = model(x0)
        #print("fixed point for node?: ", node_fixed_point)
        cur_J = torch.squeeze(F.jacobian(model, x0)).clone().detach()

    # jacobian_rk4
    else:
        rk4 = lambda x: torchdiffeq.odeint(dyn_sys_func, x, t_eval_point, method=method)
        #print("fixed point?: ", rk4(x0)[1])
        cur_J = F.jacobian(rk4, x0)[1]
    
    jac_2 = torch.matmul(cur_J, cur_J.T)
    #print(method, cur_J)
    eig = torch.linalg.eigvals(cur_J)
    print("eigenvalue: ", eig)
    print("eigenvalue of J^2: ", torch.linalg.eigvals(jac_2), "\n")

    return



    

def compute_wasserstein(device, int_time, init_state, time_step, model_name):
    ti, tf = int_time

    # Load the saved model
    model = ODE_Lorenz().to(device)
    model_path = "../test_result/expt_lorenz/AdamW/"+str(time_step)+'/'+str(model_name)+'/model.pt'
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("Finished Loading model")

    node_data = simulate(model, ti, tf, init_state, time_step)
    node_data = node_data.detach().cpu().numpy()
    true_data = simulate(lorenz, ti, tf, init_state, time_step).detach().cpu().numpy()

    # Compute Wasserstein Distance
    dist_x = scipy.stats.wasserstein_distance(node_data[:, 0], true_data[:, 0])
    dist_y = scipy.stats.wasserstein_distance(node_data[:, 1], true_data[:, 1])
    dist_z = scipy.stats.wasserstein_distance(node_data[:, 2], true_data[:, 2])
    print(dist_x, dist_y, dist_z)
    print(torch.norm(torch.tensor([dist_x, dist_y, dist_z])))

    return


def compute_timeavg(device, int_time, init_state, time_step, model_name):
    ti, tf = int_time

    # Load the saved model
    model = ODE_Lorenz().to(device)
    model_path = "../test_result/expt_lorenz/AdamW/"+str(time_step)+'/'+str(model_name)+'/model.pt'
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("Finished Loading model")

    node_data = simulate(model, ti, tf, init_state, time_step)
    node_data = node_data.detach().cpu().numpy()
    true_data = simulate(lorenz, ti, tf, init_state, time_step).detach().cpu().numpy()

    # Compute Wasserstein Distance
    dist_x = np.mean(node_data[:, 0]) - np.mean(true_data[:, 0])
    dist_y = np.mean(node_data[:, 1]) - np.mean(true_data[:, 1])
    dist_z = np.mean(node_data[:, 2]) - np.mean(true_data[:, 2])
    print(dist_x, dist_y, dist_z)
    print(torch.norm(torch.tensor([dist_x, dist_y, dist_z])))

    return


def test_stability(device, x0, method, time_step, optim_name, dyn_sys_info, dyn_sys):
    ''' 1. Generate 100 pair of (S, S') 
        2. Train with S and S' and obtain optimal w_S, w_S'
        3. On the test dataset, find Loss difference 
        4. Repeat it for 100 pairs'''


    x0 = x0.to(device).double()
    print("initial point: ", x0)
    t_eval_point = torch.linspace(0, time_step, 2).to(device)
    dyn_sys_func, dim = dyn_sys_info

    # jacobian_node
    if method == "NODE":
        # Load the model
        model = sol.create_NODE(device, dyn_sys=dyn_sys, n_nodes=dim,  n_hidden=64, T=time_step).double()
        path = "../test_result/expt_"+dyn_sys+"/"+optim_name+"/"+str(time_step)+'/'+'model.pt'
        model.load_state_dict(torch.load(path))
        model.eval()
        #node_fixed_point = model(x0)
        #print("fixed point for node?: ", node_fixed_point)
        cur_J = torch.squeeze(F.jacobian(model, x0)).clone().detach()

    # jacobian_rk4
    else:
        rk4 = lambda x: torchdiffeq.odeint(dyn_sys_func, x, t_eval_point, method=method)
        #print("fixed point?: ", rk4(x0)[1])
        cur_J = F.jacobian(rk4, x0)[1]
    

    return



##### ----- test run ----- #####
if __name__ == '__main__':
    torch.set_printoptions(sci_mode=False, precision=5)

    # ---- Test Wasserstein Distance ----- #
    # init_state_out = torch.tensor([1.,1.,-1.])
    # init_state_inner = torch.tensor([14.9440, 13.9801, 36.6756])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    '''compute_timeavg(device, [0, 500], init_state.to(device), 0.01, "MSE_0")
    compute_timeavg(device, [0, 500], init_state.to(device), 0.01, "MSE_5")
    compute_timeavg(device, [0, 500], init_state.to(device), 0.01, "JAC_0")
    compute_timeavg(device, [0, 500], init_state.to(device), 0.01, "JAC_5")'''

    # ----- Test Jacobian ----- #
    '''device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.randn(3)
    #x = torch.tensor([0,0,0]).double()
    #test_jacobian(device, x, "NODE", 0.01, "AdamW", [lorenz, 3], "lorenz")
    test_jacobian(device, x, "rk4", 0.01, "AdamW", [lorenz, 3], "lorenz")'''

    # ----- compute lyapunov exponent ----- #
    init_state_out = torch.tensor([1.,1.,-1.])
    init_state_inner = torch.tensor([14.9440, 13.9801, 36.6756])

    longer_traj = simulate(lorenz, 0, 500, init_state_out, 0.01)

    t_eval_point = torch.arange(0,500,1e-2)
    # true_traj = torchdiffeq.odeint(lorenz, x, t_eval_point, method='rk4', rtol=1e-8) #transition phase
    MSE0_path = "../test_result/expt_lorenz/AdamW/0.01/MSE_0/model.pt"
    MSE5_path = "../test_result/expt_lorenz/AdamW/0.01/MSE_5/model.pt"
    JAC0_path = "../test_result/expt_lorenz/AdamW/0.01/JAC_0/model.pt"
    JAC5_path = "../test_result/expt_lorenz/AdamW/0.01/JAC_5/model.pt"

    MSE0_LE_NODE = lyap_exps("lorenz", [lorenz, "lorenz", 3], longer_traj, iters=5*10**4, time_step= 0.01, optim_name="AdamW", method="NODE", path=MSE0_path)
    
    MSE5_LE_NODE = lyap_exps("lorenz", [lorenz, "lorenz", 3], longer_traj, iters=5*10**4, time_step= 0.01, optim_name="AdamW", method="NODE", path=MSE5_path)
    
    JAC0_LE_NODE = lyap_exps("lorenz", [lorenz, "lorenz", 3], longer_traj, iters=5*10**4, time_step= 0.01, optim_name="AdamW", method="NODE", path=JAC0_path)
    
    JAC5_LE_NODE = lyap_exps("lorenz", [lorenz, "lorenz", 3], longer_traj, iters=5*10**4, time_step= 0.01, optim_name="AdamW", method="NODE", path=JAC5_path)

    print(MSE0_LE_NODE, MSE5_LE_NODE, JAC0_LE_NODE, JAC5_LE_NODE)
    
    LE_rk4 = lyap_exps("lorenz", [lorenz, "lorenz", 3], longer_traj, iters=5*10**4, time_step= 0.01, optim_name="AdamW", method="rk4", path=JAC5_path)
    print("rk4 LE: ", LE_rk4)

    # relative_error(optim_name="AdamW", time_step=0.01)
    # fixed_x = torch.tensor([0.01, 0.01, 0.01], requires_grad=True)
    '''device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #init = torch.tensor([-8.6445e-01,-1.19299e+00,1.4918e+01])
    init = torch.tensor([1., 0., 0.])
    plot_time_average(init.to(device), dyn_sys="lorenz", time_step=0.01, optim_name='AdamW', tau=500, component=2, model_name="JAC_0")'''

    #perturbed_multi_step_error("rk4", x, eps, optim_name="AdamW", time_step=0.01, integration_time=1500)
    #perturbed_multi_step_error("NODE", x, eps, optim_name="AdamW", time_step=0.01, integration_time=1500)