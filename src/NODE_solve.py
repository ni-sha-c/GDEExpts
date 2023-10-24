import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.autograd.functional as F
import torch.optim as optim
import torchdiffeq
from scipy import stats
import numpy as np
from matplotlib.pyplot import *
import multiprocessing

from .NODE import *
import sys
sys.path.append('..')
from examples.Brusselator import *
from examples.Lorenz_periodic import *
from examples.Lorenz import *
from examples.Sin import *
from examples.Tent_map import *


def simulate(dyn_system, ti, tf, init_state, time_step):
    ''' func: call derivative function
        param:
              dyn_system = dynamical system of our interest 
              ti, tf = interval of integration
              init_state = initial state, in array format like [1,3]
              time_step = time step size used for time integrator '''

    init = torch.Tensor(init_state)
    t_eval_point = torch.arange(ti,tf,time_step)
    traj = torchdiffeq.odeint(dyn_system, init, t_eval_point, method='rk4', rtol=1e-8) 
    print("Finished Simulating")

    return traj



def simulate_NODE(dyn_system, model, ti, tf, init_state, time_step):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    x = torch.Tensor(init_state).double()
    t_eval_point = torch.arange(ti,tf,time_step)
    num_step = tf*int(1/time_step)
    traj = torch.zeros(num_step, 3).to(device)

    for i in range(num_step):
        traj[i] = x # shape [3]
        cur_pred = model(x)
        x = cur_pred
    
    print("Finished Simulating")
    return traj


def create_data(traj, n_train, n_test, n_nodes, n_trans):
    ''' func: call simulate to create graph and train, test dataset
        args: ti, tf, init_state = param for simulate()
              n_train = num of training instance
              n_test = num of test instance
              n_nodes = num of nodes in graph
              n_trans = num of transition phase '''

    ##### create training dataset #####
    X = np.zeros((n_train, n_nodes))
    Y = np.zeros((n_train, n_nodes))

    for i in range(n_train):
        X[i] = traj[n_trans+i]
        Y[i] = traj[n_trans+1+i]

    X = torch.tensor(X).reshape(n_train,n_nodes)
    Y = torch.tensor(Y).reshape(n_train,n_nodes)

    ##### create test dataset #####
    X_test = np.zeros((n_test, n_nodes))
    Y_test = np.zeros((n_test, n_nodes))

    for i in range(n_test):
        X_test[i] = traj[n_trans+n_train+i]
        Y_test[i] = traj[n_trans+1+n_train+i]

    X_test = torch.tensor(X_test).reshape(n_test, n_nodes)
    Y_test = torch.tensor(Y_test).reshape(n_test, n_nodes)

    return [X, Y, X_test, Y_test]



def create_NODE(device, dyn_sys, n_nodes, n_hidden, T):
    ''' Create Neural ODE based on dynamical system of our interest 
        '''

    DYNSYS_MAP = {'sin' : ODEFunc_Sin,
                  'tent_map' : ODEFunc_Tent,
                  'brusselator' : ODEFunc_Brusselator,
                  'lorenz_periodic' : ODEFunc_Lorenz_periodic,
                  'lorenz' : ODEFunc_Lorenz}
    ODEFunc = DYNSYS_MAP[dyn_sys]
    neural_func = ODEFunc(y_dim=n_nodes, n_hidden=n_hidden).to(device)
    node = ODEBlock(T=T, odefunc=neural_func, method='rk4', atol=1e-9, rtol=1e-9, adjoint=False).to(device)

    m = nn.Sequential(
        node).to(device)
    return m



def define_optimizer(optim_name, model, lr, weight_decay):

    optim_mapping = {
    "Adam": optim.Adam,
    "AdamW": optim.AdamW,
    "SGD": optim.SGD,
    "RMSprop": optim.RMSprop}

    if optim_name in optim_mapping:
        optim_class = optim_mapping[optim_name]
        optimizer = optim_class(model.parameters(), lr=lr, weight_decay =weight_decay)
    else:
        print(optim_name, " is not in the optim_mapping!")
    
    return optimizer



def define_dyn_sys(dyn_sys):
    DYNSYS_MAP = {'sin' : [sin, 1],
                  'tent_map' : [tent_map, 1],
                  'brusselator' : [brusselator, 2],
                  'lorenz_periodic' : [lorenz_periodic, 3],
                  'lorenz' : [lorenz, 3]}
    dyn_sys_info = DYNSYS_MAP[dyn_sys]
    dyn_sys_func, dim = dyn_sys_info

    return dyn_sys_func, dim



def create_iterables(dataset, batch_size):
    X, Y, X_test, Y_test = dataset

    # Dataloader
    train_data = torch.utils.data.TensorDataset(X, Y)
    test_data = torch.utils.data.TensorDataset(X_test, Y_test)

    # Data iterables
    train_iter = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    return train_iter, test_iter


def jacobian_loss(True_J, cur_model_J, output_loss):
    reg_param = 1.0

    print("True_J", True_J)
    print("cur_model_J", cur_model_J)
    diff_jac = True_J - cur_model_J
    norm_diff_jac = torch.norm(diff_jac)
    jac_loss = torch.sqrt(norm_diff_jac)
    total_loss = reg_param * jac_loss + output_loss
    # try torch.mean

    return total_loss



def jacobian_parallel(dyn_sys, model, X, t_eval_point, device, node):

    dyn_sys_f, dim = define_dyn_sys(dyn_sys)

    with multiprocessing.Pool(processes=20) as pool:
        results = pool.map(single_jacobian, [(dyn_sys_f, model, x, t_eval_point, device, node) for x in X])

    return results

def single_jacobian(args):
    '''Compute Jacobian of dyn_sys
    Param:  '''
    dyn_sys_f, model, x, t_eval_point, device, node = args

    if node == True:
        jac = torch.squeeze(F.jacobian(model, x))
    else:
        jac = F.jacobian(lambda x: torchdiffeq.odeint(dyn_sys_f, x, t_eval_point, method="rk4"), x, vectorize=True)[1]
    
    return jac



def train(dyn_sys, model, device, dataset, true_t, optim_name, criterion, epochs, lr, weight_decay, time_step, real_time, tran_state, minibatch=False, batch_size=0):

    # Initialize
    pred_train, true_train, loss_hist, test_loss_hist = ([] for i in range(4))
    optimizer = define_optimizer(optim_name, model, lr, weight_decay)
    X, Y, X_test, Y_test = dataset
    X, Y, X_test, Y_test = X.to(device), Y.to(device), X_test.to(device), Y_test.to(device)
    num_train = X.shape[0]
    multiprocessing.set_start_method('spawn')
    

    # Compute True Jacobian
    t_eval_point = torch.linspace(0, time_step, 2).to(device)
    #True_J = torch.zeros(num_train, 3, 3).to(device)
    # for i in range(num_train):
    #     x0 = X[i].double()

    # func = lambda x: torchdiffeq.odeint(lorenz, x, t_eval_point, method="rk4")
    # result = func(X.view(3, 10000))
    # print("func", result)
    # print(result[0])
    # print(result.shape)
    # True_J = F.jacobian(lambda x: torchdiffeq.odeint(lorenz, x, t_eval_point, method="rk4"), X, vectorize=True)
    # print("tj:", True_J)
    True_J = jacobian_parallel(dyn_sys, model, X, t_eval_point, device, node=False) # 10000 x 3 x 3
    True_J = torch.stack(True_J)
    print(True_J.shape)
    print("Finished Computing True Jacobian")

    for i in range(epochs): # looping over epochs
        model.train()
        model.double()

        if minibatch == True:
            train_iter, test_iter = create_iterables(dataset, batch_size=batch_size)
            y_pred = torch.zeros(len(train_iter), batch_size, 3)
            y_true = torch.zeros(len(train_iter), batch_size, 3)
            k = 0

            for xk, yk in train_iter:
                xk = xk.to(device) # [batch size,3]
                yk = yk.to(device)
                output = model(xk)

                # save predicted node feature for analysis
                y_pred[k] = output
                y_true[k] = yk
                k += 1

            optimizer.zero_grad()
            loss = criterion(y_pred, y_true)
            train_loss = loss.item()
            loss.backward()
            optimizer.step()

            # leave it for debug purpose for now, and remove it
            pred_train.append(y_pred.detach().cpu().numpy())
            true_train.append(y_true.detach().cpu().numpy())

        elif minibatch == False:

            y_pred = model(X).to(device)
            cur_model_J = F.jacobian(model, X[:100], vectorize=True)
            cur_model_J = cur_model_J.view(100, 3, 3)
            print("vec", cur_model_J)
            print("vec", cur_model_J.shape)
            #cur_model_J = jacobian_parallel(dyn_sys, model, X, t_eval_point, device, node=True)

            optimizer.zero_grad()
            MSE_loss = criterion(y_pred, Y)
            cur_model_J = torch.stack(cur_model_J)
            train_loss = jacobian_loss(True_J, cur_model_J, MSE_loss)
            train_loss.backward()
            optimizer.step()

            # leave it for debug purpose for now, and remove it
            pred_train.append(y_pred.detach().cpu().numpy())
            true_train.append(Y.detach().cpu().numpy())
        
        loss_hist.append(train_loss)
        print(i, MSE_loss.item(), train_loss.item())

        ##### test one_step #####
        pred_test, test_loss = evaluate(dyn_sys, model, X_test, Y_test, device, criterion, i, optim_name)
        test_loss_hist.append(test_loss)

        ##### test multi_step #####
        if (i+1) == epochs:
           test_multistep(dyn_sys, model, epochs, true_t, device, i, optim_name, lr, time_step, real_time, tran_state)

    return pred_train, true_train, pred_test, loss_hist, test_loss_hist



def evaluate(dyn_sys, model, X_test, Y_test, device, criterion, iter, optimizer_name):

  with torch.no_grad():
    model.eval()

    # calculating outputs again with zeroed dropout
    y_pred_test = model(X_test)

    # save predicted node feature for analysis
    pred_test = y_pred_test.detach().cpu()
    Y_test = Y_test.detach().cpu()

    test_loss = criterion(pred_test, Y_test).item()

  return pred_test, test_loss



def test_multistep(dyn_sys, model, epochs, true_traj, device, iter, optimizer_name, lr, time_step, integration_time, tran_state):

  # num_of_extrapolation_dataset
  num_data, dim = true_traj.shape
  test_t = torch.linspace(0, integration_time, num_data)
  pred_traj = torch.zeros(num_data, dim).to(device)

  with torch.no_grad():
    model.eval()
    model.double()

    # initialize X
    print(true_traj[0])
    X = true_traj[0].to(device)

    # calculating outputs 
    for i in range(num_data):
        pred_traj[i] = X # shape [3]
        cur_pred = model(X.double())
        X = cur_pred
        if i+1 % 2000 == 0:
            print("Calculating ", (i+1), "th timestep")

    # save predicted trajectory
    pred_traj_csv = np.asarray(pred_traj.detach().cpu())
    true_traj_csv = np.asarray(true_traj.detach().cpu())
    np.savetxt('../test_result/expt_'+str(dyn_sys)+'/'+ optimizer_name + '/' + str(time_step) + '/' +"pred_traj.csv", pred_traj_csv, delimiter=",")
    np.savetxt('../test_result/expt_'+str(dyn_sys)+'/'+ optimizer_name + '/' + str(time_step) + '/' +"true_traj.csv", true_traj_csv, delimiter=",")

    # plot traj
    plot_multi_step_traj_3D(dyn_sys, optimizer_name, test_t, pred_traj, true_traj)

    # Plot Error ||pred - true||
    multi_step_pred_error_plot(dyn_sys, device, epochs, pred_traj, true_traj, optimizer_name, lr, time_step, integration_time, tran_state)

  return



def plot_multi_step_traj_3D(dyn_sys, optim_n, test_t, pred_traj, true_traj):
    #plot the x, y, z

    figure(figsize=(18, 6))
    title(f"Multi-Step Predicted Trajectory of Lorenz")
    plot(test_t, pred_traj[:, 0].detach().cpu(), c='C0', ls='--', label='Prediction of x', linewidth=3)
    plot(test_t, pred_traj[:, 1].detach().cpu(), c='C1', ls='--', label='Prediction of y', linewidth=3)
    plot(test_t, pred_traj[:, 2].detach().cpu(), c='C2', ls='--', label='Prediction of z', linewidth=3)

    plot(test_t, true_traj[:, 0].detach().cpu(), c='C3', marker=',', label='Ground Truth of x', alpha=0.6)
    plot(test_t, true_traj[:, 1].detach().cpu(), c='C4', marker=',', label='Ground Truth of y', alpha=0.6)
    plot(test_t, true_traj[:, 2].detach().cpu(), c='C5', marker=',', label='Ground Truth of z', alpha=0.6)

    xlabel('t')
    ylabel('y')
    legend(loc='best')
    savefig('../plot/expt_'+str(dyn_sys)+'_'+ optim_n + '_multi_step_pred.svg', format='svg', dpi=600, bbox_inches ='tight', pad_inches = 0.1)
   
    return



def multi_step_pred_error_plot(dyn_sys, device, num_epoch, pred_traj, Y, optimizer_name, lr, time_step, integration_time, tran_state):
    ''' func: plot error vs real time 
        args:   pred_traj = pred_traj by Neural ODE (csv file)
                Y = true_traj (csv file) '''

    one_iter = int(1/time_step)
    #test_x = torch.arange(0, integration_time, time_step)[tran_state:]
    test_x = torch.arange(0, integration_time, time_step)
    pred = pred_traj.detach().cpu()
    Y = Y.cpu()

    # calculate error
    error_x = np.abs(pred[:, 0] - Y[:, 0]) # np.linalg.norm

    # Save error in csv
    np.savetxt('../test_result/expt_'+str(dyn_sys)+'/'+ optimizer_name + '/' + str(time_step) + '/'+ "error_hist_" + str(time_step) + ".csv", error_x, delimiter=",")

    # Convert y to ln(y)
    error_x = np.clip(error_x, 1e-12, None)
    log_e_error = np.log(error_x)

    if dyn_sys == "lorenz":
        # Find the index for tangent slope
        max_index = torch.argmax(torch.tensor(log_e_error[0:one_iter]))
        MIE_start = one_iter*20
        max_index_end = torch.argmax(torch.tensor(log_e_error[MIE_start:one_iter*30]))

        lin_x = test_x[max_index:max_index_end+MIE_start+1]
        print("lin x:", lin_x)

        # Find tangent line
        linout = stats.linregress(lin_x, log_e_error[max_index:max_index_end+MIE_start+1])
        y_tangent = lin_x*linout[0] + linout[1] #log_e_error[max_index]
        print("estimated slope: ", linout[0])
        print("bias: ", linout[1], np.log(linout[1]))
        print("slope: ", np.abs(y_tangent[0] - y_tangent[-1])/(lin_x[-1] - lin_x[0]))

    # Plot semilogy error
    fig, ax = subplots(figsize=(24, 12))
    ax.plot(test_x, log_e_error, linewidth=1, alpha=0.9)
    if dyn_sys == "lorenz":
        ax.plot(lin_x, y_tangent, linewidth=2, alpha=0.9)
    ax.grid(True)
    ax.set_xlabel(r"$n \times \delta t$", fontsize=24)
    ax.set_ylabel(r"$log |x(t) - x\_pred(t)|$", fontsize=24)
    ax.legend(['x component', 'approx slope'])
    ax.xaxis.set_tick_params(labelsize=24)
    ax.yaxis.set_tick_params(labelsize=24)
    tight_layout()
    #ax.set_title(r"log |x(t) - x_pred(t)|"+" After {num_epoch} Epochs")
    fig.savefig('../test_result/expt_'+str(dyn_sys)+'/'+ optimizer_name + '/' + str(time_step) + '/'+'error_plot_' + str(time_step) +'.svg', format='svg', dpi=800, bbox_inches ='tight', pad_inches = 0.1)

    print("multi step pred error: ", error_x[-1])

    return