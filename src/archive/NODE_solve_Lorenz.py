import torch
import torch.nn as nn
import torch.optim as optim
import torchdiffeq
from scipy import stats
import numpy as np
from matplotlib.pyplot import *

from .NODE import ODEBlock, ODEFunc_Lorenz
import sys
sys.path.append('..')
from examples.Lorenz import lorenz

def simulate(ti, tf, init_state, time_step):
    ''' func: call derivative function
        args: ti, tf = interval of integration
              init_state = initial state, in array format like [1,3]
              time_step = time step size used for time integrator '''

    init = torch.Tensor(init_state)
    t_eval_point = torch.arange(ti,tf,time_step)
    traj = torchdiffeq.odeint(lorenz, init, t_eval_point, method='rk4', rtol=1e-8) 
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



def create_NODE(device, n_nodes, n_hidden, T):
    # define NODE model
    torch.manual_seed(42)

    neural_func = ODEFunc_Lorenz(y_dim=n_nodes, n_hidden=64).to(device)
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



def train(model, device, dataset, true_t, optimizer, criterion, epochs, lr, time_step, integration_time):

    # return loss, test_loss, model_final
    num_grad_steps = 0

    pred_train = []
    true_train = []
    loss_hist = []
    test_loss_hist = []
    train_loss = 0
    optim_name = 'AdamW'
    X, Y, X_test, Y_test = dataset

    for i in range(epochs): # looping over epochs
        model.train()
        model.double()

        X = X.to(device)
        Y = Y.to(device)

        y_pred = model(X).to(device)

        optimizer.zero_grad()
        loss = criterion(y_pred, Y)
        train_loss = loss.item()
        loss.backward()
        optimizer.step()

        num_grad_steps += 1

        pred_train.append(y_pred.detach().cpu().numpy())
        true_train.append(Y.detach().cpu().numpy())
        loss_hist.append(train_loss)
        print(num_grad_steps, train_loss)

        ##### test one_step #####
        pred_test, test_loss = evaluate(model, X_test, Y_test, device, criterion, i, optim_name)
        test_loss_hist.append(test_loss)

        ##### test multi_step #####
        #if (i+1) % 2000 == 0:
        if (i+1) == epochs:
            test_multistep(model, epochs, true_t, device, i, optim_name, lr, time_step, integration_time)
        

    return pred_train, true_train, pred_test, loss_hist, test_loss_hist



def evaluate(model, X_test, Y_test, device, criterion, iter, optimizer_name):

  with torch.no_grad():
    model.eval()

    X = X_test.to(device)
    Y = Y_test.to(device)

    # calculating outputs again with zeroed dropout
    y_pred_test = model(X)

    # save predicted node feature for analysis
    pred_test = y_pred_test.detach().cpu()
    Y = Y.detach().cpu()

    test_loss = criterion(pred_test, Y).item()
    # pred_test n_test x 3

    if (iter+1) % 2000 == 0:
        figure(figsize=(20, 15))
        ax = axes(projection='3d')
        ax.grid()
        ax.plot3D(Y[:, 0], Y[:, 1], Y[:, 2], 'gray', linewidth=4)
        
        z = pred_test[:, 2]
        ax.scatter3D(pred_test[:, 0], pred_test[:, 1], z, c=z, cmap='hsv', alpha=0.3, linewidth=0)
        ax.set_title(f"Iteration {iter+1}")
        savefig('expt_lorenz/'+ optimizer_name + '/trajectory/' +str(iter+1)+'.png', format='png', dpi=400, bbox_inches ='tight', pad_inches = 0.1)
        close("all")
    
  return pred_test, test_loss



def test_multistep(model, epochs, true_traj, device, iter, optimizer_name, lr, time_step, integration_time):

  # num_of_extrapolation_dataset
  test_t = torch.linspace(0, 1, true_traj.shape[0])
  pred_traj = torch.zeros(true_traj.shape[0], 3).to(device)

  with torch.no_grad():
    model.eval()
    model.double()

    # initialize X
    X = true_traj[0].to(device)

    # calculating outputs 
    for i in range(true_traj.shape[0]):
        pred_traj[i] = X # shape [3]
        X = torch.reshape(X, (1,3))
        cur_pred = model(X.double())
        X = cur_pred
        if i+1 % 2000 == 0:
            print("Calculating ", (i+1), "th timestep")

    # save predicted trajectory
    pred_traj_csv = np.asarray(pred_traj.detach().cpu())
    np.savetxt('expt_lorenz/'+ optimizer_name + '/' + str(time_step) + '/' +"pred_traj.csv", pred_traj_csv, delimiter=",")

    #plot the x, y, z
    if (iter+1) % 2000 == 0:
    # if (iter+1) == epochs:
        figure(figsize=(40, 15))
        title(f"Iteration {iter+1}")
        plot(test_t, pred_traj[:, 0].detach().cpu(), c='C0', ls='--', label='Prediction of x', linewidth=3)
        plot(test_t, pred_traj[:, 1].detach().cpu(), c='C1', ls='--', label='Prediction of y', linewidth=3)
        plot(test_t, pred_traj[:, 2].detach().cpu(), c='C2', ls='--', label='Prediction of z', linewidth=3)


        plot(test_t, true_traj[:, 0].detach().cpu(), c='C3', marker=',', label='Ground Truth of x', alpha=0.6)
        plot(test_t, true_traj[:, 1].detach().cpu(), c='C4', marker=',', label='Ground Truth of y', alpha=0.6)
        plot(test_t, true_traj[:, 2].detach().cpu(), c='C5', marker=',', label='Ground Truth of z', alpha=0.6)

        #plt.axvspan(25, 50, color='gray', alpha=0.2, label='Outside Training')
        xlabel('t')
        ylabel('y')
        legend(loc='best')
        savefig('expt_lorenz/'+ optimizer_name + '/multi_step_pred/' +str(iter+1)+'.png', format='png', dpi=400, bbox_inches ='tight', pad_inches = 0.1)
        close("all")   

    # Plot Error ||pred - true||
    if (iter+1) == epochs:
        error_plot(device, epochs, pred_traj, true_traj, optimizer_name, lr, time_step, integration_time)

  return




def error_plot(device, num_epoch, pred_traj, Y, optimizer_name, lr, time_step, integration_time):
    '''plot error vs real time'''

    test_x = torch.linspace(0, integration_time, Y.shape[0])
    pred = pred_traj.detach().cpu()
    Y = Y.cpu()

    # calculate error
    error_x = np.abs(pred[:, 0] - Y[:, 0])

    # Save error in csv
    err_csv = error_x.numpy()
    np.savetxt('expt_lorenz/'+ optimizer_name + '/' + str(time_step) + '/'+ "error_hist_" + str(time_step) + ".csv", err_csv, delimiter=",")

    # Plot error
    one_iter = int(1/time_step)
    max_index = torch.argmax(error_x[0:one_iter])
    print("max start index", max_index)
    max_index_end = torch.argmax(error_x[0:one_iter*25])
    print("max end index", max_index_end)

    lin_x = test_x[max_index:max_index_end+1]
    #lin_x = torch.arange(0, (max_index_end+1 - max_index))
    print("new x", lin_x)
    linout = stats.linregress(lin_x, error_x.detach().numpy()[max_index:max_index_end+1])
    y_tangent = lin_x*linout[0] + test_x[max_index]
    print("estimated slope: ", linout[0])
    print("bias: ", linout[1])

    fig, ax = subplots()
    ax.semilogy(test_x, error_x.detach().numpy(), linewidth=2)
    ax.plot(test_x[max_index:max_index_end+1], y_tangent)
    ax.grid(True)
    ax.set_xlabel(r"$n \times \delta t$", fontsize=10)
    ax.set_ylabel(r"$log |x(t) - x\_pred(t)|$", fontsize=10)
    ax.legend(['x component', 'approx slope'])
    ax.xaxis.set_tick_params(labelsize=10)
    ax.yaxis.set_tick_params(labelsize=10)
    tight_layout()
    ax.set_title(f"|x(t) - x_pred(t)| After {num_epoch} Epochs")
    fig.savefig('expt_lorenz/'+ optimizer_name + '/' + str(time_step) + '/'+'error_plot_' + str(time_step) +'.png', format='png', dpi=400, bbox_inches ='tight', pad_inches = 0.1)

    print("multi step pred error: ", error_x[-1])

    return