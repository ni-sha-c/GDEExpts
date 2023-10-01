import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torchdiffeq
from scipy import stats
import numpy as np
from matplotlib.pyplot import *

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



def create_iterables(dataset, batch_size):
    X, Y, X_test, Y_test = dataset

    # Dataloader
    train_data = torch.utils.data.TensorDataset(X, Y)
    test_data = torch.utils.data.TensorDataset(X_test, Y_test)

    # Data iterables
    train_iter = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    return train_iter, test_iter


def get_batch(device, batch_time, data_size, batch_size, true_y, t):
    ''' func: transform trajectory into mini-batch of size T x batch_size x d
        param: batch_time = T
               batch_size = M

    Adapted from https://github.com/rtqichen/torchdiffeq/blob/master/examples/ode_demo.py '''

    s = torch.from_numpy(np.arange(data_size - batch_time, dtype=np.int64)[:batch_size])

    # Define a set of M starting points
    batch_y0 = true_y[s]  # (M, D)
    # Define batch time
    batch_t = t[:batch_time]  # (T)
    # Generate batch from M different stating points
    batch_y = torch.stack([true_y[s + i] for i in range(batch_time)], dim=0)  # (T, M, D)
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)



# def train(dyn_sys, model, device, dataset, true_t, optimizer, criterion, epochs, lr, time_step, real_time, tran_state):

#     # return loss, test_loss, model_final
#     num_grad_steps = 0

#     pred_train = []
#     true_train = []
#     loss_hist = []
#     test_loss_hist = []
#     train_loss = 0
#     optim_name = 'AdamW'
#     X, Y, X_test, Y_test = dataset
#     batch_time = 2
#     data_size = int(100/0.01)
#     batch_size = 9500
#     t = torch.arange(0.0, 100.0, 0.01).to(device)
#     y0 = torch.randn(3).to(device)
#     true_y = torchdiffeq.odeint(lorenz, y0, t, method='rk4', rtol=1e-8)

#     for i in range(epochs): # looping over epochs
#         model.train()
#         model.double()

#         optimizer.zero_grad()
#         batch_y0, batch_t, batch_y = get_batch(device, batch_time, data_size, batch_size, true_y, t)

#         #print(batch_y0[:5]) #[batch_size, dim]
#         #print(batch_t[:5]) #[batch_t]
#         #print(batch_y[:5]) #[batch_t, batch_size, dim]
#         func = ODEFunc_Lorenz().to(device)
#         y_pred = torchdiffeq.odeint(func, batch_y0, batch_t, method='rk4').to(device)

#         loss = criterion(y_pred, batch_y) #(y_true - y_pred) % 2 * pi
#         train_loss = loss.item()
#         loss.backward()
#         optimizer.step()

#         pred_train.append(y_pred.detach().cpu().numpy())
#         true_train.append(batch_y.detach().cpu().numpy())
#         loss_hist.append(train_loss)
#         print(i, train_loss)

#         ##### test one_step #####
#         #pred_test, test_loss = evaluate(dyn_sys, model, X_test, Y_test, device, criterion, i, optim_name)
#         #test_loss_hist.append(test_loss)

#         ##### test multi_step #####
#         #if (i+1) % 2000 == 0:
#         #if (i+1) == epochs:
#         #    test_multistep(dyn_sys, model, epochs, true_t, device, i, optim_name, lr, time_step, real_time, tran_state)
        

#     return pred_train, true_train, pred_test, loss_hist, test_loss_hist

def train(dyn_sys, model, device, dataset, true_t, optim_name, criterion, epochs, lr, weight_decay, time_step, real_time, tran_state, minibatch=False, batch_size=0):

    # Initialize
    pred_train, true_train, loss_hist, test_loss_hist = ([] for i in range(4))
    optimizer = define_optimizer(optim_name, model, lr, weight_decay)
    X, Y, X_test, Y_test = dataset

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
            X = X.to(device)
            Y = Y.to(device)

            y_pred = model(X).to(device)

            optimizer.zero_grad()
            loss = criterion(y_pred, Y)
            train_loss = loss.item()
            loss.backward()
            optimizer.step()

            # leave it for debug purpose for now, and remove it
            pred_train.append(y_pred.detach().cpu().numpy())
            true_train.append(Y.detach().cpu().numpy())
        
        loss_hist.append(train_loss)
        print(i, train_loss)

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

    X = X_test.to(device)
    Y = Y_test.to(device)

    # calculating outputs again with zeroed dropout
    y_pred_test = model(X)

    # save predicted node feature for analysis
    pred_test = y_pred_test.detach().cpu()
    Y = Y.detach().cpu()

    test_loss = criterion(pred_test, Y).item()

    # TODO: Update trajectory plot

    # if (iter+1) % 2000 == 0:
    #     figure(figsize=(20, 15))
    #     ax = axes(projection='3d')
    #     ax.grid()
    #     ax.plot3D(Y[:, 0], Y[:, 1], Y[:, 2], 'gray', linewidth=4)
        
    #     z = pred_test[:, 2]
    #     ax.scatter3D(pred_test[:, 0], pred_test[:, 1], z, c=z, cmap='hsv', alpha=0.3, linewidth=0)
    #     ax.set_title(f"Iteration {iter+1}")
    #     savefig('expt_'+str(dyn_sys)+'/'+ optimizer_name + '/trajectory/' +str(iter+1)+'.png', format='png', dpi=400, bbox_inches ='tight', pad_inches = 0.1)
    #     close("all")
    
  return pred_test, test_loss



def test_multistep(dyn_sys, model, epochs, true_traj, device, iter, optimizer_name, lr, time_step, integration_time, tran_state):

  # num_of_extrapolation_dataset
  num_data, dim = true_traj.shape
  test_t = torch.linspace(0, 1, num_data)
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
        #X = torch.reshape(X, (1,3))
        cur_pred = model(X.double())
        X = cur_pred
        if i+1 % 2000 == 0:
            print("Calculating ", (i+1), "th timestep")

    # save predicted trajectory
    pred_traj_csv = np.asarray(pred_traj.detach().cpu())
    np.savetxt('../test_result/expt_'+str(dyn_sys)+'/'+ optimizer_name + '/' + str(time_step) + '/' +"pred_traj.csv", pred_traj_csv, delimiter=",")

    #plot the x, y, z
    # TODO: update this
    # if (iter+1) % 2000 == 0:
    # # if (iter+1) == epochs:
    #     figure(figsize=(40, 15))
    #     title(f"Iteration {iter+1}")
    #     plot(test_t, pred_traj[:, 0].detach().cpu(), c='C0', ls='--', label='Prediction of x', linewidth=3)
    #     plot(test_t, pred_traj[:, 1].detach().cpu(), c='C1', ls='--', label='Prediction of y', linewidth=3)
    #     plot(test_t, pred_traj[:, 2].detach().cpu(), c='C2', ls='--', label='Prediction of z', linewidth=3)


    #     plot(test_t, true_traj[:, 0].detach().cpu(), c='C3', marker=',', label='Ground Truth of x', alpha=0.6)
    #     plot(test_t, true_traj[:, 1].detach().cpu(), c='C4', marker=',', label='Ground Truth of y', alpha=0.6)
    #     plot(test_t, true_traj[:, 2].detach().cpu(), c='C5', marker=',', label='Ground Truth of z', alpha=0.6)

    #     #plt.axvspan(25, 50, color='gray', alpha=0.2, label='Outside Training')
    #     xlabel('t')
    #     ylabel('y')
    #     legend(loc='best')
    #     savefig('expt_'+str(dyn_sys)+'/'+ optimizer_name + '/multi_step_pred/' +str(iter+1)+'.png', format='png', dpi=400, bbox_inches ='tight', pad_inches = 0.1)
    #     close("all")   

    # Plot Error ||pred - true||
    if (iter+1) == epochs:
        multi_step_pred_error_plot(dyn_sys, device, epochs, pred_traj, true_traj, optimizer_name, lr, time_step, integration_time, tran_state)

  return




def multi_step_pred_error_plot(dyn_sys, device, num_epoch, pred_traj, Y, optimizer_name, lr, time_step, integration_time, tran_state):
    '''plot error vs real time'''

    one_iter = int(1/time_step)
    test_x = torch.arange(0, integration_time, time_step)[tran_state:]
    error_x = np.zeros(test_x.shape)
    pred = pred_traj.detach().cpu()
    Y = Y.cpu()

    # calculate error
    # error_x = np.abs(pred[:, 0] - Y[:, 0])
    #for i in range(error_x.shape[0]):
    error_x = np.abs(pred[:, 0] - Y[:, 0]) # np.linalg.norm

    # Save error in csv
    np.savetxt('../test_result/expt_'+str(dyn_sys)+'/'+ optimizer_name + '/' + str(time_step) + '/'+ "error_hist_" + str(time_step) + ".csv", error_x, delimiter=",")

    # Convert y to ln(y)
    log_e_error = np.log(error_x)

    if dyn_sys == "lorenz":
        # Find the index for tangent slope
        max_index = torch.argmax(torch.tensor(log_e_error[0:one_iter]))
        MIE_start = one_iter*15
        max_index_end = torch.argmax(torch.tensor(log_e_error[MIE_start:one_iter*20]))# +MIE_start
        print("max start index", max_index)
        print("max end index", max_index_end)

        lin_x = test_x[max_index:max_index_end+MIE_start+1]
        print("lin x:", lin_x)

        # Find tangent line
        linout = stats.linregress(lin_x, log_e_error[max_index:max_index_end+MIE_start+1])
        y_tangent = lin_x*linout[0] + linout[1] #log_e_error[max_index]
        print("estimated slope: ", linout[0])
        print("bias: ", linout[1], np.log(linout[1]))
        print("slope: ", np.abs(y_tangent[0] - y_tangent[-1])/(lin_x[-1] - lin_x[0]))

    # Plot semilogy error
    fig, ax = subplots()
    ax.plot(test_x, log_e_error, linewidth=2)
    if dyn_sys == "lorenz":
        ax.plot(lin_x, y_tangent)
    ax.grid(True)
    ax.set_xlabel(r"$n \times \delta t$", fontsize=10)
    ax.set_ylabel(r"$\log |x(t) - x\_pred(t)|$", fontsize=10)
    ax.legend(['x component', 'approx slope'])
    ax.xaxis.set_tick_params(labelsize=10)
    ax.yaxis.set_tick_params(labelsize=10)
    tight_layout()
    ax.set_title(r"\log |x(t) - x_pred(t)| After {num_epoch} Epochs")
    fig.savefig('../test_result/expt_'+str(dyn_sys)+'/'+ optimizer_name + '/' + str(time_step) + '/'+'error_plot_' + str(time_step) +'.png', format='png', dpi=400, bbox_inches ='tight', pad_inches = 0.1)

    print("multi step pred error: ", error_x[-1])

    return