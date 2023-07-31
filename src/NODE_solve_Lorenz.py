import torch
import torch.nn as nn
import torchdiffeq
import numpy as np
import matplotlib.pyplot as plt

from examples.Lorenz import lorenz
from src.NODE import ODEBlock, ODEFunc_Lorenz

def simulate(ti, tf, init_state, num_state=100001):
    ''' func: call derivative function
        args: ti, tf = interval of integration
              init_state = initial state, in array format like [1,3]
              num_state = num of state you want to generate '''

    init = torch.Tensor(init_state)
    t_eval_point = torch.linspace(ti, tf, num_state)
    res = torchdiffeq.odeint(lorenz, init, t_eval_point) 
    return res


def create_data(ti, tf, init_state, num_state, n_train=200, n_test=200, n_nodes=2, n_trans=90000):
    ''' func: call simulate to create graph and train, test dataset
        args: ti, tf, init_state = param for simulate()
              n_train = num of training instance
              n_test = num of test instance
              n_nodes = num of nodes in graph
              n_trans = num of transition phase '''
    
    ##### call simulate #####
    res = simulate(ti, tf, init_state, num_state)
    print("Finished Simulating")

    ##### create training dataset #####
    X = np.zeros((n_train, n_nodes))
    Y = np.zeros((n_train, n_nodes))

    for i in range(n_train):
        X[i] = res[n_trans+i]
        Y[i] = res[n_trans+1+i]

    X = torch.tensor(X).reshape(n_train,n_nodes)
    Y = torch.tensor(Y).reshape(n_train,n_nodes)

    ##### create test dataset #####
    X_test = np.zeros((n_test, n_nodes))
    Y_test = np.zeros((n_test, n_nodes))

    for i in range(n_test):
        X_test[i] = res[n_trans+n_train+i]
        Y_test[i] = res[n_trans+1+n_train+i]

    X_test = torch.tensor(X_test).reshape(n_test, n_nodes)
    Y_test = torch.tensor(Y_test).reshape(n_test, n_nodes)

    return X, Y, X_test, Y_test


def create_NODE(device, n_nodes, T):
    # define NODE model
    torch.manual_seed(42)

    neural_func = ODEFunc_Lorenz(y_dim=n_nodes, n_hidden=3).to(device)
    node = ODEBlock(T=T, odefunc=neural_func, method='euler', atol=1e-6, rtol=1e-6, adjoint=True).to(device)

    m = nn.Sequential(
        node).to(device)
    return m


def train(model, device, X, Y, X_test, Y_test, true_t, optimizer, criterion, epochs, lr, time_step):

    # return loss, test_loss, model_final
    num_grad_steps = 0

    pred_train = []
    true_train = []
    loss_hist = []
    train_loss = 0
    optim_name = 'AdamW'

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

        ##### test #####
        pred_test, test_loss_hist = evaluate(model, X_test, Y_test, device, criterion, i, optim_name)

        if (i+1) % 2000 == 0:
            test_multistep(model, epochs, true_t, device, i, optim_name, lr, time_step)
        

    return pred_train, true_train, pred_test, loss_hist, test_loss_hist


def evaluate(model, X_test, Y_test, device, criterion, iter, optimizer_name):

  test_loss_hist = []

  with torch.no_grad():
    model.eval()

    X = X_test.to(device)
    Y = Y_test.to(device)
    test_t = torch.linspace(0, 50, X.shape[0])

    # calculating outputs again with zeroed dropout
    y_pred_test = model(X)

    # save predicted node feature for analysis
    pred_test = y_pred_test.detach().cpu()
    Y = Y.detach().cpu()

    test_loss = criterion(pred_test, Y).item()
    test_loss_hist.append(test_loss)
    # pred_test n_test x 3

    if iter % 2000 == 0:
        plt.figure(figsize=(20, 15))
        ax = plt.axes(projection='3d')
        ax.grid()
        ax.plot3D(Y[:, 0], Y[:, 1], Y[:, 2], 'gray', linewidth=4)
        
        z = pred_test[:, 2]
        ax.scatter3D(pred_test[:, 0], pred_test[:, 1], z, c=z, cmap='hsv', alpha=0.3, linewidth=0)
        ax.set_title(f"Iteration {iter}")
        plt.savefig('expt_lorenz/'+ optimizer_name + '/trajectory/' +str(iter)+'.png', format='png', dpi=400, bbox_inches ='tight', pad_inches = 0.1)
        plt.close("all")
    
  return pred_test, test_loss_hist



def test_multistep(model, epochs, true_traj, device, iter, optimizer_name, lr, time_step):

  test_t = torch.linspace(0, 1, true_traj.shape[0])
  pred_traj = torch.zeros(true_traj.shape[0], 3).to(device)

  with torch.no_grad():
    model.eval()
    model.double()

    # initialize X
    X = true_traj[0].to(device)

    # calculating outputs 
    for i in range(test_t.shape[0]):
        cur_pred = model(X.double())
        pred_traj[i] = cur_pred
        X = pred_traj[i]

    # plot the x, y, z
    # if (iter+1) % 2000 == 0:
    #     plt.figure(figsize=(10, 7.5))
    #     plt.title(f"Iteration {iter+1}")
    #     plt.plot(test_t, pred_traj[:, 0].detach().cpu(), c='C0', ls='--', label='Prediction of x', linewidth=3)
    #     plt.plot(test_t, pred_traj[:, 1].detach().cpu(), c='C1', ls='--', label='Prediction of y', linewidth=3)
    #     plt.plot(test_t, pred_traj[:, 2].detach().cpu(), c='C2', ls='--', label='Prediction of z', linewidth=3)


    #     plt.plot(test_t, true_traj[:, 0].detach().cpu(), c='C3', marker=',', label='Ground Truth of x', alpha=0.6)
    #     plt.plot(test_t, true_traj[:, 1].detach().cpu(), c='C4', marker=',', label='Ground Truth of y', alpha=0.6)
    #     plt.plot(test_t, true_traj[:, 2].detach().cpu(), c='C5', marker=',', label='Ground Truth of z', alpha=0.6)

    #     #plt.axvspan(25, 50, color='gray', alpha=0.2, label='Outside Training')
    #     plt.xlabel('t')
    #     plt.ylabel('y')
    #     plt.legend(loc='best')
    #     plt.savefig('expt_lorenz/'+ optimizer_name + '/multi_step_pred/' +str(iter+1)+'.png', format='png', dpi=400, bbox_inches ='tight', pad_inches = 0.1)
    #     plt.close("all")   

    # Plot Error ||pred - true||
    if iter+1 == epochs:
        error_plot(device, epochs, pred_traj, true_traj, optimizer_name, lr, time_step)

  return




def error_plot(device, num_epoch, pred_train, Y, optimizer_name, lr, time_step):

    plt.figure(figsize=(10, 7.5))
    plt.title(f"|x(t) - x_pred(t)| After {num_epoch} Epochs")
    time_len = 80 * time_step
    print("time_len: ", time_len)

    test_x = torch.linspace(0, time_len, Y.shape[0])
    pred = pred_train.detach().cpu()
    Y = Y.cpu()

    # calculate error
    error_x = np.abs(pred[:, 0] - Y[:, 0])

    # Save error in csv
    err_csv = error_x.numpy()
    np.savetxt('expt_lorenz/'+ optimizer_name + '/' + str(time_step) + '/'+ "error_hist_" + str(time_step) + ".csv", err_csv, delimiter=",")

    # Save error plot in png file
    plt.semilogy(test_x, error_x, linewidth=1)
    plt.xlabel('Time')
    plt.ylabel('Error')
    plt.legend(['element x', 'element y', 'element z'])
    plt.savefig('expt_lorenz/'+ optimizer_name + '/' + str(time_step) + '/'+'training_error_plot_' + str(time_step) +'.png', format='png', dpi=400, bbox_inches ='tight', pad_inches = 0.1)
    plt.close("all")

    return