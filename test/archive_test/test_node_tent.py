import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
sys.path.append('..')

from examples import Tent_map as dyn_sys
sys.path.append('..')
from src.NODE import ODEBlock, ODEFunc_Tent
#from src import NODE_solve_Lorenz_periodic as sol 
#from src import NODE_util as util

def create_data():
    ''' create tent_map data for NODE training sequentially '''

    int_time = torch.arange(0,2,0.001)
    data = torch.zeros(1, len(int_time))

    for i in range(len(int_time)):
        data[0, i] = dyn_sys.tent_map(int_time[i])

    return data


def create_NODE(device, n_nodes, T):
    # define NODE model
    torch.manual_seed(42)

    neural_func = ODEFunc_Tent(y_dim=n_nodes, n_hidden=64).to(device)
    node = ODEBlock(T=T, odefunc=neural_func, method='rk4', atol=1e-9, rtol=1e-9, adjoint=False).to(device)

    m = nn.Sequential(
        node).to(device)
    return m



def train(model, device, X, Y, X_test, Y_test, optimizer, criterion, epochs, lr, time_step):

    # return loss, test_loss, model_final
    num_grad_steps = 0

    pred_train = []
    true_train = []
    loss_hist = []
    test_loss_hist = []
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

        ##### test one_step #####
        pred_test, test_loss = evaluate(model, X_test, Y_test, device, criterion, i, optim_name)
        test_loss_hist.append(test_loss)
        

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
    
  return pred_test, test_loss



def plot_attractor(optim_name, num_epoch, lr, time_step):
    ''' func: plotting the attractor '''

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device: ", device)

    ##### create dataset #####
    data = create_data()
    n_train = int(data.shape[1]*0.8) # cause it's 0-indexed
    n_test = int(data.shape[1]*0.2) - 1
    n_trans = 0
    data = data.T

    ##### training test data split #####
    X = np.zeros((n_train, 1))
    Y = np.zeros((n_train, 1))

    for i in range(n_train):
        X[i] = data[n_trans+i].detach().numpy()
        Y[i] = data[n_trans+1+i].detach().numpy()

    X_test = np.zeros((n_test, 1))
    Y_test = np.zeros((n_test, 1))

    for i in range(n_test):
        X_test[i] = data[n_trans+n_train+i].detach().numpy()
        print(n_trans+1+n_train+i)
        Y_test[i] = data[n_trans+1+n_train+i].detach().numpy()

    X = torch.tensor(X).reshape(n_train, 1)
    Y = torch.tensor(Y).reshape(n_train, 1)
    X_test = torch.tensor(X_test).reshape(n_test, 1)
    Y_test = torch.tensor(Y_test).reshape(n_test, 1)

    print("created data!")


    ##### create model #####
    m = create_NODE(device, n_nodes=1, T=time_step)
    print("created model!")

    ##### train #####
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay =5e-4)

    pred_train, true_train, pred_test, loss_hist, test_loss_hist = train(m,
                                                                             device,
                                                                             X,
                                                                             Y,
                                                                             X_test,
                                                                             Y_test, 
                                                                             optimizer,
                                                                             criterion,
                                                                             epochs=num_epoch,
                                                                             lr=lr,
                                                                             time_step=time_step)
    print("train loss: ", loss_hist[-1])
    print("test loss: ", test_loss_hist[-1])

    return 


##### run experiment #####    

plot_attractor('AdamW', 8000, 5e-4, 1e-2) # optimizer name, epoch, lr, time_step