import torch
import torch.nn as nn
import torchdiffeq
import numpy as np
import matplotlib.pyplot as plt
import os
#os.makedirs("trajectory_sin_png")

from examples.Sin import sin
from src.NODE import ODEBlock, ODEFunc_Sin


def simulate(ti, tf, init_state, num_state=10001):
    ''' func: call derivative function
        args: ti, tf = interval of integration
              init_state = initial state, in array format like [1,3]
              num_state = num of state you want to generate '''

    init = torch.Tensor(init_state)
    t_eval_point = torch.linspace(ti, tf, num_state)
    res = torchdiffeq.odeint(sin, init, t_eval_point) 
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

    ##### create test dataset #####
    X_test = np.zeros((n_test, n_nodes))
    Y_test = np.zeros((n_test, n_nodes))

    for i in range(n_test):
        X_test[i] = res[n_trans+n_train+i]
        Y_test[i] = res[n_trans+1+n_train+i]

    X_test = torch.tensor(X_test).reshape(n_test, n_nodes)
    Y_test = torch.tensor(Y_test).reshape(n_test, n_nodes)

    return X, Y, X_test, Y_test



def create_NODE(device, n_nodes):
    # define NODE model
    torch.manual_seed(42)

    neural_func = ODEFunc_Sin(y_dim=n_nodes, n_hidden=512).to(device)
    node = ODEBlock(T=0.025, odefunc=neural_func, method='euler', atol=1e-6, rtol=1e-6, adjoint=True).to(device)

    m = nn.Sequential(node).to(device)
    return m



def train(model, device, X, Y, X_test, Y_test, optimizer, criterion, epochs):
    # return loss, test_loss, model_final

    num_grad_steps = 0
    pred_train = []
    true_train = []
    loss_hist = []
    train_loss = 0
    optim_name = 'Gradient Descent'

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

    if iter % 500 == 0:
        plt.figure(figsize=(10, 7.5))
        plt.title(f"Iteration {iter}")
        plt.plot(test_t, pred_test[:, 0], c='C0', ls='--', label='Prediction', linewidth=2)
        plt.plot(test_t, Y[:, 0], c='gray', label='Ground Truth', alpha=0.5, linewidth=2)
        #plt.axvspan(25, 50, color='gray', alpha=0.2, label='Outside Training')
        plt.xlabel('t')
        plt.ylabel('y')
        plt.legend(loc='best')
        plt.savefig('expt_sin_second_order/'+ optimizer_name + '/trajectory/' +str(iter)+'.png', format='png', dpi=400, bbox_inches ='tight', pad_inches = 0.1)
        plt.close("all")
    
  return pred_test, test_loss_hist







