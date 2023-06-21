import torch
import torch.nn as nn
import torchdiffeq
import numpy as np
import matplotlib.pyplot as plt
import os
#os.makedirs("PNG")

from examples.Sin import sin
from src.neuralODE_Sin import ODEBlock, ODEFunc

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

    ##### create graph #####
    g = 0

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

    return X, Y, X_test, Y_test, g



def modify_graph(g, device):
    '''adapted from ...'''

    # modification on graph #
    #g.add_edges(torch.tensor([1]), torch.tensor([0]))

    # compute diagonal of normalization matrix D according to standard formula
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    # add to dgl.Graph in order for the norm to be accessible at training time
    g.ndata['norm'] = norm.unsqueeze(1).to(device)

    return g


def data_loader(X, Y, X_test, Y_test):
    # Dataloader
    train_data = torch.utils.data.TensorDataset(X, Y)
    test_data = torch.utils.data.TensorDataset(X_test, Y_test)

    # Data iterables
    train_iter = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=False)
    test_iter = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    return train_iter, test_iter



def create_NODE(device, n_nodes):
    # define NODE model
    torch.manual_seed(42)

    neural_func = ODEFunc(y_dim=n_nodes, n_hidden=512).to(device) # change it to one layer
    node = ODEBlock(odefunc=neural_func, method='euler', atol=1e-6, rtol=1e-6, adjoint=True).to(device)

    m = nn.Sequential(node).to(device)
    return m



def train(model, device, X, Y, X_test, Y_test, optimizer, criterion, epochs):

    # return loss, test_loss, model_final
    num_grad_steps = 0

    pred_train = []
    true_train = []
    loss_hist = []

    train_loss = 0

    for i in range(epochs): # looping over epochs
        model.train()
        model.double()

        X = X.to(device)
        Y = Y.to(device)

        output = model(X)

        # save predicted node feature for analysis
        y_pred = output.to(device)
        

        optimizer.zero_grad()
        loss = criterion(y_pred, Y)
        train_loss = loss.item()
        loss.backward()
        optimizer.step()

        num_grad_steps += 1

        pred_train.append(y_pred.detach().numpy())
        true_train.append(Y.detach().numpy())
        loss_hist.append(train_loss)
        print(num_grad_steps, train_loss)

        ##### test #####
        fig, pred_test, test_loss_hist = evaluate(model, X_test, Y_test, device, criterion, i)

    return pred_train, true_train, pred_test, loss_hist, test_loss_hist, fig


def evaluate(model, X_test, Y_test, device, criterion, iter):
  test_loss_hist = []

  with torch.no_grad():
    model.eval()

    X = X_test.to(device)
    Y = Y_test.to(device)

    test_t = torch.linspace(0, 50, X.shape[0])

    # calculating outputs again with zeroed dropout
    y_pred_test = model(X)

    # save predicted node feature for analysis
    pred_test = y_pred_test.detach()

    test_loss = criterion(pred_test, Y).item()
    test_loss_hist.append(test_loss)

    if iter % 200 == 0:
        figure = plt.figure(figsize=(10, 7.5))
        plt.title(f"Iteration {iter}")
        plt.plot(test_t, pred_test[:, 0], c='C0', ls='--', label='Prediction')
        plt.plot(test_t, Y[:, 0], c='C1', ls='--', label='Ground Truth')
        plt.axvspan(25, 50, color='gray', alpha=0.2, label='Outside Training')
        plt.xlabel('t')
        plt.ylabel('y')
        plt.legend(loc='best')
        plt.savefig('PNG/'+str(iter)+'.png', format='png', dpi=400, bbox_inches ='tight', pad_inches = 0.1)
        plt.show()
        plt.close("all")


    
  return figure, pred_test, test_loss_hist







