import torch
import torch.nn as nn
import torchdiffeq
import numpy as np
import dgl

from examples.Brusselator import brusselator
from src.GDE import GCNLayer, GDEFunc, ControlledGDEFunc
from src.neuralODE import ODEBlock, ODEFunc

def simulate(ti, tf, init_state, num_state=100001):
    ''' func: call derivative function
        args: ti, tf = interval of integration
              init_state = initial state, in array format like [1,3]
              num_state = num of state you want to generate '''

    init = torch.Tensor(init_state)
    t_eval_point = torch.linspace(ti, tf, num_state)
    res = torchdiffeq.odeint(brusselator, init, t_eval_point) 
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
    u = torch.tensor([0])
    v = torch.tensor([1])
    g = dgl.graph((u, v), num_nodes=2)

    g.ndata["feat"] = res[n_trans-1]
    g.ndata["label"] = res[n_trans-1]

    ##### create training dataset #####
    X = np.zeros((n_train, n_nodes))
    Y = np.zeros((n_train, n_nodes))

    for i in range(n_train):
        X[i] = res[n_trans+i]
        Y[i] = res[n_trans+1+i]

    X = torch.tensor(X).reshape(n_train,2)
    Y = torch.tensor(Y).reshape(n_train,2)

    ##### create test dataset #####
    X_test = np.zeros((n_test, n_nodes))
    Y_test = np.zeros((n_test, n_nodes))

    for i in range(n_test):
        X_test[i] = res[n_trans+n_train+i]
        Y_test[i] = res[n_trans+1+n_train+i]

    X_test = torch.tensor(X_test).reshape(n_test, 2)
    Y_test = torch.tensor(Y_test).reshape(n_test, 2)

    return X, Y, X_test, Y_test, g



def modify_graph(g, device):
    '''adapted from ...'''

    # modification on graph #
    g.add_edges(torch.tensor([1]), torch.tensor([0]))

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



def create_NODE(device):
    # define NODE model
    torch.manual_seed(42)

    neural_func = ODEFunc().to(device)
    node = ODEBlock(odefunc=neural_func, method='euler', atol=1e-6, rtol=1e-6, adjoint=True).to(device)

    m = nn.Sequential(
        nn.Linear(2,2),
        node,
        nn.Linear(2,2)).to(device)
    return m



def train(model, device, train_iter, test_iter, optimizer, criterion, epochs):

    # return loss, test_loss, model_final
    num_grad_steps = 0

    pred_train = []
    true_train = []
    x_train = []
    loss_hist = []

    train_loss = 0

    for i in range(epochs): # looping over epochs
        model.train()
        model.double()

        y_pred = torch.zeros(len(train_iter), 1, 2)
        y_true = torch.zeros(len(train_iter), 1, 2)
        k = 0
        x_train = torch.zeros(len(train_iter), 1, 2)
        
        for xk,yk in train_iter:
            xk = xk.to(device)
            yk = yk.to(device)

            output = model(xk)

            # save predicted node feature for analysis
            y_pred[k] = output[-1].to(device)
            y_true[k] = yk[-1]
            x_train[k] = xk[-1]
            k += 1

        loss = criterion(y_pred, y_true)
        train_loss = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        num_grad_steps += 1

        pred_train.append(y_pred.detach().numpy())
        true_train.append(y_true.detach().numpy())
        loss_hist.append(train_loss)
        print(num_grad_steps, train_loss)

        ##### test #####
        pred_test, true_test, test_loss_hist = evaluate(model, test_iter, device, criterion)

    return pred_train, true_train, x_train, pred_test, true_test, loss_hist, test_loss_hist


def evaluate(model, test_iter, device, criterion):
  pred_test = torch.zeros(len(test_iter), 1, 2)
  true_test = torch.zeros(len(test_iter), 1, 2)
  test_loss_hist = []

  with torch.no_grad():
    model.eval()
    k = 0

    for x,y in test_iter:
      xk = x.to(device)
      yk = y.to(device)

      # calculating outputs again with zeroed dropout
      y_pred_test = model(xk)

      # save predicted node feature for analysis
      pred_test[k] = y_pred_test[-1].detach()
      true_test[k] = yk[-1].detach()
      k += 1

    test_loss = criterion(pred_test, true_test).item()
    test_loss_hist.append(test_loss)
    #print("test loss: ", test_loss)
    
  return pred_test, true_test, test_loss_hist







