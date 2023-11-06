
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.autograd.functional as F
import torch.optim as optim
import torchdiffeq
import numpy as np
from matplotlib.pyplot import *
import multiprocessing
import torch
import torch.nn as nn


def lorenz(t, u, rho=28.0):
    """ Lorenz chaotic differential equation: du/dt = f(t, u)
    t: time T to evaluate system
    u: state vector [x, y, z]
    return: new state vector in shape of [3]"""

    sigma = 10.0
    #rho = 28.0
    beta = 8/3

    res = torch.stack([
        sigma * (u[1] - u[0]),
        u[0] * (rho - u[2]) - u[1],
        (u[0] * u[1]) - (beta * u[2])
    ])

    return res

"""### Create Dataset"""

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

"""### Create Model"""

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
    DYNSYS_MAP = {'lorenz' : [lorenz, 3]} 
    dyn_sys_info = DYNSYS_MAP[dyn_sys]
    dyn_sys_func, dim = dyn_sys_info

    return dyn_sys_func, dim


"""### Define Jacobian Loss"""

def jacobian_loss(True_J, cur_model_J, output_loss):
    reg_param = 0.11

    diff_jac = True_J - cur_model_J
    norm_diff_jac = torch.norm(diff_jac)

    total_loss = reg_param * norm_diff_jac + output_loss

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

def Jacobian_Matrix(input, sigma, r, b):
    '''Compute Jacobian Matrix'''

    x, y, z = input
    return torch.stack([torch.tensor([-sigma, sigma, 0]), torch.tensor([r - z, -1, -x]), torch.tensor([y, x, -b])])



"""### Create Training Loop"""

class ODE_Lorenz(nn.Module):
    '''Define Neural Network that approximates differential equation system of Chaotic Lorenz'''

    def __init__(self):
        super(ODE_Lorenz, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32 * 9),
            nn.GELU(),
            nn.Linear(32 * 9, 64 * 9),
            nn.GELU(),
            nn.Linear(64 * 9, 3)
        )
        # self.t = torch.linspace(0, 0.01, 2)

    def forward(self, t, y):
        res = self.net(y)
        return res

# Time Integrator
def solve_odefunc(odefunc, t, y0):
    ''' Solve odefunction using torchdiffeq.odeint() '''

    solution = torchdiffeq.odeint(odefunc, y0, t, rtol=1e-9, atol=1e-9, method="rk4")
    final_state = solution[-1]
    return final_state

# y0 = torch.randn(3)
# t = torch.linspace(0, 0.01, 2)

# odefunc = ODE_Lorenz()

# ##### Sanity Check #####
# final_state = solve_odefunc(odefunc, t, y0)
# print("ODE state:", final_state)
# final_state = solve_odefunc(lorenz, t, y0)
# print("True state:", final_state)



def train(dyn_sys, model, device, dataset, true_t, optim_name, criterion, epochs, lr, weight_decay, time_step, real_time, tran_state, minibatch=False, batch_size=0):

    # Initialize
    pred_train, true_train, loss_hist, test_loss_hist = ([] for i in range(4))
    optimizer = define_optimizer(optim_name, model, lr, weight_decay)
    X, Y, X_test, Y_test = dataset
    X, Y, X_test, Y_test = X.to(device), Y.to(device), X_test.to(device), Y_test.to(device)
    num_train = X.shape[0]

    # Compute True Jacobian
    t_eval_point = torch.linspace(0, time_step, 2).to(device).double()

    True_J = torch.ones(num_train, 3, 3).to(device)
    for i in range(num_train):
        True_J[i] = Jacobian_Matrix(X[i, :], sigma=10.0, r=28.0, b=8/3)
    print(True_J.shape)
    print("Finished Computing True Jacobian")

    for i in range(epochs): # looping over epochs
        m.train()
        m.double()

        y_pred = solve_odefunc(m, t_eval_point, X).to(device)

        optimizer.zero_grad()
        MSE_loss = criterion(y_pred, Y)
        jacrev = torch.func.jacrev(m, argnums=1)
        # print("jacrev", jacrev)
        compute_batch_jac = torch.vmap(jacrev, in_dims=(None, 0))
        cur_model_J = compute_batch_jac(t_eval_point, X).to(device)
        # print(cur_model_J.grad_fn)
        train_loss = jacobian_loss(True_J, cur_model_J, MSE_loss)
        train_loss.backward()
        optimizer.step()

        # leave it for debug purpose for now, and remove it
        #pred_train.append(y_pred.detach().cpu().numpy())
        #true_train.append(Y.detach().cpu().numpy())

        loss_hist.append(train_loss)
        print(i, MSE_loss.item(), train_loss.item())

        ##### test one_step #####
        if i % 1000 == 0:
            pred_test, test_loss = evaluate(dyn_sys, m, X_test, Y_test, device, criterion, i, optim_name)
            test_loss_hist.append(test_loss)

        ##### test multi_step #####


    return pred_train, true_train, pred_test, loss_hist, test_loss_hist


def evaluate(dyn_sys, model, X_test, Y_test, device, criterion, iter, optimizer_name):

  with torch.no_grad():
    model.eval()

    t_eval_point = torch.linspace(0, time_step, 2).to(device).double()
    y_pred_test = solve_odefunc(model, t_eval_point, X_test).to(device)

    # save predicted node feature for analysis
    pred_test = y_pred_test.detach().cpu()
    Y_test = Y_test.detach().cpu()

    test_loss = criterion(pred_test, Y_test).item()
    print(test_loss)

  return pred_test, test_loss

def test_multistep(dyn_sys, model, epochs, true_traj, device, iter, optimizer_name, lr, time_step, integration_time, tran_state):

  # num_of_extrapolation_dataset
  t_eval_point = torch.linspace(0, time_step, 2).to(device).double()
  num_data, dim = true_traj.shape
  test_t = torch.linspace(0, integration_time, num_data)
  pred_traj = torch.zeros(num_data, dim).to(device)

  with torch.no_grad():
    model.eval()
    model.double()
    model.to(device)

    # initialize X
    print(true_traj[0])
    X = true_traj[0].to(device)

    # calculating outputs
    for i in range(num_data):
        pred_traj[i] = X # shape [3]
        # cur_pred = model(t.to(device), X.double())
        cur_pred = solve_odefunc(model, t_eval_point, X.double()).to(device)
        X = cur_pred

    # save predicted trajectory
    pred_traj_csv = np.asarray(pred_traj.detach().cpu())
    true_traj_csv = np.asarray(true_traj.detach().cpu())

    # plot traj
    plot_multi_step_traj_3D(dyn_sys, optimizer_name, test_t, pred_traj, true_traj)

    # Plot Error ||pred - true||
    multi_step_pred_error_plot(dyn_sys, device, epochs, pred_traj, true_traj, optimizer_name, lr, time_step, integration_time, tran_state)

  return



def plot_multi_step_traj_3D(dyn_sys, optim_n, test_t, pred_traj, true_traj):
    #plot the x, y, z

    fig, axs = subplots(2, figsize=(18, 9), sharex=True)
    fig.suptitle("Multi-Step Predicted Trajectory of Lorenz", fontsize=24)
    axs[0].plot(test_t, pred_traj[:, 0].detach().cpu(), c='C0', ls='--', label='Prediction of x', alpha=0.7)
    axs[0].plot(test_t, pred_traj[:, 1].detach().cpu(), c='C1', ls='--', label='Prediction of y', alpha=0.7)
    axs[0].plot(test_t, pred_traj[:, 2].detach().cpu(), c='C2', ls='--', label='Prediction of z', alpha=0.7)
    axs[0].grid(True)
    axs[0].legend(loc='best', fontsize=20)
    axs[0].set_ylabel(r'$\Phi_{NODE}(t)$', fontsize=24)
    axs[0].tick_params(labelsize=24)

    axs[1].plot(test_t, true_traj[:, 0].detach().cpu(), c='C3', marker=',', label='Ground Truth of x', alpha=0.7)
    axs[1].plot(test_t, true_traj[:, 1].detach().cpu(), c='C4', marker=',', label='Ground Truth of y', alpha=0.7)
    axs[1].plot(test_t, true_traj[:, 2].detach().cpu(), c='C5', marker=',', label='Ground Truth of z', alpha=0.7)
    axs[1].grid(True)
    axs[1].legend(loc='best', fontsize=20)
    axs[1].tick_params(labelsize=24)
    axs[1].set_ylabel(r'$\Phi_{rk4}(t)$', fontsize=24)

    xlabel('t', fontsize=24)
    tight_layout()
    savefig('_multi_step_pred.svg', format='svg', dpi=600, bbox_inches ='tight', pad_inches = 0.1)

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
    slope = [np.exp(0.9*x)+error_x[15] for x in test_x[:500]]
    slope = np.array(slope)
    
    fig, ax = subplots(figsize=(24, 12))
    ax.semilogy(test_x[15:], error_x[15:], linewidth=2, alpha=0.9, color="b")
    ax.semilogy(test_x[15:500], slope[15:], linewidth=2, ls="--", color="gray", alpha=0.9)
    ax.grid(True)
    ax.set_xlabel(r"$n \times \delta t$", fontsize=24)
    ax.set_ylabel(r"$log |\Phi_{rk4}(t) - \Phi_{NODE}(t)|$", fontsize=24)
    ax.legend(['x component', 'approx slope'], fontsize=20)
    ax.tick_params(labelsize=24)
    tight_layout()
    fig.savefig('error_plot_' + str(time_step) +'.svg', format='svg', dpi=800, bbox_inches ='tight', pad_inches = 0.1)

    print("multi step pred error: ", error_x[-1])

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
        model = ODE_Lorenz().double()
        path = 'model.pt'
        model.to(device)
        model.load_state_dict(torch.load(path), strict=False)
        model.eval()

        for i in range(0, iters):
            if i % 1000 == 0:
                print(i)

            #update x0
            x0 = true_traj[i].to(device).double()
            t_eval_point = t_eval_point.to(device)
            # cur_J = model(x0).clone().detach()
            cur_J = F.jacobian(lambda x: torchdiffeq.odeint(model, x, t_eval_point, method="rk4"), x0)[1]
            #print(cur_J)
            J = torch.matmul(cur_J.to("cpu"), U.to("cpu").double())

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

            # QR Decomposition for J
            Q, R = np.linalg.qr(J.clone().detach().numpy())

            lyap_exp.append(np.log(abs(R.diagonal())))
            U = torch.tensor(Q).double() #new axes after iteration

        LE = [sum([lyap_exp[i][j] for i in range(iters)]) / (real_time) for j in range(dim)]

    return torch.tensor(LE)

"""### Run the Code"""

# Set device
torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device: ", device)

# Hyperparameters
dyn_sys_func, dim = lorenz, 3
time_step = 1e-2
lr=5e-4
weight_decay = 5e-4
num_epoch = 10000
integration_time = 100
num_train = 3000
num_test= 3000
num_trans= 0 #1000
iters= 5*(10**4)
minibatch=False
batch_size=500
optim_name="AdamW"
dyn_sys="lorenz"

# Assign Initial Point of Orbit
x0 = torch.randn(dim)
x_multi_0 = torch.randn(dim)
print("initial point:", x_multi_0)

# Initialize Model and Dataset Parameters
criterion = torch.nn.MSELoss()
real_time = iters * time_step
print("real time: ", real_time)

# Generate Training/Test/Multi-Step Prediction Data
traj = simulate(dyn_sys_func, 0, integration_time, x0, time_step)
multi_step_traj = simulate(dyn_sys_func, 0, real_time, x0, time_step)
dataset = create_data(traj, n_train=num_train, n_test=num_test, n_nodes=dim, n_trans=num_trans)

# Create model
m = ODE_Lorenz().to(device)

# Train the model, return node
pred_train, true_train, pred_test, loss_hist, test_loss_hist = train(dyn_sys, m, device, dataset, multi_step_traj, optim_name, criterion, num_epoch, lr, weight_decay, time_step, real_time, num_trans, minibatch=minibatch, batch_size=batch_size)
print("train loss: ", loss_hist[-1])

# save the model
torch.save(m.state_dict(), "model.pt")
print("Saved new model!")

# dyn_sys, model, epochs, true_traj, device, iter, optimizer_name, lr, time_step, integration_time, tran_state
test_multistep(dyn_sys, m, num_epoch, multi_step_traj, device, 20000, optim_name, lr, time_step, real_time, 0)

# dyn_sys, dyn_sys_info, true_traj, iters, x0, time_step, optim_name, method
LE_node = lyap_exps("lorenz", [lorenz, 3], true_traj = multi_step_traj, iters=5*(10**4), time_step= time_step, optim_name="AdamW", method="NODE")
print(LE_node)

LErk4_node = lyap_exps("lorenz", [lorenz, 3], true_traj = multi_step_traj, iters=5*(10**4), time_step= time_step, optim_name="AdamW", method="rk4")
print(LErk4_node)
