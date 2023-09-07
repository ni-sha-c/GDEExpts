import torch
import numpy as np
import torch.autograd.functional as F
import sys
import torchdiffeq
from matplotlib.pyplot import *
sys.path.append('..')

from scipy.integrate import odeint
from src import NODE_solve_Lorenz as sol 
from src import NODE_util as util
from examples import Lorenz as func
import test_node_lorenz as test_node

def test_jac_node(x, optim_name, time_step):
    #jac_node_x
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = x.to(device)

    # load the saved model
    model = sol.create_NODE(device, n_nodes=3, T=time_step).double()
    path = "expt_lorenz/"+optim_name+"/"+str(time_step)+'/'+'model.pt'

    model.load_state_dict(torch.load(path))
    model.eval()

    # compute the jacobian of neural ode
    jacobian_node = F.jacobian(model, x)

    print("----- JAC_NODE -----")
    squeeze_jac_node = torch.squeeze(jacobian_node)
    print(squeeze_jac_node)
    return



def test_autodiff(x, eps, time_step, method):

    deltat = time_step
    # ----- jac_numerical_sol ----- #
    if method == "Euler":

        x_reshaped = x.reshape(1,3)
        jacobian_true_ode = deltat*F.jacobian(func.lorenz_jac, x_reshaped)
        jacobian_true_ode[0][0][0] += 1
        jacobian_true_ode[1][0][1] += 1
        jacobian_true_ode[2][0][2] += 1

        print("----- JAC_TRUE -----")
        squeeze_jac = torch.squeeze(jacobian_true_ode)
        print(squeeze_jac)


    elif method == "RK4":
        # initialize
        t_eval_point = torch.linspace(0, time_step, 2)
        jac_rk4_fd = torch.zeros(3,3).double()
        
        # Automatic Differentiation
        jac_rk4_ad = F.jacobian(lambda x: torchdiffeq.odeint(func.lorenz, x, t_eval_point, method='rk4'), x)[1]
        #jac_rk4_ad = F.jacobian(lambda x: sol.simulate(0, time_step, x, 2), x)[1]
        jac_rk4_ad = torch.squeeze(jac_rk4_ad)
        
        # Finite Differentiation using Central Difference Approximation
        for i in range(3):
            x_plus = x.clone()
            x_minus = x.clone()

            # create perturbed input
            x_plus[i] = x_plus[i] + eps
            x_minus[i] = x_minus[i] - eps

            # create model output
            m_plus = torchdiffeq.odeint(func.lorenz, x_plus, t_eval_point, method='rk4')[1]
            m_minus = torchdiffeq.odeint(func.lorenz, x_minus, t_eval_point, method='rk4')[1]

            # compute central diff
            diff = m_plus.clone().detach() - m_minus.clone().detach()
            final = diff/2/eps
            jac_rk4_fd[:, i] = final

        print("jac_rk4_fd\n", jac_rk4_fd)

        print(torch.allclose(jac_rk4_ad, jac_rk4_fd, rtol=1e-05))

        diff = jac_rk4_ad - jac_rk4_fd
        diff = diff.reshape(3,3)
        #print("diff: ", diff)
        norm = torch.linalg.matrix_norm(diff)

        return norm

        """
        print("try torchdiffeq.odeint with euler")
        jac_euler = F.jacobian(lambda x: torchdiffeq.odeint(func.lorenz, x, t_eval_point, method='euler'), x)
        print(torch.squeeze(jac_euler)[1])

        
        print("try torchdiffeq.odeint with dopri5")
        jac_dopri5 = F.jacobian(lambda x: torchdiffeq.odeint(func.lorenz, x, t_eval_point, method='dopri5'), x)
        print(torch.squeeze(jac_dopri5)[1])

        
        jacobian_true_ode = F.jacobian(lambda x: sol.simulate(0, time_step, x, 2), x)    
        print("----- JAC_SIMULATE -----")
        squeeze_jac = torch.squeeze(jacobian_true_ode)
        squeeze_jac = squeeze_jac[1]
        print(squeeze_jac, "\n")
        """
    """
    # ----- central difference approximation of jac ----- # 
    # ----- f'(x) = [f(x+h)-f(x-h)]/2h ----- #
    dnode_dx = torch.ones((3, 3)).double()

    for i in range(3):
        x_plus = x.clone()
        x_minus = x.clone()

        # create perturbed input
        x_plus[i] = x_plus[i] + eps
        x_minus[i] = x_minus[i] - eps

        # create model output
        if method == "Euler":
            m_plus = x_plus + deltat*np.array(func.lorenz_jac(x_plus.reshape(1,3)))
            m_minus = x_minus + deltat*np.array(func.lorenz_jac(x_minus.reshape(1,3)))

            # compute central diff
            diff = m_plus.clone().detach() - m_minus.clone().detach()
            final = diff/2/eps
            dnode_dx[:,i] = final

        elif method =="RK4":
            m_plus = sol.simulate(0, time_step, x_plus, 2)
            m_minus = sol.simulate(0, time_step, x_minus, 2)

            # compute central diff
            diff = m_plus[1] - m_minus[1]
            final = diff/2/eps
            dnode_dx[:,i] = final

        # assert
        print(torch.allclose(dnode_dx.double(), squeeze_jac.double(), rtol=1e-05))

    #torch.set_printoptions(precision=10)
    print("----- APPX -----")
    print(dnode_dx.double(), "\n")
    """

    #return



##### ----- test run ----- #####
torch.set_printoptions(sci_mode=True, precision=12)

# create random input
x = torch.rand(3).double()
x.requires_grad = True
print("random x", x)

# train the model
optim_name = 'AdamW'
time_step = 5e-4

print("## --------------- NeuralODE --------------- ##")
test_jac_node(x, optim_name, time_step)

print("## --------------- RK4 --------------- ##")
eps_arr = torch.logspace(start=-7,end=-2,steps=10)
err = torch.zeros(10).double()
for i, eps in enumerate(eps_arr):
    print("eps: ", eps)
    err[i] = test_autodiff(x, eps, time_step, "RK4")

fig, ax = subplots()
ax.semilogx(eps_arr, err, ".", ms = 10.0)
ax.grid(True)
ax.set_xlabel("$\epsilon$", fontsize=20)
ax.set_ylabel("Diff in FD and AD jacobian", fontsize=20)
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
tight_layout()
fig.savefig("jac_test_rk4.png")

# print("## --------------- Euler --------------- ##")
# test_autodiff(x, eps, time_step, "Euler")