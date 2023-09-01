import torch
import numpy as np
import torch.autograd.functional as F
import sys
import torchdiffeq
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
    print("created model")
    model = sol.create_NODE(device, n_nodes=3, T=time_step)
    path = "expt_lorenz/"+optim_name+"/"+str(time_step)+'/'+'model.pt'

    print("Loading ... ")
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
        jacobian_true_ode = deltat*F.jacobian(func.lorenz_jac, x)
        jacobian_true_ode[0][0][0] += 1
        jacobian_true_ode[1][0][1] += 1
        jacobian_true_ode[2][0][2] += 1
        print("----- JAC_TRUE -----")
        squeeze_jac = torch.squeeze(jacobian_true_ode)
        print(squeeze_jac)

    elif method == "RK4":
        #lambda solve_lorenz(x) : sol.simulate(0, 0.001, x, 2)
        x_reshaped = x.reshape(3)
        
        print("try torchdiffeq.odeint with rk4")
        t_eval_point = torch.linspace(0, time_step, 2)
        jac_try = F.jacobian(lambda x: torchdiffeq.odeint(func.lorenz, x, t_eval_point, method='rk4'), x_reshaped)
        print(torch.squeeze(jac_try))

        print("try torchdiffeq.odeint with dopri5")
        jac_try = F.jacobian(lambda x: torchdiffeq.odeint(func.lorenz, x, t_eval_point, method='dopri5'), x_reshaped)
        print(torch.squeeze(jac_try))

        
        jacobian_true_ode = F.jacobian(lambda x: sol.simulate(0, time_step, x, 2), x_reshaped)    
        print("----- JAC_TRUE -----")
        squeeze_jac = torch.squeeze(jacobian_true_ode)
        squeeze_jac = squeeze_jac[1]
        print(squeeze_jac)


    # ----- central difference approximation of jac ----- # 
    # ----- f'(x) = [f(x+h)-f(x-h)]/2h ----- #
    dnode_dx = torch.ones((3, 3))

    for i in range(3):
        x_plus = x.clone().T[:,-1]
        x_minus = x.clone().T[:,-1]

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
            m_plus = sol.simulate(0, time_step, x_plus.reshape(3), 2)
            m_minus = sol.simulate(0, time_step, x_minus.reshape(3), 2)

            # compute central diff
            diff = m_plus[1].clone().detach() - m_minus[1].clone().detach()
            final = diff/2/eps
            dnode_dx[:,i] = final

        # assert
        print(torch.allclose(dnode_dx.double(), squeeze_jac.double(), atol=1e-05))

    print("----- APPX -----")
    print(dnode_dx)

    return



##### ----- test run ----- #####

# create random input
x = torch.rand(1, 3)
print("random x", x)

# train the model
optim_name = 'AdamW'
time_step = 1e-2
eps = 1e-1

test_jac_node(x, optim_name, time_step)
print("## ---------- RK4 ---------- ##")
test_autodiff(x, eps, time_step, "RK4")
print("## ---------- Euler ---------- ##")
test_autodiff(x, eps, time_step, "Euler")