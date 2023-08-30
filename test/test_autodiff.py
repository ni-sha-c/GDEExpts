import torch
import numpy as np
import torch.autograd.functional as F
import sys
import torchdiffeq
sys.path.append('..')

from src import NODE_solve_Lorenz as sol 
from src import NODE_util as util
from examples import Lorenz as func

def test_jac_node(x, eps):
    #jac_node_x
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = x.to(device)
    m = sol.create_NODE(device, n_nodes=3, T=0.001)
    jacobian_node = F.jacobian(lambda t: m(t), x)

    print("----- JAC_NODE -----")
    print(jacobian_node)
    return

def test_autodiff(x, eps):
    #jac_ode_x
    jacobian_true_ode = F.jacobian(func.lorenz_jac, x)
    print("----- JAC_TRUE -----")
    print(jacobian_true_ode)


    # central difference approximation
    # f'(x) = [f(x+h)-f(x-h)]/2h
    dnode_dx = torch.ones(3, 3)

    for i in range(3):
        x_plus = x.clone()
        x_minus = x.clone()

        # create perturbed input
        x_plus[:, i] = x_plus[:, i] + eps
        x_minus[:, i] = x_minus[:, i] - eps
        
        # create model output
        m_plus = sol.simulate(0, 0.001, x_plus.reshape(3), 2)
        m_minus = sol.simulate(0, 0.001, x_minus.reshape(3), 2)

        # compute central diff
        final = np.abs(m_plus[1] - m_minus[1])/2*eps
        dnode_dx[:,i] = final
    
    print("----- APPX -----")
    print(dnode_dx)
    return

##### ----- test run ----- #####
x = torch.rand(1, 3)
print("random x", x)
test_jac_node(x, 1e-5)
test_autodiff(x, 1e-5)