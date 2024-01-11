import torch
import torch.nn as nn
import torchdiffeq
# from .KAF import KAF

# Time Integrator
def solve_odefunc(odefunc, t, y0):
    ''' Solve odefunction using torchdiffeq.odeint() '''

    solution = torchdiffeq.odeint_adjoint(odefunc, y0, t, rtol=1e-9, atol=1e-9, method="rk4")
    final_state = solution[-1]
    return final_state


# class ODEBlock(nn.Module):
#     def __init__(self, T, odefunc:nn.Module, method:str='rk4', rtol:float=1e-9, atol:float=1e-9, adjoint:bool=False):
#         """ Standard ODEBlock class. Can handle all types of ODE functions
#             :method:str = {'euler', 'rk4', 'dopri5', 'adams'}
#         """
#         super().__init__()
#         self.odefunc = odefunc
#         self.method = method
#         self.adjoint_flag = adjoint
#         self.atol, self.rtol = atol, rtol
#         self.T = T

#     def forward(self, x:torch.Tensor): #T:float=0.0025
#         self.integration_time = torch.tensor([0, self.T]).float()
#         self.integration_time = self.integration_time.type_as(x)

#         if self.adjoint_flag:
#             out = torchdiffeq.odeint_adjoint(self.odefunc, x, self.integration_time,
#                                              rtol=self.rtol, atol=self.atol, method=self.method)
#         else:
#             out = torchdiffeq.odeint(self.odefunc, x, self.integration_time,
#                                      rtol=self.rtol, atol=self.atol, method=self.method)

#         return out[-1]


class ODE_Lorenz(nn.Module):
    '''Define Neural Network that approximates differential equation system of Chaotic Lorenz'''

    def __init__(self, y_dim=3, n_hidden=32*9):
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


class ODE_Sin (nn.Module):

  def __init__( self , y_dim=2 , n_hidden=4) :
    super(ODE_Sin , self ).__init__()
    self.net = nn.Sequential(
      nn.Linear(1, 32 * 9),
      nn.GELU(),
      nn.Linear(32 * 9, 64 * 9),
      nn.GELU(),
      nn.Linear(64 * 9, 1)
    )

  def forward(self , t, y): 
    # because of how odeint works, t is necessary here!
    traj = torchdiffeq.odeint(self.net, y, torch.tensor([0, 0.01]), method='rk4', rtol=1e-8) 
    return traj



class ODE_Tent (nn.Module):

  def __init__( self , y_dim=1 , n_hidden=4) :
    super(ODE_Tent , self ).__init__()
    self.net = nn.Sequential(
      nn.Linear(y_dim, n_hidden),
      nn.ReLU(),
      nn.Linear(n_hidden, n_hidden),
      nn.ReLU(),
      nn.Linear(n_hidden, n_hidden),
      nn.ReLU(),
      nn.Linear(n_hidden, n_hidden),
      nn.ReLU(),
      nn.Linear(n_hidden, y_dim)
    )

  def forward(self , t, y): 
    return self.net(y)



class ODE_Brusselator (nn.Module):
  
  def __init__( self , y_dim=2 , n_hidden=4) :
    super(ODE_Brusselator , self ).__init__()
    self.net = nn.Sequential(
      nn.Linear(y_dim, 40*9),
      nn.GELU(),
      nn.Linear(40*9, 40*9),
      nn.GELU(),
      nn.Linear(40*9, y_dim)

      # nn.Linear(y_dim, 64),
      # nn.Tanh(),
      # nn.Linear(64, 512),
      # nn.Tanh(),
      # nn.Linear(512, 512),
      # nn.Tanh(),
      # nn.Linear(512, 256),
      # nn.Tanh(),
      # nn.Linear(256, 64),
      # nn.Tanh(),
      # nn.Linear(64, y_dim)
    )

  def forward(self , t, y): 
    return self.net(y)



# class ODEFunc_Lorenz (nn.Module):
#   ''' adapted from ... '''
  
#   def __init__( self , y_dim=3 , n_hidden=4) :
#     super(ODEFunc_Lorenz , self ).__init__()

#     self.net = nn.Sequential(
#       nn.Linear(3, 32*9),
#       nn.GELU(),
#       nn.Linear(32*9, 64*9),
#       nn.GELU(),
#       nn.Linear(64*9, 3)
#       # other activation function lists:
#       # nn.SiLU(),
#       # KAF(256),
#     )

#   def forward(self , t, y): 
#     #torch.set_grad_enabled(True) 
#     res = self.net(y)
#     # #print("at t: ", t, res)

#     return res


class ODE_Lorenz_periodic (nn.Module):
  
  def __init__( self , y_dim=3 , n_hidden=4) :
    super(ODE_Lorenz_periodic , self ).__init__()
    self.net = nn.Sequential(
      nn.Linear(3, 32*9),
      nn.GELU(),
      nn.Linear(32*9, 64*9),
      nn.GELU(),
      nn.Linear(64*9, 3)

      # nn.Linear(y_dim, 40*9),
      # nn.GELU(),
      # nn.Linear(40*9, 20*9),
      # nn.GELU(),
      # nn.Linear(20*9, 20*9),
      # nn.GELU(),
      # nn.Linear(20*9, y_dim)


      # nn.Linear(y_dim, 20*9),
      # nn.GELU(),
      # nn.Linear(20*9, 30*9),
      # nn.GELU(),
      # nn.Linear(30*9, 20*9),
      # nn.GELU(),
      # nn.Linear(20*9, y_dim)

      # nn.Linear(y_dim, 9),
      # nn.SiLU(),
      # nn.Linear(9, 4*9),
      # nn.SiLU(),
      # nn.Linear(4*9, 4*9),
      # nn.SiLU(),
      # nn.Linear(4*9, 4*9),
      # nn.SiLU(),
      # nn.Linear(4*9, 9),
      # nn.SiLU(),
      # nn.Linear(9, y_dim)

      # nn.Linear(y_dim, 16*9),
      # nn.SiLU(),
      # nn.Linear(16*9, 32*9),
      # nn.SiLU(),
      # nn.Linear(32*9, 64*9),
      # nn.SiLU(),
      # nn.Linear(64*9, 32*9),
      # nn.SiLU(),
      # nn.Linear(32*9, 16*9),
      # nn.SiLU(),
      # nn.Linear(16*9, y_dim)

      # nn.Linear(y_dim, 256),
      # nn.SiLU(),
      # nn.Linear(256, 1024),
      # nn.SiLU(),
      # nn.Linear(1024, 512),
      # nn.SiLU(),
      # nn.Linear(512, 512),
      # nn.SiLU(),
      # nn.Linear(512, 512),
      # nn.SiLU(),
      # nn.Linear(512, 256),
      # nn.SiLU(),
      # nn.Linear(256, y_dim)
    )

  def forward(self, t, y): 
    return self.net(y)