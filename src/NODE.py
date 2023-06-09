import torch
import torch.nn as nn
import torchdiffeq

class ODEBlock(nn.Module):
    def __init__(self, T, odefunc:nn.Module, method:str='euler', rtol:float=1e-6, atol:float=1e-6, adjoint:bool=True):
        """ Standard ODEBlock class. Can handle all types of ODE functions
            :method:str = {'euler', 'rk4', 'dopri5', 'adams'}
        """
        super().__init__()
        self.odefunc = odefunc
        self.method = method
        self.adjoint_flag = adjoint
        self.atol, self.rtol = atol, rtol
        self.T = T

    def forward(self, x:torch.Tensor): #T:float=0.0025
        self.integration_time = torch.tensor([0, self.T]).float()
        self.integration_time = self.integration_time.type_as(x)

        if self.adjoint_flag:
            out = torchdiffeq.odeint_adjoint(self.odefunc, x, self.integration_time,
                                             rtol=self.rtol, atol=self.atol, method=self.method)
        else:
            out = torchdiffeq.odeint(self.odefunc, x, self.integration_time,
                                     rtol=self.rtol, atol=self.atol, method=self.method)

        return out[-1]
    

class ODEFunc_Sin (nn.Module):
  ''' adapted from ... '''

  def __init__( self , y_dim=2 , n_hidden=4) :
    super(ODEFunc_Sin , self ).__init__()
    self.net = nn.Sequential(
      nn.Linear(y_dim, n_hidden),
      nn.Tanh(),
      nn.Linear(n_hidden, n_hidden),
      nn.Tanh(),
      nn.Linear(n_hidden, y_dim)
    )

  def forward(self , t, y): 
    # because of how odeint works, t is necessary here!
    return self.net(y)



class ODEFunc_Brusselator (nn.Module):
  ''' adapted from ... '''
  
  def __init__( self , y_dim=2 , n_hidden=4) :
    super(ODEFunc_Brusselator , self ).__init__()
    self.net = nn.Sequential(
      nn.Linear(y_dim, 64),
      nn.Tanh(),
      nn.Linear(64, 512),
      nn.Tanh(),
      nn.Linear(512, 512),
      nn.Tanh(),
      nn.Linear(512, 256),
      nn.Tanh(),
      nn.Linear(256, 64),
      nn.Tanh(),
      nn.Linear(64, y_dim)
    )

  def forward(self , t, y): 
    return self.net(y)


class ODEFunc_Lorenz (nn.Module):
  ''' adapted from ... '''
  
  def __init__( self , y_dim=3 , n_hidden=4) :
    super(ODEFunc_Lorenz , self ).__init__()
    self.net = nn.Sequential(
      nn.Linear(y_dim, 256),
      nn.Tanh(),
      nn.Linear(256, 512),
      nn.Tanh(),
      nn.Linear(512, 512),
      nn.Tanh(),
      nn.Linear(512, 256),
      nn.Tanh(),
      nn.Linear(256, y_dim)
    )

  def forward(self , t, y): 
    return self.net(y)