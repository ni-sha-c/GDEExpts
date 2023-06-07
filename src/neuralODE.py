import torch
import torch.nn as nn
import torchdiffeq

class ODEBlock(nn.Module):
    def __init__(self, odefunc:nn.Module, method:str='euler', rtol:float=1e-5, atol:float=1e-5, adjoint:bool=True):
        """ Standard ODEBlock class. Can handle all types of ODE functions
            :method:str = {'euler', 'rk4', 'dopri5', 'adams'}
        """
        super().__init__()
        self.odefunc = odefunc
        self.method = method
        self.adjoint_flag = adjoint
        self.atol, self.rtol = atol, rtol

    def forward(self, x:torch.Tensor, T:float=0.2):
        self.integration_time = torch.tensor([0, T]).float()
        self.integration_time = self.integration_time.type_as(x)

        if self.adjoint_flag:
            out = torchdiffeq.odeint_adjoint(self.odefunc, x, self.integration_time,
                                             rtol=self.rtol, atol=self.atol, method=self.method)
        else:
            out = torchdiffeq.odeint(self.odefunc, x, self.integration_time,
                                     rtol=self.rtol, atol=self.atol, method=self.method)

        return out[-1]
    

class ODEFunc (nn.Module):
  ''' adapted from ... '''
  def __init__( self , y_dim=2 , n_hidden=4) :
    super(ODEFunc , self ).__init__()
    self.net = nn.Sequential(
      nn.Linear(y_dim, 16),
      # nn.ReLU(),
      nn.Linear(16 , y_dim)
    )

  def forward(self , t, y): # because of how odeint works, t is necessary here!
    return self.net(y)