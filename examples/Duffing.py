import torch

def duffing(t, X):

#    x'' + delta x' + alpha x + beta x^3 = gamma * cos ( omega * t )
#
#    y1' = y2
#    y2' = - delta y2 - alpha y1 - beta y1^3 + gamma * cos ( omega * t )

    x1, x2 = X
    delta = 
    alpha = 
    beta = 
    gamma = 
    omega = 

    dx1dt = x2
    dx2dt = - delta * x2 - alpha * x1 - beta * x1**3 + gamma * torch.cos(omega * t)

    res = torch.tensor([ dx1dt, dx2dt ])

    return res