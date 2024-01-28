import torch

def lv(t, X):

    x,y = X

    alpha = 0.67
    beta = 1.33
    delta = 1
    gamma = 1


    dx = alpha*x - beta*x*y
    dy = delta*x*y - gamma*y
    return torch.stack([dx, dy])

def gm(t, X):
    # discrete time
    x, y = X
    a=-1.1
    b=-0.2
    mu=-2.0

    def Gumowski(x, mu):
        g = x*mu + 2*x*x*(1 - mu) / (1 + x*x)
        return g


    dx= a*y*(1 - b*y*y) + y + Gumowski(x, mu)
    dy= -x + Gumowski(dx, mu)
    return torch.stack([dx, dy])


# def vdp(t, X):
#     '''https://en.wikipedia.org/wiki/Van_der_Pol_oscillator'''
#     x,y = X

#     mu = 2

#     dx = mu * (x - (x**3)/3 - y)
#     dy = x / mu


#     return torch.stack([dx, dy])
