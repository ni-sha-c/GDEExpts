import torch as th
from torch.fft import fft, ifft
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.pyplot import cm

''' returns solution not rhs '''

def ks(N, x):
    # Spatial grid and initial condition
    # u = th.cos(x/16)*(1+th.sin(x/16))
    u = x
    v = fft(u)

    # Precompute ETDRK4 scalar quantities
    h = 1/4
    k = th.cat([th.arange(0,N/2),th.tensor([0.]),th.arange(-N/2+1,0)],0)/16
    L = k**2 - k**4 
    E = (h*L).exp()
    E2 = (h*L/2).exp()
    M = 16
    r = (1j*th.pi*(th.arange(1,M+1)-.5)/M)
    LR = h*L[:,None].repeat_interleave(M,1) + r[None,:].repeat_interleave(N,0)
    Q = h*(((LR/2).exp()-1)/LR).mean(dim=1).real
    f1 = h*((-4-LR+LR.exp()*(4-3*LR+LR**2))/LR**3).mean(dim=1).real
    f2 = h*((2+LR+LR.exp()*(-2+LR))/LR**3).mean(dim=1).real
    f3 = h*((-4-3*LR-LR**2+LR.exp()*(4-LR))/LR**3).mean(dim=1).real

    # Timestepping
    uu = [u]
    tt = [0]
    tmax = 600
    nmax = int(tmax/h)
    nplt = int((tmax/100)/h)
    g = -.5j*k

    for n in range(1,nmax+1):
        t = n*h
        Nv = g * fft(ifft(v).real**2)
        a = E2*v + Q*Nv
        Na = g * fft(ifft(a).real**2)
        b = E2*v + Q*Na
        Nb = g * fft(ifft(b).real**2)
        c = E2*a + Q*(2*Nb-Nv)
        Nc = g * fft(ifft(c).real**2)
        v = E*v + Nv*f1 + 2*(Na+Nb)*f2 + Nc*f3
        if n%nplt==0:
            u = ifft(v).real
            uu.append(u)
            tt.append(t) 

    uu = th.stack(uu)
    tt = th.tensor(tt)
    return uu, tt, L
# print(uu.shape, tt.shape)

# plt.imshow(uu)
# plt.show()

N = 128
# x = 32*th.pi*th.arange(1,N+1)/N
x = -0.5 + th.rand(N)

uu, t, L = ks(N, x)

# plot the result
fig, ax = plt.subplots(figsize=(10,8))
# x = np.arange(0, L+dx, dx)
# t = np.arange9(0, T+dt, dt)
t = t.detach().cpu().numpy()
print("t", t.shape)

xx, tt = np.meshgrid(x, t)
u = uu.detach().cpu().numpy()
levels = np.arange(-3, 3, 0.01)

cs = ax.contourf(xx, tt, u.T, cmap=cm.jet)
fig.colorbar(cs)

ax.set_xlabel("x")
ax.set_ylabel("t")
ax.set_title(f"Kuramoto-Sivashinsky: L = {L}")
fig.savefig("../plot/KS_new.png")