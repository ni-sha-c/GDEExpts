from numpy import *
from matplotlib.pyplot import *
def deriv(X, a, b):
    """Return the derivatives dx/dt and dy/dt."""
    x, y = X
    dxdt = a - (1+b)*x + x**2 * y
    dydt = b*x - x**2 * y
    return array([dxdt, dydt])

def simulate(x0, a, b, T, dt):
    x = zeros((T, 2))
    for i in range(T-1):
        x[i+1] = x[i] + dt*deriv(x[i], a, b)
    return x


x0 = random.rand(2)
T = 100000
a, b = 1, 2
dt = 0.2
x_orbit = simulate(x0, a, b, T, dt)


fig, ax = subplots()
T_plot = 10000
ax.plot(x_orbit[-T_plot:,0], x_orbit[-T_plot:,1], ".", ms=5.0)
ax.xaxis.set_tick_params(labelsize=24)
ax.yaxis.set_tick_params(labelsize=24)
ax.grid(True)
ax.set_xlabel("x", fontsize=24)
ax.set_ylabel("y", fontsize=24)

fig1, ax1 = subplots()
T_plot = 1000
ax1.plot(linspace(0,T_plot-1,T_plot)*dt, x_orbit[-T_plot:,0], ".--", ms=5.0)
ax1.xaxis.set_tick_params(labelsize=24)
ax1.yaxis.set_tick_params(labelsize=24)
ax1.grid(True)
ax1.set_xlabel("time", fontsize=24)
ax1.set_ylabel("x", fontsize=24)


