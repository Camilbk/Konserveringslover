import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation

''' Spatial grid '''
x_sim = 3 # m  # length of grid
dx = 0.01   # m
nx = int(x_sim/dx) + 1


# CFL condition: 2*dt max|fdiff(u)| < dx, we need this to hold

''' Time grid '''
t_sim = 3   # s  # time lenght
dt = 0.001      # s
nt = int(t_sim/dt)


''' Flux function: f'(u) '''
def gdiff(u_i):
    #non-linear
    numerator = -2*(u_i-1)*u_i
    denominator = (2*u_i**2 -2*u_i+1)**2
    return numerator/denominator

def fdiff(u_i):
    return np.ones(len(u_i))
    #return np.exp(u_i)

def pdiff(u_i): #burgers eqn.
    return u_i



''' Results '''
u = np.zeros((nt, nx))
xgrid, tgrid = np.meshgrid(np.linspace(0, x_sim, nx), np.linspace(0, t_sim, nt))

x = np.array(xgrid[0, :])
t = np.array(tgrid[:, 0])

''' Initial and boundary conditions '''
def initial_condition(u):
    #we have zeros by initialization
    midpoint = int(len(x - 1)/2)
    #typical riemann problem
    eps = 10
    u[0,:midpoint-eps] = -1
    u[0, (midpoint - eps):(eps + midpoint) ] = x[-eps+midpoint:eps+midpoint]/eps
    u[0, eps + midpoint:] = 1
    #u[0, :midpoint] = -1
    #u[0, midpoint:] = 1
    print(u[0])
    return u

def boundary(u, n ):
    u[n, 0] = -1
    u[n, -1] = 1
    return u

#print(u[0, :])


''' Simulation '''
'''
u( x_j, t_n+1  ) = u[n, 1:-1]
u( x_j, t_n  ) = u[n - 1, 1:-1]

u( x_j+1, t_n  ) = u[n-1, 0:-2]
u( x_j-1, t_n  ) = u[n-1, 2:]

'''

def init(line):  # only required for blitting to give a clean slate.
    line.set_ydata([np.nan] * len(u))
    return line,

def animate(line, i):
    line.set_ydata(u[i,:])  # update the data.
    print("hei")
    return line,

def naiveMethod(u, flux_func):
    # space: central finite differences
    # time: forward Euler
    u = initial_condition(u)
    if flux_func == 'fdiff':
        for n in range(1, nt):
            u[n, 1:-1] = u[n-1, 1:-1] - (fdiff(u[n-1, 1:-1]) * (u[n-1, 0:-2] - u[n-1, 2:]) * dt/(2*dx))
            u = boundary(u, n)

    elif flux_func == 'pdiff':
        for n in range(1, nt):
            u[n, 1:-1] = u[n - 1, 1:-1] - (pdiff(u[n - 1, 1:-1]) * (u[n - 1, 0:-2] - u[n - 1, 2:]) * dt / (2 * dx))
            u = boundary(u, n)

    elif flux_func == 'gdiff':
        for n in range(1, nt):
            u[n, 1:-1] = u[n - 1, 1:-1] - (gdiff(u[n - 1, 1:-1]) * (u[n - 1, 0:-2] - u[n - 1, 2:]) * dt / (2 * dx))
            u = boundary(u, n)

    fig, ax = plt.subplots()
    c = ax.pcolormesh(x, t, u, cmap='RdBu', vmin=u.min(), vmax=u.max())
    ax.set_title('Naive Method')
    plt.xlabel("x")
    plt.ylabel("t")
    # set the limits of the plot to the limits of the data
    ax.axis([x.min(), x.max(), t.min(), t.max()])
    fig.colorbar(c, ax=ax)



def Lax_Friedricks(u, flux_func):
    u = initial_condition(u)
    if flux_func == 'fdiff':
        for n in range(1, nt):
            u[n, 1:-1] = 0.5 * (u[n-1, 0:-2] + u[n-1, 2:]) - dt/(2*dx)*( fdiff( u[n-1, 0:-2] ) - fdiff( u[n-1, 2:] ))
            u = boundary(u, n)

    elif flux_func == 'pdiff':
        u = boundary(u,1)
        for n in range(1, nt):
            u[n, 1:-1] = 0.5 * (u[n-1, 0:-2] + u[n-1, 2:]) - dt/(2*dx)*( pdiff( u[n-1, 0:-2] ) - pdiff( u[n-1, 2:] ))
            #u = boundary(u, n)

    elif flux_func == 'gdiff':
        for n in range(1, nt):
            u[n, 1:-1] = 0.5 * (u[n-1, 0:-2] + u[n-1, 2:]) - dt/(2*dx)*( gdiff( u[n-1, 0:-2] ) - gdiff( u[n-1, 2:] ))
            u = boundary(u, n)


    fig, ax = plt.subplots()
    c = ax.pcolormesh(x, t, u, cmap='RdBu', vmin=u.min(), vmax=u.max())
    ax.set_title('Lax-Friedricks ')
    # set the limits of the plot to the limits of the data
    ax.axis([x.min(), x.max(), t.min(), t.max()])
    plt.xlabel("x")
    plt.ylabel("t")
    fig.colorbar(c, ax=ax)
    plt.show()


    #line = init(u)
    #ani = animation.FuncAnimation(fig, animate, init_func=init, interval=2, blit=True, save_count=50)
    #plt.title("snapshot of u at a time")
    #plt.xlabel("x")
    #plt.ylabel("u")
    #plt.show()




def Gordunov(u, flux_func):
    u = initial_condition(u)
    for n in range(1, nt):
        u[n, 1:-1] = u[n-1, 1:-1] - dt/dx * ((fdiff( u[n-1, 0:-2] ) - fdiff(u[n - 1, 1:-1]))/2 - (fdiff(u[n - 1, 1:-1] ) - fdiff(u[n-1, 2:]))/2)
        #u[n, 1:-1] = u[n - 1, 1:-1] - dt / dx * (fdiff(u[n - 1, 0:-2])  - fdiff(u[n - 1, 2:]))
        u = boundary(u, n)

    fig, ax = plt.subplots()
    c = ax.pcolormesh(x, t, u, cmap='RdBu', vmin=u.min(), vmax=u.max())
    ax.set_title('Gordunov ')
    # set the limits of the plot to the limits of the data
    ax.axis([x.min(), x.max(), t.min(), t.max()])
    plt.xlabel("x")
    plt.ylabel("t")
    fig.colorbar(c, ax=ax)


#naiveMethod(u, 'fdiff')
Lax_Friedricks(u, 'fdiff')
#Gordunov(u, 'fdiff')
plt.show()


