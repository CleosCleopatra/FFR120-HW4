
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(17)
a0 = 0.3
a = 0.1 * a0
omega = np.pi / 30
gamma = 1  # Decline rate of the predators due to starvation.
c = 0.1 * gamma

def symplectic_Euler(x0, y0, dt, alpha, beta, gamma, delta, T):
    """
    Function to generate a trajectory of the prey-predator
    system with the symplectic Euler method.
    Here we chose: x explicit and y implicit.
    
    Parameters
    ==========
    x0, y0 : Initial position.
    dt : Time step.
    alpha, beta, gamma, delta : parameters of the model.
    T : Total time of the trajectory.
    """    
     
    N = int(np.ceil(T / dt))
    x = np.zeros(N)
    y = np.zeros(N)
    
    x[0] = x0
    y[0] = y0

    for i in range(N - 1):
        t = i * dt

        alpha = a0 + a * np.cos(omega * t)
        gam_loc = gamma + c * np.cos(omega * t) 

        # Estimate for y[i + 1].
        y_est= y[i] / (1 - dt * (delta * x[i] - gam_loc))
        
        # Use the estimate to determine the next x, y.

        x[i + 1] = x[i] + dt * (alpha - beta * y_est) * x[i]
        y[i + 1] = y[i] + dt * (delta * x[i] - gam_loc) * y_est
    
    return x, y


alpha = 0.3  # Exponential growth rate of the prey.
beta = 0.6  # Decline rate of the preys due to predation.

delta = 1  # Growth rate of the predators due to predation.
dt = 0.1  # Time step.
T = 400  # Total time.

dalpha = 0.1 * alpha
dgamma = 0.1 * gamma
# dgamma = 0 * gamma
phi = 0
# phi = np.pi / 2
# phi = np.pi 

T_change = 12 * 5  # 12 is the period of the system, approximately
omega = 2 * np.pi / T_change

# Equilibrium point.
x_eq = gamma / delta
y_eq = alpha / beta

dx = 0 * x_eq


xt = []
yt = []
t = np.arange(0, T, dt)


x0 = 0.9 * x_eq  # Initial population (preys).
y0 = 0.9 * y_eq  # Initial population (predators).

x, y = symplectic_Euler(x0, y0, dt, alpha, beta, gamma, delta, T)
print(x)
plt.plot(x, y, label=f'trajectory')
plt.plot(x[0], y[0], 'o', color='green')
plt.plot(x[-1], y[-1], 'x', color='red')
plt.plot(x_eq, y_eq, '.', color='k', label='equilibrium')
plt.title('Prey-Predator traj')
plt.legend()
plt.axis('equal')
    
plt.show()





plt.plot(t, x, '-')
plt.legend()
plt.title('prey population')
plt.xlabel('t (time)')
plt.ylabel('x, y (population)')
plt.show()

plt.plot(t, y, '-')
plt.legend()
plt.title('predator population')
plt.xlabel('t (time)')
plt.ylabel('x, y (population)')
plt.show()












import numpy as np
import matplotlib.pyplot as plt

np.random.seed(17)
a0 = 0.3
a = 0.1 * a0
omega = np.pi / 30
gamma = 1  # Decline rate of the predators due to starvation.
c = 0.1 * gamma

def symplectic_Euler(x0, y0, dt, alpha, beta, gamma, delta, T):
    """
    Function to generate a trajectory of the prey-predator
    system with the symplectic Euler method.
    Here we chose: x explicit and y implicit.
    
    Parameters
    ==========
    x0, y0 : Initial position.
    dt : Time step.
    alpha, beta, gamma, delta : parameters of the model.
    T : Total time of the trajectory.
    """    
     
    N = int(np.ceil(T / dt))
    x = np.zeros(N)
    y = np.zeros(N)
    
    x[0] = x0
    y[0] = y0

    for i in range(N - 1):
        t = i * dt

        alpha = a0 + a * np.cos(omega * t)
        gam_loc = gamma + c * np.cos(omega * t) 

        # Estimate for y[i + 1].
        y_est= y[i] / (1 - dt * (delta * x[i] - gam_loc))
        
        # Use the estimate to determine the next x, y.

        x[i + 1] = x[i] + dt * (alpha - beta * y_est) * x[i]
        y[i + 1] = y[i] + dt * (delta * x[i] - gam_loc) * y_est
    
    return x, y


alpha = 0.3  # Exponential growth rate of the prey.
beta = 0.6  # Decline rate of the preys due to predation.

delta = 1  # Growth rate of the predators due to predation.
dt = 0.1  # Time step.
T = 400  # Total time.

dalpha = 0.1 * alpha
dgamma = 0.1 * gamma
# dgamma = 0 * gamma
phi = 0
# phi = np.pi / 2
# phi = np.pi 

T_change = 12 * 5  # 12 is the period of the system, approximately
omega = 2 * np.pi / T_change

# Equilibrium point.
x_eq = gamma / delta
y_eq = alpha / beta

dx = 0 * x_eq


xt = []
yt = []
t = np.arange(0, T, dt)


x0 = 0.1 * x_eq  # Initial population (preys).
y0 = 0.1 * y_eq  # Initial population (predators).

x, y = symplectic_Euler(x0, y0, dt, alpha, beta, gamma, delta, T)
print(x)
plt.plot(x, y, label=f'trajectory')
plt.plot(x[0], y[0], 'o', color='green')
plt.plot(x[-1], y[-1], 'x', color='red')
plt.plot(x_eq, y_eq, '.', color='k', label='equilibrium')
plt.title('Prey-Predator traj, x0=0.1*x_eq, y0=0.1*y_eq')
plt.legend()
plt.axis('equal')
    
plt.show()





plt.plot(t, x, '-')
plt.legend()
plt.title('prey population, x0=0.1*x_eq, y0=0.1*y_eq')
plt.xlabel('t (time)')
plt.ylabel('x, y (population)')
plt.show()

plt.plot(t, y, '-')
plt.legend()
plt.title('predator population, x0=0.1*x_eq, y0=0.1*y_eq')
plt.xlabel('t (time)')
plt.ylabel('x, y (population)')
plt.show()