
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(17)


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
        # Estimate for y[i + 1].
        y_est= y[i] / (1 - dt * (delta * x[i] - gamma))
        
        # Use the estimate to determine the next x, y.
        x[i + 1] = x[i] + dt * (alpha - beta * y_est) * x[i]
        y[i + 1] = y[i] + dt * (delta * x[i] - gamma) * y_est
    
    return x, y

def symplectic_Euler_tdep(x0, y0, dt, alpha0, beta, gamma0, delta, T, omega, dalpha, dgamma, phi):
    """
    Function to generate a trajectory of the prey-predator
    system with the symplectic Euler method.
    Here we chose: x explicit and y implicit.
    
    Parameters
    ==========
    x0, y0 : Initial position.
    dt : Time step.
    alpha0, beta, gamma0, delta : parameters of the model.
    T : Total time of the trajectory.
    """    
     
    N = int(np.ceil(T / dt))
    x = np.zeros(N)
    y = np.zeros(N)
    
    x[0] = x0
    y[0] = y0

    for i in range(N - 1):
        alpha = alpha0 + dalpha * np.cos(omega * i * dt)
        gamma = gamma0 + dgamma * np.cos(omega * i * dt + phi)
        # Estimate for y[i + 1].
        y_est= y[i] / (1 - dt * (delta * x[i] - gamma))
        
        # Use the estimate to determine the next x, y.
        x[i + 1] = x[i] + dt * (alpha - beta * y_est) * x[i]
        y[i + 1] = y[i] + dt * (delta * x[i] - gamma) * y_est
    
    return x, y


alpha = 0.3  # Exponential growth rate of the prey.
beta = 0.6  # Decline rate of the preys due to predation.
gamma = 1  # Decline rate of the predators due to starvation.
delta = 1  # Growth rate of the predators due to predation.
dt = 0.1  # Time step.
T = 60  # Total time.

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
list_dy = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]) * y_eq * (-1)
num_dy = list_dy.size

x_ref = np.zeros([num_dy, int(np.ceil(T / dt))])
y_ref = np.zeros([num_dy, int(np.ceil(T / dt))])

x_tdep = np.zeros([num_dy, int(np.ceil(T / dt))])
y_tdep = np.zeros([num_dy, int(np.ceil(T / dt))])

xt = []
yt = []
t = np.arange(0, T, dt)

for j, dy in enumerate(list_dy):
    x0 = x_eq  # Initial population (preys).
    y0 = y_eq + dy  # Initial population (predators).

    x, y = symplectic_Euler(x0, y0, dt, alpha, beta, gamma, delta, T)
    plt.plot(x, y, label=f'dy={dy:.2f}')
    plt.plot(x[0], y[0], 'o', color='green')
    plt.plot(x[-1], y[-1], 'x', color='red')
    xt.append(x)
    yt.append(y)


plt.plot(x_eq, y_eq, '.', color='k', label='equilibrium')
plt.title('Prey-Predator traj')
plt.legend()
plt.axis('equal')
    
plt.show()



plt.figure(figsize=(10, 5))
for i in range(len(xt)):
    plt.plot(t, xt[i], '-', label=f"y={dy}")
    plt.legend()
    plt.title('prey population')
    plt.xlabel('t (time)')
    plt.ylabel('x, y (population)')
plt.show()

for i in range(len(xt)): 
    plt.plot(t, yt[i], '-', label=f"y={dy}")
    plt.legend()
    plt.title('predator population')
    plt.xlabel('t (time)')
    plt.ylabel('x, y (population)')
plt.show()