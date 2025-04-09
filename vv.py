import numpy as np
import matplotlib.pyplot as plt

# Physical constants
k = 1.0          # spring constant (harmonic potential)
m = 1.0          # mass
dt = 0.01        # time step
T = 20.0         # total simulation time
N = int(T / dt)  # number of time steps

# Initialize arrays
x = np.zeros(N)
v = np.zeros(N)
a = np.zeros(N)
t = np.linspace(0, T, N)
E = np.zeros(N)

# Initial conditions
x[0] = 1.0       # initial position
v[0] = 0.0       # initial velocity
a[0] = -k * x[0] / m  # initial acceleration

# Velocity Verlet Integration Loop
for i in range(1, N):
    # Step 1: position update
    x[i] = x[i-1] + v[i-1]*dt + 0.5*a[i-1]*dt**2

    # Step 2: compute new acceleration
    a_new = -k * x[i] / m

    # Step 3: velocity update
    v[i] = v[i-1] + 0.5*(a[i-1] + a_new)*dt

    # Store new acceleration
    a[i] = a_new

    # Compute total energy: E = K + V
    kinetic = 0.5 * m * v[i]**2
    potential = 0.5 * k * x[i]**2
    E[i] = kinetic + potential

# Plotting results
plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.plot(t, x, label='Position', color='blue')
plt.ylabel('x(t)')
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(t, v, label='Velocity', color='green')
plt.ylabel('v(t)')
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(t, E, label='Total Energy', color='red')
plt.ylabel('E(t)')
plt.xlabel('Time')
plt.grid()

plt.tight_layout()
plt.suptitle("Velocity Verlet MD: Simple Harmonic Oscillator", y=1.02)
plt.show()
