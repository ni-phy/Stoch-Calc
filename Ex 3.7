import numpy as np
import matplotlib.pyplot as plt

def alpha(Xt):
    return -2*Xt

def beta(Xt):
    return 2*Xt

def euler(alpha, beta, min_t, max_t, dt, X0):
    t = np.arange(min_t, max_t + dt, dt)
    X = np.zeros(len(t))
    X[0] = X0
    
    for i in range(1, len(t)):
        dW = np.random.normal(0, np.sqrt(dt))  # Brownian increment
        X[i] = X[i-1] + alpha(X[i-1]) * dt + beta(X[i-1]) * dW
    
    return t, X

min_t = 0
max_t = 2
dt = 0.005
X0 = 1.0

### 3.7.1 ###
# Plot one graph the solution of the SDE

t, X = euler(alpha, beta, min_t, max_t, dt, X0)

plt.plot(t, X, label='Euler-Maruyama')
plt.title('Euler-Maruyama Simulation of SDE')
plt.xlabel('Time')
plt.ylabel('$X_t$')
plt.legend()
plt.show()

### 3.7.2 ###
count = 0
plt.figure(figsize=(10, 6))
for i in range(10000):
    t, X = euler(alpha, beta, min_t, max_t, dt, X0)
    plt.plot(t, X)
    if X[-1]>3:
        count+=1
plt.title('Euler-Maruyama Simulations of SDE')
plt.xlabel('Time')
plt.ylabel('$X_t$')
plt.legend()
plt.show()
print(f'$P(X_2>3)=$: {count/10000}')
