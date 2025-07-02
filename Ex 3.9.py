import numpy as np
import matplotlib.pyplot as plt

def alpha(Xt):
    return Xt

def beta(Xt):
    return 2*Xt

def euler(alpha, beta, min_t, max_t, dt, X0):
    t = np.arange(min_t, max_t + dt, dt)
    X = np.zeros(len(t))
    Bt = np.zeros(len(t))  # Brownian motion increments
    X[0] = X0
    for i in range(1, len(t)):
        dW = np.random.normal(0, np.sqrt(dt))  # Brownian increment
        Bt[i] = Bt[i-1] + dW
        X[i] = X[i-1] + alpha(X[i-1]) * dt + beta(X[i-1]) * dW
    
    return t, X, Bt

min_t = 0
max_t = 1
dt = 1/1000
X0 = 1.0

count = 0
t14 = int(0.25//dt)
t12 = int(0.5//dt)
t34 = int(0.75//dt)
avg_D = np.zeros(4)

plt.figure(figsize=(10, 6))
for i in range(20):
    t, X, Bt = euler(alpha, beta, min_t, max_t, dt, X0)
    plt.plot(t, X)
    print(f'X_{1/4}: {X[t14]}, X_{1/2}: {X[t12]}, X_{3/4}: {X[t34]}, X_{1}: {X[-1]}')
    print(f'B_{1/4}: {Bt[t14]}, B_{1/2}: {Bt[t12]}, B_{3/4}: {Bt[t34]}, B_{1}: {Bt[-1]}')
    avg_D[0] += X[t14]- Bt[t14]
    avg_D[1] += X[t12]- Bt[t12]
    avg_D[2] += X[t34]- Bt[t34]
    avg_D[3] += X[-1] - Bt[-1]
plt.title('Euler-Maruyama Simulations of SDE')
plt.xlabel('Time')
plt.ylabel('$X_t$')
plt.legend()
plt.show()

print(f'Average X_t-B_t at t=1/4: {avg_D[0]/20}, at 1/2: {avg_D[1]/20}, at 3/4: {avg_D[2]/20}, at 1: {avg_D[3]/20}')