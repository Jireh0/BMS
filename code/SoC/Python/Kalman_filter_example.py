import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(1,100,100)
# Initialize
u = 0.6 # acceleration
v0 = 5  # initial velocity
s0 = 0  # initial position
X_true = np.array([[s0], [v0]])
size = t.shape[0] + 1  # time step
dims = X_true.shape[0]
# covariance matrix of noise for process (model) and measurement
Q = np.array([[1e1,0],[0,1e1]])
R = np.array([[1e4,0],[0,1e4]])
X = X_true
P = np.array([[0.1,0],[0,0.1]])  # empirical

# Transition Matrix
A = np.array([[1,1],[0,1]])
B = np.array([[1/2],[1]])
H = np.array([[1,0],[0,1]])

real_pos = np.array([0] * size)
real_pos[0] = s0
real_vel = np.array([0] * size)

measure_pos = np.array([0] * size)
measure_pos[0] = real_pos[0] + np.random.normal(0,R[0][0]**0.5)
measure_vel = np.array([0] * size)
# initial optimal values are determined by the observation
optim_pos = np.array([0] * size)
optim_pos[0] = measure_pos[0]
optim_vel = np.array([0] * size)

for i in range(1, size):
    # current position, velocity, measurement ...
    w = np.array([[np.random.normal(0,Q[0][0]**0.5)], [np.random.normal(0,Q[1][1]**0.5)]])
    X_true = A @ X_true + B * u + w
    real_pos[i] = X_true[0]
    real_vel[i] = X_true[1]
    v = np.array([[np.random.normal(0,R[0][0]**0.5)], [np.random.normal(0,R[1][1]**0.5)]])
    Z = H @ X_true + v
    # prediction
    X_ = A @ X + B * u
    P_ = A @ P @ A.T + Q
    # update
    K = P_ @ H.T @ np.linalg.inv(H @ P_ @ H.T + R)
    X = X_ + K @ (Z - H @ X_)
    I = np.identity(dims)
    P = (I - K @ H) @ P_

    optim_pos[i] = X[0][0]
    optim_vel[i] = X[1][0]
    measure_pos[i] = Z[0]
    measure_vel[i] = Z[1]

t = np.concatenate((np.array([0]), t))
plt.plot(t,real_pos,label='real positions')
plt.plot(t,measure_pos,label='measured positions')    
plt.plot(t,optim_pos,label='kalman filtered positions')

plt.legend()
plt.show()

plt.plot(t,real_vel,label='real velocity')
plt.plot(t,measure_vel,label='measured velocity')    
plt.plot(t,optim_vel,label='kalman filtered velocity')

plt.legend()
plt.show()