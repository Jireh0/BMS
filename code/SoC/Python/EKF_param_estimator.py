import numpy as np

class EKFParamEstimator:
    def __init__(self, theta0, Q_theta, R, g_func):
        """
        EKF for parameters estimation (e.g. R0, Qn) given known x, u, and y
        
        Parameters:
        - theta0: initial parameter guess (1D np.array)
        - Q_theta: process noise covariance (matrix)
        - R: measurement noise (scalar)
        - g_func: function g(x, u, theta)
        - dg_dtheta_func: Jacobian dg/dtheta (returns 1D array) 
        """
        self.theta = theta0
        self.P = np.eye(len(theta0)) * 1e-3  # initial covariance
        self.Q = Q_theta  # Q_theta = np.array([[1e-6 0], [0 1e-4]]) in the paper
        self.R = R  # R = 10 in the paper
        self.g_func = g_func
        # self.dg_dtheta_func = dg_dtheta_func

        self.history  = [theta0.copy()]

    def step(self, x, u, y_meas):
        """
        Perform one EKF step
        
        """
        # Prediction
        theta_ = self.theta.copy()
        P_ = self.P + self.Q

        # Update using measurement data y
        y_ = self.g_func(x,u,theta_)
        # TODO 完成对H的计算，按照论文的格式
        S = H @ P_ @ H.T + self.R
        K = P_ @ H.T @ np.linalg.inv(S)
        self.theta = theta_ + (K.flatten() * (y_meas - y_))
        I = np.eye(len(self.theta))
        self.P = (I - K @ H) @ P_