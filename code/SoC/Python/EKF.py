import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
from EKF_param_estimator import EKFParamEstimator

class EKFSOC:
    def __init__(self, data_path = 'matlab/EKF', Ts = 0.1, Qn = 30.23 * 3600, poly_order=8):
        self.data_path = data_path
        self.Ts = Ts
        self.Qn = Qn
        self.poly_order = poly_order

        self._load_data()
        self._prepare_mpdel()
        self._init_param_estimator()
    

    def _load_data(self):
        try:
            # 数据加载
            self.R1 = sio.loadmat(f'{self.data_path}/R1.mat')['R1'].item() + 0.0007
            self.R2 = sio.loadmat(f'{self.data_path}/R2.mat')['R2'].item() + 0.0005
            self.R0 = sio.loadmat(f'{self.data_path}/R0.mat')['R0'].item()
            self.C1 = sio.loadmat(f'{self.data_path}/C1.mat')['C1'].item()
            self.C2 = sio.loadmat(f'{self.data_path}/C2.mat')['C2'].item()
            discharge = sio.loadmat(f'{self.data_path}/discharge.mat')['discharge'] 
            OCV_SOC = sio.loadmat(f'{self.data_path}/OCV_SOC.mat')['OCV_SOC']
        except FileNotFoundError:
            print(os.getcwd())
            raise

        # 数据提取和拟合
        self.tm = discharge[0,:].T
        self.Cur = -discharge[1,:].T
        self.Vot = discharge[2,:].T
        self.RSOC = discharge[3,:].T
        self.T = len(self.tm) - 1
        self.t = np.arange(len(self.tm)) * self.Ts    #生成等距Ts的时间戳
        self.warmup_idx = np.where(abs(self.RSOC - 0.99) < 1e-6)[0].item()
        self.warmup_t = self.t[self.warmup_idx].item()

        x = OCV_SOC[1,:]
        y = OCV_SOC[0,:]
        self.p = np.polyfit(x, y, self.poly_order)
        self.dp = np.polyder(self.p)
        
    def _prepare_mpdel(self):
        Ts,R1,R2,C1,C2 = self.Ts, self.R1, self.R2, self.C1, self.C2
        self.a1 = 1 + (-Ts / (R1 * C1))  # np.exp(-Ts / (R1 * C1))的欧拉近似
        self.a2 = 1 + (-Ts / (R2 * C2))  # np.exp(-Ts / (R2 * C2))的欧拉近似
        self.b1 = R1 * (1 - self.a1)
        self.b2 = R2 * (1 - self.a2)
    
    def __init_param_estimator(self):
        def g_func(x, u, theta):
            R0 = theta[0]
            Soc = x[2]
            V_RC = x[0] + x[1]
            U_ocv = np.polyval(self.p, Soc)
            return U_ocv - V_RC - R0 * u

        # def dg_dtheta(x, u, theta):
        #     return np.array([-u])
        
        theta0 = np.array([self.R0], [self.Qn])
        Q_theta = np.diag([1e-6, 1e-4])
        R = 10
        self.param_estimator = EKFParamEstimator(theta0, Q_theta, R, g_func)

    def run(self):
        a1, a2, b1, b2 = self.a1, self.a2, self.b1, self.b2
        A = np.array([[a1, 0, 0],
                    [0, a2, 0],
                    [0, 0, 1]])
        B = np.array([[b1], [b2], [-self.Ts / self.Qn]])
        C_mat = np.array([[-1, -1, 0]])
        D = -self.R0

        # Initialize
        Q = 1e-6 * np.eye(3)
        R = 0.1
        P = np.diag([1e-2, 1e-2, 0.1])

        X = np.zeros((3, self.T + 1))
        X[:, 0] = [0, 0, 0.8]
        Uoc = np.zeros(self.T + 1)
        Vekf = np.zeros(self.T + 1)

        Uoc[0] = np.polyval(self.p, X[2, 0])
        R0_k = self.param_estimator.theta[0]
        Vekf[0] = Uoc[0] + C_mat @ X[:, 0] + D * self.Cur[0]

        # === 5. EKF 循环 ===
        for k in range(self.T):
            X[:, k + 1] = A @ X[:, k] + (B * self.Cur[k]).flatten()
            Uoc[k + 1] = np.polyval(self.p, X[2, k + 1])
            dOCV = np.polyval(self.dp, X[2, k + 1])
            H = np.array([[-1, -1, dOCV]])
            R0_k = self.param_estimator.theta[0]
            Vekf[k + 1] = Uoc[k + 1] + C_mat @ X[:, k + 1] + D * self.Cur[k + 1]
            P = A @ P @ A.T + Q
            S = H @ P @ H.T + R
            K = (P @ H.T) @ np.linalg.inv(S)
            X[:, k + 1] += (K.flatten() * (self.Vot[k + 1] - Vekf[k + 1]))
            P = (np.eye(3) - K @ H) @ P

            self.param_estimator.step(X[:,k+1], self.Cur[k+1], self.Vot[k+1])

        # === 6. 误差分析 ===
        V_error = self.Vot - Vekf
        SOC_error = abs(self.RSOC - X[2, :])
        SOC_mae = np.mean(np.abs(SOC_error[self.warmup_idx:]))
        SOC_rmse = np.sqrt(np.mean(SOC_error[self.warmup_idx:]**2))
        SOC_max = np.max(np.abs(SOC_error[self.warmup_idx:]))
        V_rmse = np.sqrt(np.mean(V_error[self.warmup_idx:]**2))
        R0_series = np.array(self.param_estimator.get_estimates())[:,0]

        # === 7. 输出结果 ===
        print(f"SOC MAE:  {SOC_mae:.5f}")
        print(f"SOC RMSE: {SOC_rmse:.5f}")
        print(f"SOC Max:  {SOC_max:.5f}")
        print(f"Voltage RMSE: {V_rmse:.5f}")
        print(f"Estimated R0 mean: {np.mean(R0_series[self.warmup_idx:]):.5f} Ω")

        # === 8. 可视化 ===
        plt.figure(figsize=(10, 4))
        plt.plot(self.t, self.Vot, label="True Voltage")
        plt.plot(self.t, Vekf, label="Estimated Voltage-EKF", linestyle='--')
        plt.axvline(self.warmup_t, color='r', linestyle='--', label="99% SOC")
        plt.title("Voltage")
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (V)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plt.figure(figsize=(10, 4))
        plt.plot(self.t, self.RSOC, label="True SOC")
        plt.plot(self.t, X[2, :], label="Estimated SOC-EKF", linestyle='--')
        plt.axvline(self.warmup_t, color='r', linestyle='--', label="99% SOC")
        plt.title("SOC")
        plt.xlabel("Time (s)")
        plt.ylabel("SOC")
        plt.grid(True)
        plt.legend()
        plt.legend()
        plt.tight_layout()

        plt.figure(figsize=(10, 4))
        plt.plot(self.t, SOC_error, label="SOC Error")
        plt.axvline(self.warmup_t, color='r', linestyle='--', label="99% SOC")
        plt.title("SOC Absolte Error")
        plt.xlabel("Time (s)")
        plt.ylabel("Absolute Error")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()       