import numpy as np
import matplotlib.pyplot as plt
from EKF import EKFSOC

class EKFSOC_R0(EKFSOC):
    def run(self):
        A = np.eye(4)
        A[0,0] = self.a1
        A[1,1] = self.a2
        B = np.array([[self.b1],[self.b2],[-self.Ts/self.Qn],[0]])

        Q = np.diag([1e-6, 1e-6, 1e-7, 1e-8])
        R = 0.1
        P = np.diag([1e-2, 1e-2, 0.1, 1e-4])

        X = np.zeros((4, self.T + 1))
        X[:,0] = [0, 0, 0.8, self.R0]
        Uoc = np.zeros(self.T + 1)
        Vekf = np.zeros(self.T + 1)

        Uoc[0] = np.polyval(self.p, X[2,0])
        Vekf[0] = Uoc[0] - X[0,0] - X[1,0] - X[3,0] * self.Cur[0]

        for k in range(self.T):
            X[:, k + 1] = A @ X[:, k] + (B * self.Cur[k]).flatten()
            SoC_k = X[2, k+1]
            R0_k = X[3, k+1]
            Uoc[k + 1] = np.polyval(self.p, SoC_k)
            dOCV = np.polyval(self.dp, SoC_k)
            Vekf[k + 1] = Uoc[k + 1] - X[0, k + 1] - X[1, k + 1] - R0_k * self.Cur[k + 1]

            H = np.array([[-1, -1, dOCV, -self.Cur[k + 1]]])
            P = A @ P @ A.T + Q
            S = H @ P @ H.T + R
            K = (P @ H.T) @ np.linalg.inv(S)
            X[:, k + 1] += K.flatten() * (self.Vot[k + 1] - Vekf[k + 1])
            P = (np.eye(4) - K @ H) @ P

        V_error = self.Vot - Vekf
        SOC_error = abs(self.RSOC - X[2, :])
        SOC_mae = np.mean(SOC_error[self.warmup_idx:])
        SOC_rmse = np.sqrt(np.mean(SOC_error[self.warmup_idx:]**2))
        SOC_max = np.max(SOC_error[self.warmup_idx:])
        V_rmse = np.sqrt(np.mean(V_error[self.warmup_idx:]**2))
        R0_mean = np.mean(X[3, self.warmup_idx:])

        print(f"SOC MAE:  {SOC_mae:.5f}")
        print(f"SOC RMSE: {SOC_rmse:.5f}")
        print(f"SOC Max:  {SOC_max:.5f}")
        print(f"Voltage RMSE: {V_rmse:.5f}")
        print(f"Estimated R₀ mean: {R0_mean:.5f} Ω")

        plt.figure(figsize=(10, 4))
        plt.plot(self.t, X[3, :], label="Estimated R₀ (EKF)")
        plt.axhline(self.R0, color='g', linestyle='--', label="Initial R₀")
        plt.xlabel("Time (s)")
        plt.ylabel("Internal Resistance (Ω)")
        plt.title("Online R₀ Estimation via EKF")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
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
        plt.show() 