import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os


def ekf_soc_estimation(data_path = 'matlab/EKF', Ts = 0.1, Qn = 30.23 * 3600, poly_order = 8):
    
    try:
        # 数据加载
        R0 = sio.loadmat(f'{data_path}/R0.mat')['R0'].item()
        R1 = sio.loadmat(f'{data_path}/R1.mat')['R1'].item() + 0.0007
        R2 = sio.loadmat(f'{data_path}/R2.mat')['R2'].item() + 0.0005
        C1 = sio.loadmat(f'{data_path}/C1.mat')['C1'].item()
        C2 = sio.loadmat(f'{data_path}/C2.mat')['C2'].item()
        discharge = sio.loadmat(f'{data_path}/discharge.mat')['discharge'] 
        OCV_SOC = sio.loadmat(f'{data_path}/OCV_SOC.mat')['OCV_SOC']
    except FileNotFoundError:
        print(os.getcwd())

    # 数据提取和拟合
    tm = discharge[0,:].T
    Cur = -discharge[1,:].T
    Vot = discharge[2,:].T
    RSOC = discharge[3,:].T
    T = len(tm) - 1
    t = np.arange(len(tm)) * Ts    #生成等距Ts的时间戳
    warmup_idx = np.where(abs(RSOC - 0.99) < 1e-6)[0].item()
    warmup_t = t[warmup_idx].item()

    x = OCV_SOC[1,:]
    y = OCV_SOC[0,:]
    p = np.polyfit(x, y, poly_order)
    dp = np.polyder(p)

    a1 = 1 + (-Ts / (R1 * C1))  # np.exp(-Ts / (R1 * C1))的欧拉近似
    a2 = 1 + (-Ts / (R2 * C2))  # np.exp(-Ts / (R2 * C2))的欧拉近似
    b1 = R1 * (1 - a1)
    b2 = R2 * (1 - a2)

    A = np.array([[a1, 0, 0],
                  [0, a2, 0],
                  [0, 0, 1]])
    B = np.array([[b1], [b2], [-Ts / Qn]])
    C_mat = np.array([[-1, -1, 0]])
    D = -R0

    # Initialize
    Q = 1e-6 * np.eye(3)
    R = 0.1
    P = np.diag([1e-2, 1e-2, 0.1])

    X = np.zeros((3, T + 1))
    X[:, 0] = [0, 0, 0.8]
    Uoc = np.zeros(T + 1)
    Vekf = np.zeros(T + 1)

    Uoc[0] = np.polyval(p, X[2, 0])
    Vekf[0] = Uoc[0] + C_mat @ X[:, 0] + D * Cur[0]

    # === 5. EKF 循环 ===
    for k in range(T):
        X[:, k + 1] = A @ X[:, k] + (B * Cur[k]).flatten()
        Uoc[k + 1] = np.polyval(p, X[2, k + 1])
        dOCV = np.polyval(dp, X[2, k + 1])
        H = np.array([[-1, -1, dOCV]])
        Vekf[k + 1] = Uoc[k + 1] + C_mat @ X[:, k + 1] + D * Cur[k + 1]
        P = A @ P @ A.T + Q
        S = H @ P @ H.T + R
        K = (P @ H.T) @ np.linalg.inv(S)
        X[:, k + 1] += (K.flatten() * (Vot[k + 1] - Vekf[k + 1]))
        P = (np.eye(3) - K @ H) @ P

    # === 6. 误差分析 ===
    V_error = Vot - Vekf
    SOC_error = abs(RSOC - X[2, :])
    SOC_mae = np.mean(np.abs(SOC_error[warmup_idx:]))
    SOC_rmse = np.sqrt(np.mean(SOC_error[warmup_idx:]**2))
    SOC_max = np.max(np.abs(SOC_error[warmup_idx:]))
    V_rmse = np.sqrt(np.mean(V_error[warmup_idx:]**2))

    # === 7. 输出结果 ===
    print(f"SOC MAE:  {SOC_mae:.5f}")
    print(f"SOC RMSE: {SOC_rmse:.5f}")
    print(f"SOC Max:  {SOC_max:.5f}")
    print(f"Voltage RMSE: {V_rmse:.5f}")

    # === 8. 可视化 ===
    plt.figure(figsize=(10, 4))
    plt.plot(t, Vot, label="True Voltage")
    plt.plot(t, Vekf, label="Estimated Voltage-EKF", linestyle='--')
    plt.axvline(warmup_t, color='r', linestyle='--', label="99% SOC")
    plt.title("Voltage")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(10, 4))
    plt.plot(t, RSOC, label="True SOC")
    plt.plot(t, X[2, :], label="Estimated SOC-EKF", linestyle='--')
    plt.axvline(warmup_t, color='r', linestyle='--', label="99% SOC")
    plt.title("SOC")
    plt.xlabel("Time (s)")
    plt.ylabel("SOC")
    plt.grid(True)
    plt.legend()
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(10, 4))
    plt.plot(t, SOC_error, label="SOC Error")
    plt.axvline(warmup_t, color='r', linestyle='--', label="99% SOC")
    plt.title("SOC Absolte Error")
    plt.xlabel("Time (s)")
    plt.ylabel("Absolute Error")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()







