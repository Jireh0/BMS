from EKF import EKFSOC
from EKF_R0 import EKFSOC_R0

# ekf = EKFSOC(data_path='matlab/EKF')
# ekf.run()

ekf_r0 = EKFSOC_R0(data_path='matlab/EKF')
ekf_r0.run()