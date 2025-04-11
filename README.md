
This repository is about battery management system (BMS) design. It includes SOC (State-of-Charge), SOH (State-of-Health), and Internal Resistance Estimation.

2025-03-28
We finish the coupled SoC-R calculation. This is shown in the **EKF_R0.py**, this method can calculate SoC and R0 together. Alternatively, we can use the Kalman filter to estimate the SoC and Parameters (R0) separately. These are **EKF.py** and **EKF_param_estimator.py**. However, **EKF_param_estimator.py** has not been done yet.

2025-04-09
We reproduce a paper, using CNN and LSTM models to predict the **SOH** and **RUL**. The dataset in the ASA dataset, the **CNN.ipynb** is the initial version, which is also the version in the paper. However, we made some adjustments to the model (use **AWP**) tricks to improve the performance. The final version is shown as **CNN.py**. We can not reproduce the LSTM model when using the parameters specified in the paper, so we made some adjustments  (**AWP**, **RDrop**, **Batch Normalization** **gradient clip**, **weight decay**) to the model. And the effects are very good under different conditions.
