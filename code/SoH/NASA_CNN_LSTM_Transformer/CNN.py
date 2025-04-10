import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from joblib import dump, load

torch.manual_seed(42)
np.random.seed(42)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_dir = 'saved_models'
os.makedirs(model_dir, exist_ok = True)

# 加载数据
df = pd.read_csv('B0018_discharge.csv')
test = df[['cycle', 'capacity']].copy().drop_duplicates()


# 数据准备
split_threshold = 90
train_capacity = test[test['cycle'] <= split_threshold]['capacity'].values
test_capacity = test[test['cycle'] > split_threshold]['capacity'].values

# 数据归一化
scaler = MinMaxScaler()
# 1D数组 -> 2D列向量(samples, feature) -> 1D数组
train_capacity_scaled = scaler.fit_transform(train_capacity.reshape(-1, 1)).flatten()
test_capacity_scaled = scaler.transform(test_capacity.reshape(-1, 1)).flatten()

def create_sequences(data, seq_length=5):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(x), np.array(y)

X_train, y_train = create_sequences(train_capacity_scaled)
X_test, y_test = create_sequences(test_capacity_scaled)

# 重塑为2D CNN输入形状 (batch_size, in_channels, height, width)
# 这里将序列长度作为"高度"，1作为"宽度"
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], 1)

# 转换为PyTorch张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1,1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1,1)

# 创建DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# ======================== CNN ============================================================================
# 2D CNN模型定义
class BatteryCapacityPredictor2D(nn.Module):
    def __init__(self):
        super(BatteryCapacityPredictor2D, self).__init__()
        '''
        Input data size: [Batch_size, In_channels, Height, Width]
        Output data size: [Batch_size, Out_channels, Height, Width]
        H_out = [H_in + 2 * padding_H - dilation_H * (Kernel_size_H - 1) -1] / stride_H + 1
        W_out = ...
        '''
        # 第一卷积层 (对应表格中的conv_1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3,1), stride=(1,1), padding=0),
            nn.BatchNorm2d(10),
            nn.ReLU()
        )
        
        # 第二卷积层 (对应表格中的conv_2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3,1), stride=(1,1), padding=0),
            nn.BatchNorm2d(20),
            nn.ReLU()
        )
        
        # 全连接层 (对应表格中的fc)
        self.fc = nn.Linear(20, 1)  # 输入是20，因为最后一个卷积层输出20通道
        
    def forward(self, x):
        # 输入形状: (batch_size, 1, 5, 1)
        x = self.conv1(x)  # 输出: (batch_size, 10, 3, 1)
        x = self.conv2(x)  # 输出: (batch_size, 20, 1, 1)
        
        # 展平
        x = x.view(x.size(0), -1)  # 输出: (batch_size, 20)
        x = self.fc(x)  # 输出: (batch_size, 1)
        return x
# ======================= AWP =========================================
class AWP:
    def __init__(self, model, optimizer, adv_param="weight", epsilon=1e-2, alpha=0.3):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.epsilon = epsilon
        self.alpha = alpha
        self.backup = {}
        self.backup_eps = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_norm = torch.norm(param.grad)
                    if grad_norm != 0:
                        r_adv = self.alpha * param.grad / grad_norm
                        param.data.add_(r_adv)
                        param.data = torch.min(torch.max(param.data, self.backup[name] - self.epsilon),
                                               self.backup[name] + self.epsilon)

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
# =====================================================================================================
use_awp = True
# use_rdrop = True
# alpha_rdrop = 5.0
awp_start_epoch = 30

# 初始化模型
model = BatteryCapacityPredictor2D().to(device)
print(model)

# 损失函数和优化器
# criterion = nn.HuberLoss(delta=0.1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

awp = AWP(model, optimizer) if use_awp else None

# 训练循环
best_val_loss = float('inf')
best_epoch = 0
train_losses = []
val_losses = []

for epoch in range(1000):
    model.train()
    epoch_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    if use_awp:
        awp.attack()
        out_adv = model(batch_x)
        loss_adv = criterion(out_adv, batch_y)
        awp.restore()

    avg_epoch_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_epoch_loss)
    
    # 验证
    model.eval()
    with torch.no_grad():
        val_pred = model(X_test_tensor.to(device))
        val_loss = criterion(val_pred, y_test_tensor.to(device))
        val_losses.append(val_loss.item())
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        '''torch.sae(model.state_dict, PATH)保存模型 state_dict保存了模型的可学习参数(权重，偏差)
        只有有可学习参数的层(卷积层、线性层)才有state_dict,优化器对象torch.optim也有state_dict'''
        torch.save(model.state_dict(), os.path.join(model_dir, 'best_model.pth'))
        dump(scaler, os.path.join(model_dir, 'scaler.joblib'))
    
    scheduler.step()

    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/1000], Train Loss: {avg_epoch_loss:.6f}, Val Loss: {val_loss.item():.6f}, Best Val Loss: {best_val_loss:.6f} (Epoch {best_epoch+1})')

# 绘制训练曲线
plt.figure(figsize=(10,5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.axvline(x=best_epoch, color='r', linestyle='--', label='Best Model')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training History')
plt.show()

# 加载最佳模型
best_model = BatteryCapacityPredictor2D().to(device)
best_model.load_state_dict(torch.load(os.path.join(model_dir, 'best_model.pth')))
best_model.eval()

# 加载scaler
loaded_scaler = load(os.path.join(model_dir, 'scaler.joblib'))

# 使用最佳模型进行预测
with torch.no_grad():
    y_pred_scaled = best_model(X_test_tensor.to(device)).cpu()
    y_train_pred_scaled = best_model(X_train_tensor.to(device)).cpu()
    
    # 反归一化
    y_pred = loaded_scaler.inverse_transform(y_pred_scaled.numpy())
    y_train_pred = loaded_scaler.inverse_transform(y_train_pred_scaled.numpy())
    
    # 计算指标
    test_rmse = np.sqrt(mean_squared_error(test_capacity[5:], y_pred))
    test_mae = mean_absolute_error(test_capacity[5:], y_pred)
    test_mape = mean_absolute_percentage_error(test_capacity[5:], y_pred)
    train_rmse = np.sqrt(mean_squared_error(train_capacity[5:], y_train_pred))
    
    print(f'\nBest Model Performance:')
    print(f'Best Epoch: {best_epoch+1}')
    print(f'Test RMSE: {test_rmse:.4f}, Test MAE: {test_mae:.4f}, Test MAPE: {test_mape: .2%}')
    print(f'Train RMSE: {train_rmse:.4f}')

# 可视化最佳模型的结果
plt.figure(figsize=(10,8))
plt.plot(test['cycle'], test['capacity'], label = 'True')
plt.plot(test['cycle'][5:split_threshold], y_train_pred, label='Train', color = 'r')
plt.plot(test['cycle'][split_threshold+5:], y_pred, label='Predict', color = 'orange')
plt.plot([0,168],[1.4,1.4],color = 'purple', linestyle = '--')
plt.xlim(0,180)
# plt.ylim(1.0,2.0)
plt.xlabel('Cycle')
plt.ylabel('Capacity / Ah')
plt.legend()
plt.show()