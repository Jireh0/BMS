import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from joblib import dump, load

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_dir = 'saved_model'
os.makedirs(model_dir, exist_ok= True)

df = pd.read_csv('B0007_discharge.csv')
test = df[['cycle','capacity']].copy().drop_duplicates()

split_threshold = 90
train_capacity = test[test['cycle'] <= split_threshold]['capacity'].values
test_capacity = test[test['cycle'] > split_threshold]['capacity'].values

data_scaler = MinMaxScaler()
train_capacity_scaled = data_scaler.fit_transform(train_capacity.reshape(-1,1)).flatten()
test_capacity_scaled = data_scaler.transform(test_capacity.reshape(-1,1)).flatten()

def create_sequences(data, seq_length = 5):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(x), np.array(y)

X_train, y_train = create_sequences(train_capacity_scaled)
X_test, y_test = create_sequences(test_capacity_scaled)

# LSTM INPUT SHAPE (batch_size, seq_length, input_size)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

X_train_tensor = torch.tensor(X_train, dtype = torch.float32)
X_test_tensor = torch.tensor(X_test, dtype = torch.float32)
y_train_tensor = torch.tensor(y_train, dtype =torch.float32).view(-1,1)
y_test_tensor = torch.tensor(y_test, dtype = torch.float32).view(-1,1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
print(len(train_dataset))
batch_size = 16 if len(train_dataset) >= 100 else 8
train_loader = DataLoader(train_dataset, batch_size, shuffle= True)

#  ============ LSTM ==================================================================

class BatteryCapacityPredictorLSTM(nn.Module):
    def __init__(self):
        super(BatteryCapacityPredictorLSTM, self).__init__()
        '''
        input_size: 输入x的特征数
        hidden_size: 状态h的特征数

        '''
        self.lstm1 = nn.LSTM(
            input_size = 1,
            hidden_size = 10,
            num_layers = 1,
            batch_first = True
        )
        # self.bn1 = nn.BatchNorm1d(10)

        self.lstm2 = nn.LSTM(
            input_size = 10,
            hidden_size = 20,
            num_layers = 1,
            batch_first= True
        )
        self.bn2 = nn.BatchNorm1d(20)
        
        self.relu = nn.ReLU()

        self.fc = nn.Linear(20, 1)

    def forward(self, x):

        out, _ = self.lstm1(x)  # hn_1 shape (1, batch_size, 10)
        # out = self.bn1(out.transpose(1,2)).transpose(1,2)
        # lstm2 hidden_size为 20, 要么把h_n1,c_n1堆起来，要么重新初始化 
        # h_n1 = h_n1.repeat(1,1,2)
        # c_n1 = c_n1.repeat(1,1,2)

        out, _ = self.lstm2(out)

        out = out[:, -1, :]

        out = self.bn2(out)
        out = self.relu(out)

        out = self.fc(out)
        return out

# ======= 开关 ====================================================
use_fgm = False # 副作用,且训练速度变慢（太暴力了）
use_rdrop = True # 有点用,也有可能没用
use_awp = True  # 这个非常有用
use_amp = False  # 没啥用,速度还变慢了（需要在GPU上跑快）
use_amp = torch.cuda.is_available and use_amp
alpha_rdrop = 5.0
awp_start_epoch = 30
# ======================== FGM =====================================
class FGM:
    def __init__(self, model, epsilon = 1.0):
        self.model = model
        self.epsilon = epsilon
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and 'weight' in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
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

model = BatteryCapacityPredictorLSTM().to(device)
print(model)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

total_params = count_parameters(model)

# criterion = nn.HuberLoss(delta=0.1)
criterion  = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay= 1e-2)

# 修改学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size= 15, gamma= 0.5)

amp_scaler = torch.amp.GradScaler(enabled=use_amp)
fgm = FGM(model) if use_fgm else None
awp = AWP(model, optimizer) if use_awp else None

best_loss = float('inf')
best_epoch = 0
train_losses = []
val_losses = []

for epoch in range(1000):
    model.train()
    epoch_loss = 0

    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type = 'cuda', enabled=use_amp):
            if use_rdrop:
                out1 = model(batch_x)
                out2 = model(batch_x)
                loss1 = criterion(out1, batch_y)
                loss2 = criterion(out2, batch_y)
                kl_loss = nn.KLDivLoss(reduction='batchmean')(torch.log_softmax(out1, dim=-1),
                                                              torch.softmax(out2, dim = -1)) + \
                          nn.KLDivLoss(reduction='batchmean')(torch.log_softmax(out2, dim=-1),
                                                              torch.softmax(out1, dim=-1))
                loss = (loss1 + loss2) / 2 + alpha_rdrop * kl_loss
            else:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
        
        amp_scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        if use_fgm:
            fgm.attack()
            with torch.amp.autocast(device_type= 'cuda', enabled=use_amp):
                out_adv = model(batch_x)
                loss_adv = criterion(out_adv, batch_y)
            amp_scaler.scale(loss_adv).backward()
            fgm.restore()
        
        if use_awp and epoch >= awp_start_epoch:
            awp.attack()
            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                out_adv = model(batch_x)
                loss_adv = criterion(out_adv, batch_y)
            amp_scaler.scale(loss_adv).backward()

        amp_scaler.step(optimizer)
        amp_scaler.update()
        epoch_loss += loss.item()
    
    avg_epoch_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_epoch_loss)

    model.eval()
    with torch.no_grad():
        val_pred = model(X_test_tensor.to(device))
        val_loss = criterion(val_pred, y_test_tensor.to(device))
        val_losses.append(val_loss.item())

    if avg_epoch_loss * 0.4 + val_loss * 0.6 < best_loss:
        best_epoch = epoch
        best_loss = avg_epoch_loss * 0.4 + val_loss * 0.6
        torch.save(model.state_dict(), os.path.join(model_dir, 'best_lstm_model.pth'))
        dump(data_scaler, os.path.join(model_dir, 'lstm_scaler.joblib'))

    scheduler.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/1000], Train Loss: {avg_epoch_loss:.6f}, Val Loss: {val_loss.item():.6f}, Best Loss: {best_loss:.6f} (Epoch {best_epoch+1})')

plt.figure(figsize = (10,5))
plt.plot(train_losses, label = 'Training Loss')
plt.plot(val_losses, label = 'Validation Loss')
plt.axvline(x = best_epoch, color = 'r', linestyle = '--', label = 'Best Model')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('LSTM Training History')

best_model = BatteryCapacityPredictorLSTM().to(device)
best_model.load_state_dict(torch.load(os.path.join(model_dir, 'best_lstm_model.pth')))
best_model.eval()

loaded_scaler = load(os.path.join(model_dir,'lstm_scaler.joblib'))

with torch.no_grad():
    y_pred_scaled = best_model(X_test_tensor.to(device)).cpu()  #后续转换为np 或 pd需要移到cpu上
    y_train_pred_scaled = best_model(X_train_tensor.to(device)).cpu()

    y_pred = loaded_scaler.inverse_transform(y_pred_scaled.numpy())
    y_train_pred = loaded_scaler.inverse_transform(y_train_pred_scaled.numpy())

    test_rmse = np.sqrt(mean_squared_error(test_capacity[5:], y_pred))
    test_mae = mean_absolute_error(test_capacity[5:], y_pred)
    test_mape = mean_absolute_percentage_error(test_capacity[5:], y_pred)
    train_rmse = np.sqrt(mean_squared_error(train_capacity[5:], y_train_pred))

print(f'\nBest LSTM Model Performance:')
print(f'Best Epoch: {best_epoch+1}')
print(f'Test RMSE: {test_rmse:.4f}, Test MAE: {test_mae: .4f}, Test MAPE: {test_mape: .2%}')
print(f'Train RMSE: {train_rmse: .4f}')

plt.figure(figsize=(10,8))
plt.plot(test['cycle'], test['capacity'], label='True Capacity')
plt.plot(test['cycle'][5:split_threshold], y_train_pred, label='Train Prediction', color='orange')
plt.plot(test['cycle'][split_threshold+5:], y_pred, label='Test Prediction', color='red')
plt.plot([0,168],[1.4,1.4], color='purple', linestyle='--')
plt.xlim(0,180)
# plt.ylim(1.2,2.0)
plt.xlabel('Cycle')
plt.ylabel('Capacity / Ah')
plt.legend()
plt.title('LSTM Capacity Prediction (Table 3.3 Architecture)')
plt.show()

