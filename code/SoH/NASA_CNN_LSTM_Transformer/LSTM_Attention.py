import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from joblib import dump, load
import math

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_dir = 'saved_model'
os.makedirs(model_dir, exist_ok= True)

def load_data(file_path):
    df = pd.read_csv(file_path)
    data = df[['cycle','capacity']].copy().drop_duplicates()
    return data

def split_data(data, threshold):
    train_data = data[data['cycle'] <= threshold]['capacity'].values
    test_data = data[data['cycle'] > threshold]['capacity'].values
    return train_data, test_data

def scale_data(train_data, test_data):
    scaler = MinMaxScaler()
    train_data_scaled = scaler.fit_transform(train_data.reshape(-1, 1)).flatten()
    test_data_scaled = scaler.transform(test_data.reshape(-1, 1)).flatten()
    return train_data_scaled, test_data_scaled, scaler

def create_sequences(data, seq_length = 10):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(x), np.array(y)

def create_dataloader(x, y, batch_size=32):  # x, y are tensors
    x = x.unsqueeze(-1)  # Add a channel dimension for LSTM input
    y = y.unsqueeze(-1)  # Add a channel dimension for LSTM output
    dataset = TensorDataset(x,y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
#  ============ Model ==================================================================
class StandardAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(StandardAttention, self).__init__()
        self.W_q = nn.Linear(input_dim, hidden_dim)
        self.W_k = nn.Linear(input_dim, hidden_dim)
        self.W_v = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(K.size(-1))
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, V)
        return context
    
class BatteryLSTMAttention(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2):
        super(BatteryLSTMAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=0.1 if num_layers > 1 else 0,
            # bidirectional=True
        )

        self.attention = StandardAttention(hidden_dim, hidden_dim)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1)
        )
        self.bn = nn.BatchNorm1d(hidden_dim)


    def forward(self, x):

        lstm_out, _ = self.lstm(x)
        attn_out = self.attention(lstm_out)
        context = attn_out[:, -1, :]
        out = self.bn(context)
        out = self.fc(out)
        return out
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

def train_model(model, train_loader, val_data, criterion, optimizer, scheduler, epochs=1000, \
                use_fgm=False, use_rdrop=False, use_awp=True, use_amp=False, alpha_rdrop=5.0, awp_start_epoch=30):
    
    fgm = FGM(model) if use_fgm else None
    awp = AWP(model, optimizer) if use_awp else None
    
    best_loss = float('inf')
    best_epoch = 0
    train_losses = []
    val_losses = []
        
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()  # Zero the gradients
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

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if use_fgm:
                fgm.attack()
                out_adv = model(batch_x)
                loss_adv = criterion(out_adv, batch_y)
                loss_adv.backward()
                fgm.restore()

            if use_awp and epoch >= awp_start_epoch:
                awp.attack()
                out_adv = model(batch_x)
                loss_adv = criterion(out_adv, batch_y)
                loss_adv.backward()
                # awp.restore()
            
            optimizer.step() 
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_epoch_loss)

        model.eval()
        with torch.no_grad():  # No gradient calculation, save the memory
            val_x, val_y = val_data
            val_pred = model(val_x.to(device))
            val_loss = criterion(val_pred, val_y.to(device))
            val_losses.append(val_loss.item())
        
        if 0.6*val_loss + 0.4*avg_epoch_loss < best_loss:
            best_loss = 0.6*val_loss + 0.4*avg_epoch_loss
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(model_dir, 'best_lstm_attn_model.pth'))
            dump(data_scaler, os.path.join(model_dir, 'lstm_attn_scaler.joblib'))
        
        scheduler.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_epoch_loss:.6f}, Val Loss: {val_loss.item():.6f}, Best Loss: {best_loss:.6f} (best eoch {best_epoch})')

    return train_losses, val_losses
# ====================================================== MAIN ============================================================
if __name__ == '__main__':
    file_path = 'B0007_discharge.csv'
    data = load_data(file_path)

    split_threshold = 90
    train_capacity, test_capacity = split_data(data, split_threshold)
    train_capacity_scaled, test_capacity_scaled, data_scaler = scale_data(train_capacity, test_capacity)

    seq_length = 10
    X_train, y_train = create_sequences(train_capacity_scaled, seq_length)
    X_test, y_test = create_sequences(test_capacity_scaled, seq_length)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    train_loader = create_dataloader(X_train_tensor, y_train_tensor)
    val_data = (X_test_tensor.unsqueeze(-1), y_test_tensor.unsqueeze(-1))
    # ============ LSTM Attention Model =========================================   
    use_fgm = False # 副作用,且训练速度变慢（太暴力了）
    use_rdrop = False # 有点用,也有可能没用
    use_awp = True  # 这个非常有用
    use_amp = False  # 没啥用,速度还变慢了（需要在GPU上跑快）
    use_amp = torch.cuda.is_available and use_amp
    alpha_rdrop = 5.0
    awp_start_epoch = 30

    model = BatteryLSTMAttention(input_dim=1, hidden_dim=64, num_layers=2).to(device)
    print(model)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    train_losses, val_losses = train_model(model, train_loader, val_data, criterion, optimizer, scheduler, epochs=1000, \
                                           use_fgm=use_fgm, use_rdrop=use_rdrop, use_awp=use_awp, use_amp=use_amp,\
                                            alpha_rdrop=alpha_rdrop, awp_start_epoch=awp_start_epoch)
    # plotting training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('LSTM Attention Training History')
    plt.show()

    best_model = BatteryLSTMAttention().to(device)
    best_model.load_state_dict(torch.load(os.path.join(model_dir, 'best_lstm_attn_model.pth')))
    best_model.eval()
    loaded_scaler = load(os.path.join(model_dir, 'lstm_attn_scaler.joblib'))

    with torch.no_grad():
        train_pred_scaled = best_model(X_train_tensor.unsqueeze(-1).to(device)).cpu().numpy()
        test_pred_scaled = best_model(X_test_tensor.unsqueeze(-1).to(device)).cpu().numpy()
        
        train_pred = loaded_scaler.inverse_transform(train_pred_scaled)
        test_pred = loaded_scaler.inverse_transform(test_pred_scaled)

        test_rmse = np.sqrt(mean_squared_error(test_capacity[seq_length:], test_pred))
        test_mae = mean_absolute_error(test_capacity[seq_length:], test_pred)
        test_mape = mean_absolute_percentage_error(test_capacity[seq_length:], test_pred)
        train_rmse = np.sqrt(mean_squared_error(train_capacity[seq_length:], train_pred))


print(f'\nBest LSTM Attention Model Performance:')
print(f'Test RMSE: {test_rmse:.4f}, Test MAE: {test_mae:.4f}, Test MAPE: {test_mape:.2%}')
print(f'Train RMSE: {train_rmse:.4f}')

plt.figure(figsize=(10, 8))
plt.plot(data['cycle'], data['capacity'], label='True Capacity')
plt.plot(data['cycle'][seq_length:split_threshold], train_pred, label='Train Prediction', color='orange')
plt.plot(data['cycle'][split_threshold+seq_length:], test_pred, label='Test Prediction', color='red')
plt.plot([0, 168], [1.4, 1.4], color='purple', linestyle='--')
plt.xlim(0, 180)
# plt.ylim(1.2, 2.0)
plt.xlabel('Cycle')
plt.ylabel('Capacity / Ah')
plt.legend()
plt.show()