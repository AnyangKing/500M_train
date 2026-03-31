import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from tqdm import tqdm
import random
import time
import math
import optuna
from sklearn.preprocessing import StandardScaler
import joblib
import os

# ==============================================================================
# 1. 설정
# ==============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 데이터셋 및 스케일러 경로
DATASET_PATH = 'dataset_td_0.0-15.0_doa_0.0-0.5_500M.npz' 
SCALER_X_PATH = 'scaler_x_td_0.0-15.0_doa_0.0-0.5_500M.pkl'
SCALER_Y_PATH = 'scaler_y_td_0.0-15.0_doa_0.0-0.5_500M.pkl'

INPUT_DIM, OUTPUT_DIM = 25, 3
MAX_LEN = 20  # 20step 데이터셋에 맞춤
EPOCHS = 15   # 각 시도(Trial)당 학습 에포크 (Optuna는 짧게 유지)
N_TRIALS = 150 # 탐색 횟수를 150회로 상향 조정

# [전략] 탐색 속도를 위해 전체 데이터 중 일부만 사용 (예: 10만 개)
OPTUNA_SUBSET_SIZE = 500000 

# ==============================================================================
# 2. 모델 및 데이터셋 정의
# ==============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=20):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])

class TransformerEncoderOnlyModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, nhead, nlayers, dropout=0.5):
        super(TransformerEncoderOnlyModel, self).__init__()
        self.d_model = d_model
        self.encoder_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=MAX_LEN)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=nlayers)
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, src):
        src = self.encoder_embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        return self.fc_out(self.transformer_encoder(src))

class TrajectoryDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x = torch.FloatTensor(x_data)
        self.y = torch.FloatTensor(y_data)
    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

# ==============================================================================
# 3. Optuna Objective 함수
# ==============================================================================
def objective(trial, train_dataset, val_dataset):
    # 하이퍼파라미터 탐색 공간
    d_model = trial.suggest_categorical("d_model", [128, 256, 512])
    n_layers = trial.suggest_int("n_layers", 8, 15)
    n_heads = trial.suggest_categorical("n_heads", [4, 8, 16])
    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    dropout = trial.suggest_float("dropout", 0.05, 0.2)

    # d_model은 n_heads의 배수여야 함
    if d_model % n_heads != 0:
        raise optuna.exceptions.TrialPruned()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = TransformerEncoderOnlyModel(INPUT_DIM, OUTPUT_DIM, d_model, n_heads, n_layers, dropout).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_valid_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            
            # 그래디언트 클리핑으로 발산 방지
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
        
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
                valid_loss += criterion(model(x_batch), y_batch).item()
        valid_loss /= len(val_loader)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
        
        # Pruning (성능이 안 좋으면 중간에 가지치기)
        trial.report(valid_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_valid_loss

# ==============================================================================
# 4. 실행 블록
# ==============================================================================
if __name__ == '__main__':
    print("데이터 로딩 시작...")
    try:
        data = np.load(DATASET_PATH)
        x_raw, y_raw = data['x_data'], data['y_data']
    except FileNotFoundError:
        print(f"오류: {DATASET_PATH} 파일이 없습니다.")
        exit()

    # 데이터 서브셋 추출 (속도 향상을 위해 OPTUNA_SUBSET_SIZE 만큼 랜덤 샘플링)
    all_indices = list(range(len(x_raw)))
    random.shuffle(all_indices)
    subset_indices = all_indices[:OPTUNA_SUBSET_SIZE]
    
    x_subset = x_raw[subset_indices]
    y_subset = y_raw[subset_indices]

    # 서브셋 내에서 다시 Train/Val 분할
    train_size = int(0.8 * OPTUNA_SUBSET_SIZE)
    val_size = OPTUNA_SUBSET_SIZE - train_size
    
    # 스케일링
    print(f"데이터 {OPTUNA_SUBSET_SIZE}개에 대해 스케일링 적용 중...")
    scaler_x, scaler_y = StandardScaler(), StandardScaler()
    scaler_x.fit(x_subset[:train_size].reshape(-1, INPUT_DIM))
    scaler_y.fit(y_subset[:train_size].reshape(-1, OUTPUT_DIM))
    joblib.dump(scaler_x, SCALER_X_PATH)
    joblib.dump(scaler_y, SCALER_Y_PATH)

    x_scaled = scaler_x.transform(x_subset.reshape(-1, INPUT_DIM)).reshape(x_subset.shape)
    y_scaled = scaler_y.transform(y_subset.reshape(-1, OUTPUT_DIM)).reshape(y_subset.shape)

    full_dataset = TrajectoryDataset(x_scaled, y_scaled)
    train_dataset = Subset(full_dataset, range(train_size))
    val_dataset = Subset(full_dataset, range(train_size, OPTUNA_SUBSET_SIZE))

    # Optuna Study 실행
    # MedianPruner를 사용하여 성능이 안 좋은 조합은 빠르게 포기하게 설정
    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
    
    print(f"Optuna 탐색 시작 (횟수: {N_TRIALS}, 데이터 규모: {OPTUNA_SUBSET_SIZE})")
    study.optimize(lambda trial: objective(trial, train_dataset, val_dataset), n_trials=N_TRIALS) 

    print("\n" + "="*50)
    print("최적의 하이퍼파라미터 조합:")
    for key, value in study.best_trial.params.items():
        print(f"  - {key}: {value}")
    print(f"최고 Validation Loss: {study.best_trial.value:.6f}")
    print("="*50)