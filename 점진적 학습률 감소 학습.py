import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from tqdm import tqdm
import random
import time
import math
from sklearn.preprocessing import StandardScaler
import joblib
import os
import matplotlib.pyplot as plt

# Matplotlib 한글 폰트 및 마이너스 기호 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ==============================================================================
# 1. 설정 (Optuna 결과 반영)
# ==============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 경로 설정
DATASET_PATH = 'dataset_td_0.0-15.0_doa_0.0-0.5_500M.npz'
MODEL_SAVE_PATH = 'model_td_0.0-15.0_doa_0.0-0.5_500M.pt'
SCALER_X_PATH = 'scaler_x_td_0.0-15.0_doa_0.0-0.5_500M.pkl'
SCALER_Y_PATH = 'scaler_y_td_0.0-15.0_doa_0.0-0.5_500M.pkl'

# Optuna 최적 하이퍼파라미터 적용
D_MODEL = 128
N_LAYERS = 9
N_HEADS = 8
MAX_LR = 0.00047805672196318984
BATCH_SIZE = 256
DROPOUT = 0.053395242165250906

# 공통 설정
INPUT_DIM = 25 
OUTPUT_DIM = 3
MAX_LEN = 20          # 20step 데이터셋 고정
EPOCHS = 200          
WARMUP_EPOCHS = 10    # 초기 안정화를 위한 워밍업 기간
MIN_LR = 1e-6         # 코사인 스케줄러의 최종 학습률
GRAD_CLIP = 1.0       # 그래디언트 폭주 방지 임계값

# ==============================================================================
# 2. 모델 아키텍처 정의
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
    def __init__(self, input_dim, output_dim, d_model, nhead, nlayers, dropout=0.1):
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
        output = self.transformer_encoder(src)
        return self.fc_out(output)

class TrajectoryDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x = torch.FloatTensor(x_data)
        self.y = torch.FloatTensor(y_data)
    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

# ==============================================================================
# 3. 훈련 루프 정의
# ==============================================================================
def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for x_batch, y_batch in tqdm(dataloader, desc="Training", leave=False):
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x_batch), y_batch)
        loss.backward()
        
        # 그래디언트 클리핑 적용
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

def evaluate_epoch(model, dataloader, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            loss = criterion(model(x_batch), y_batch)
            epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

# ==============================================================================
# 4. 메인 실행 블록
# ==============================================================================
if __name__ == '__main__':
    print("데이터 로딩 및 스케일러 학습 시작...")
    try:
        data = np.load(DATASET_PATH)
        x_raw, y_raw = data['x_data'], data['y_data']
    except FileNotFoundError:
        print(f"오류: {DATASET_PATH} 파일이 없습니다.")
        exit()

    indices = list(range(len(x_raw)))
    random.shuffle(indices)
    train_size = int(0.8 * len(x_raw))
    val_size = int(0.1 * len(x_raw))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]

    scaler_x, scaler_y = StandardScaler(), StandardScaler()
    scaler_x.fit(x_raw[train_indices].reshape(-1, INPUT_DIM))
    scaler_y.fit(y_raw[train_indices].reshape(-1, OUTPUT_DIM))
    joblib.dump(scaler_x, SCALER_X_PATH)
    joblib.dump(scaler_y, SCALER_Y_PATH)

    x_scaled = scaler_x.transform(x_raw.reshape(-1, INPUT_DIM)).reshape(x_raw.shape)
    y_scaled = scaler_y.transform(y_raw.reshape(-1, OUTPUT_DIM)).reshape(y_raw.shape)

    train_loader = DataLoader(Subset(TrajectoryDataset(x_scaled, y_scaled), train_indices), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Subset(TrajectoryDataset(x_scaled, y_scaled), val_indices), batch_size=BATCH_SIZE, shuffle=False)

    # 모델 및 최적화 설정
    model = TransformerEncoderOnlyModel(INPUT_DIM, OUTPUT_DIM, D_MODEL, N_HEADS, N_LAYERS, DROPOUT).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=MAX_LR)
    criterion = nn.MSELoss()

    # 코사인 스케줄러 설정
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS-WARMUP_EPOCHS, eta_min=MIN_LR)

    best_valid_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'lrs': []}

    print(f"학습 시작 (D_MODEL: {D_MODEL}, N_LAYERS: {N_LAYERS}, MAX_LR: {MAX_LR:.2e})")
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        # Linear Warmup 적용
        if epoch < WARMUP_EPOCHS:
            curr_lr = (MAX_LR / WARMUP_EPOCHS) * (epoch + 1)
            for param_group in optimizer.param_groups:
                param_group['lr'] = curr_lr
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        valid_loss = evaluate_epoch(model, val_loader, criterion)
        
        # 워밍업 이후 코사인 스케줄러 업데이트
        if epoch >= WARMUP_EPOCHS:
            scheduler.step()
        
        curr_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(train_loss)
        history['val_loss'].append(valid_loss)
        history['lrs'].append(curr_lr)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            status = "(New Best!)"
        else:
            status = ""

        mins, secs = divmod(time.time() - start_time, 60)
        print(f"Epoch {epoch+1:03} | LR: {curr_lr:.2e} | Train Loss: {train_loss:.6f} | Valid Loss: {valid_loss:.6f} {status}")

    # 결과 시각화
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Valid')
    plt.title('Loss History')
    plt.yscale('log'); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['lrs'], color='green')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epochs')
    
    plt.tight_layout()
    plt.show()

    print(f"\n학습 완료. 최저 검증 손실: {best_valid_loss:.6f}")