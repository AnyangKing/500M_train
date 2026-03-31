import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from tqdm import tqdm
import random
import joblib
import os
import sys

# Matplotlib 설정 (한글 깨짐 방지용, 저장 로직은 제외) [cite: 2025-09-04]
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==============================================================================
# 1. 설정 (800m 데이터셋 및 기존 스케일러 경로)
# ==============================================================================
DATASET_PATH = 'dataset_td_0.0-15.0_doa_0.0-0.5_500M.npz'
MODEL_SAVE_PATH = 'model_lstm_td_0.0-15.0_doa_0.0-0.5_500M.pt'

# 기존 800m용 스케일러 경로
SCALER_X_PATH = 'scaler_x_td_0.0-15.0_doa_0.0-0.5_500M.pkl'
SCALER_Y_PATH = 'scaler_y_td_0.0-15.0_doa_0.0-0.5_500M.pkl'

# 하이퍼파라미터
HIDDEN_DIM = 256
NUM_LAYERS = 3
BATCH_SIZE = 512
EPOCHS = 200
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-5
DROPOUT_RATE = 0.3

WINDOW_SIZE = 20
INPUT_DIM = 25
OUTPUT_DIM = 3

# ==============================================================================
# 2. LSTM 모델 정의
# ==============================================================================
class LSTMModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, dropout):
        super(LSTMModel, self).__init__()
        # num_layers가 1보다 클 때만 dropout 적용 가능
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, 
                            dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        # 모든 시점의 결과를 출력 (seq_len 유지)
        return self.fc(out)

class TrajectoryDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x = torch.FloatTensor(x_data)
        self.y = torch.FloatTensor(y_data)
    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

# ==============================================================================
# 3. 학습 실행
# ==============================================================================
if __name__ == '__main__':
    print(f"800m 데이터셋 로딩 및 전처리 중...")
    
    if not os.path.exists(DATASET_PATH):
        print(f"데이터셋 파일을 찾을 수 없습니다: {DATASET_PATH}")
        sys.exit()

    data = np.load(DATASET_PATH)
    x_raw, y_raw = data['x_data'], data['y_data']
    
    # 기존 스케일러 로드
    try:
        scaler_x = joblib.load(SCALER_X_PATH)
        scaler_y = joblib.load(SCALER_Y_PATH)
    except FileNotFoundError:
        print("스케일러 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        sys.exit()

    # 데이터 스케일링
    x_scaled = scaler_x.transform(x_raw.reshape(-1, INPUT_DIM)).reshape(x_raw.shape)
    y_scaled = scaler_y.transform(y_raw.reshape(-1, OUTPUT_DIM)).reshape(y_raw.shape)

    # 데이터 분할 (9:1)
    indices = list(range(len(x_raw)))
    random.seed(42) # 재현성을 위한 시드 고정
    random.shuffle(indices)
    train_split = int(0.9 * len(x_raw))
    train_indices = indices[:train_split]
    val_indices = indices[train_split:]

    train_loader = DataLoader(Subset(TrajectoryDataset(x_scaled, y_scaled), train_indices), 
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Subset(TrajectoryDataset(x_scaled, y_scaled), val_indices), 
                            batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMModel(INPUT_DIM, OUTPUT_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT_RATE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_loss = float('inf')
    
    print(f"LSTM 학습 시작 (HIDDEN: {HIDDEN_DIM}, LAYERS: {NUM_LAYERS})")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        for x_b, y_b in pbar:
            x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x_b), y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
                val_loss += criterion(model(x_b), y_b).item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            new_best_str = "(New Best!)"
        else:
            new_best_str = ""
        
        print(f"Epoch {epoch+1:03} | Train Loss: {avg_train_loss:.8f} | Valid Loss: {avg_val_loss:.8f} {new_best_str}")

    print(f"[완료] LSTM 학습 완료. 모델이 '{MODEL_SAVE_PATH}'에 저장되었습니다.")