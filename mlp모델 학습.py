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

# Matplotlib 한글 설정 (이미지 저장은 하지 않으나 설정은 유지) [cite: 2025-09-04]
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==============================================================================
# 1. 설정 (800m 데이터셋 및 기존 트랜스포머 스케일러 로드)
# ==============================================================================
DATASET_PATH = 'dataset_td_0.0-15.0_doa_0.0-0.5_500M.npz'
MODEL_SAVE_PATH = 'model_mlp_td_0.0-15.0_doa_0.0-0.5_500M.pt'

# 트랜스포머 모델 학습 시 생성된 800m용 스케일러 경로
SCALER_X_PATH = 'scaler_x_td_0.0-15.0_doa_0.0-0.5_500M.pkl'
SCALER_Y_PATH = 'scaler_y_td_0.0-15.0_doa_0.0-0.5_500M.pkl'

# 하이퍼파라미터 (제공해주신 mlp학습.py 기준)
HIDDEN_DIM = 1024
BATCH_SIZE = 512
EPOCHS = 200
LEARNING_RATE = 3e-5  
WEIGHT_DECAY = 1e-5   
DROPOUT_RATE = 0.3    

WINDOW_SIZE = 20
INPUT_DIM = 25
OUTPUT_DIM = 3

# ==============================================================================
# 2. 모델 및 데이터셋 정의
# ==============================================================================
class MLPModel(nn.Module):
    def __init__(self, input_dim, output_dim, window_size, hidden_dim, dropout):
        super(MLPModel, self).__init__()
        self.window_size = window_size
        self.input_dim = input_dim
        
        self.net = nn.Sequential(
            nn.Linear(window_size * input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            
            nn.Linear(hidden_dim, window_size * output_dim)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        out = self.net(x)
        return out.view(batch_size, self.window_size, -1)

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
    print(f"800m 데이터셋 및 기존 스케일러 로딩 중...")
    
    # 데이터 존재 여부 확인
    if not os.path.exists(DATASET_PATH):
        print(f"데이터셋 파일을 찾을 수 없습니다: {DATASET_PATH}")
        sys.exit()

    data = np.load(DATASET_PATH)
    x_raw, y_raw = data['x_data'], data['y_data']
    
    # 기존 스케일러 로드 (fit 없이 transform만 수행)
    try:
        scaler_x = joblib.load(SCALER_X_PATH)
        scaler_y = joblib.load(SCALER_Y_PATH)
    except FileNotFoundError:
        print("기존 스케일러 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        sys.exit()

    indices = list(range(len(x_raw)))
    random.shuffle(indices)
    train_split = int(0.9 * len(x_raw))
    train_indices = indices[:train_split]
    val_indices = indices[train_split:]

    # 기존 스케일러로 변환 수행
    x_scaled = scaler_x.transform(x_raw.reshape(-1, INPUT_DIM)).reshape(x_raw.shape)
    y_scaled = scaler_y.transform(y_raw.reshape(-1, OUTPUT_DIM)).reshape(y_raw.shape)

    train_loader = DataLoader(Subset(TrajectoryDataset(x_scaled, y_scaled), train_indices), 
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Subset(TrajectoryDataset(x_scaled, y_scaled), val_indices), 
                            batch_size=BATCH_SIZE, shuffle=False)

    model = MLPModel(INPUT_DIM, OUTPUT_DIM, WINDOW_SIZE, HIDDEN_DIM, DROPOUT_RATE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_loss = float('inf')
    history = {'train': [], 'val': []}

    print(f"MLP 학습 시작 (LR: {LEARNING_RATE}, Dropout: {DROPOUT_RATE}, 800m 환경)")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        # tqdm을 사용한 진행도 출력
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
        history['train'].append(avg_train_loss)
        history['val'].append(avg_val_loss)

        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            new_best_str = "(New Best!)"
        else:
            new_best_str = ""
        
        # 학습 로그 출력 (이미지 저장은 생략)
        print(f"Epoch {epoch+1:03} | Train Loss: {avg_train_loss:.8f} | Valid Loss: {avg_val_loss:.8f} {new_best_str}")

    print(f"[완료] 학습이 끝났습니다. 최적 모델이 '{MODEL_SAVE_PATH}'에 저장되었습니다.")