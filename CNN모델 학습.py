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

# Matplotlib 한글 설정 (이미지 저장은 하지 않으나 설정은 유지)
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==============================================================================
# 1. 설정 (기존 MLP 학습 환경과 동일하게 유지)
# ==============================================================================
DATASET_PATH = 'dataset_td_0.0-15.0_doa_0.0-0.5_500M.npz'
MODEL_SAVE_PATH = 'model_cnn_td_0.0-15.0_doa_0.0-0.5_500M.pt'

# 기존 스케일러 경로
SCALER_X_PATH = 'scaler_x_td_0.0-15.0_doa_0.0-0.5_500M.pkl'
SCALER_Y_PATH = 'scaler_y_td_0.0-15.0_doa_0.0-0.5_500M.pkl'

# 하이퍼파라미터
BATCH_SIZE = 512
EPOCHS = 200
LEARNING_RATE = 3e-5  
WEIGHT_DECAY = 1e-5 
DROPOUT_RATE = 0.3    

WINDOW_SIZE = 20
INPUT_DIM = 25
OUTPUT_DIM = 3

# ==============================================================================
# 2. 1D-CNN 모델 정의
# ==============================================================================
class CNN1DModel(nn.Module):
    def __init__(self, input_dim, output_dim, window_size, dropout):
        super(CNN1DModel, self).__init__()
        # PyTorch Conv1d는 (Batch, Channels, Seq_len) 형태를 입력받음
        # input_dim(25)이 채널이 됨
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU()
        )
        
        # 각 타임스텝별로 좌표를 출력하기 위해 1x1 Convolution 사용
        self.output_layer = nn.Conv1d(128, output_dim, kernel_size=1)

    def forward(self, x):
        # x: (Batch, Seq_len, Input_dim) -> (Batch, 25, 20)로 변환
        x = x.transpose(1, 2)
        
        # 특징 추출
        features = self.conv_layers(x)
        
        # 출력: (Batch, 3, 20)
        out = self.output_layer(features)
        
        # 다시 (Batch, Seq_len, Output_dim) 형태로 복구
        return out.transpose(1, 2)

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
    print(f"1D-CNN 모델 학습 준비 중 (Device: {DEVICE})")
    
    if not os.path.exists(DATASET_PATH):
        print(f"데이터셋 파일을 찾을 수 없습니다: {DATASET_PATH}")
        sys.exit()

    data = np.load(DATASET_PATH)
    x_raw, y_raw = data['x_data'], data['y_data']
    
    # 스케일러 로드
    scaler_x = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)

    # 데이터 분할 및 스케일링
    indices = list(range(len(x_raw)))
    random.shuffle(indices)
    train_split = int(0.9 * len(x_raw))
    train_indices = indices[:train_split]
    val_indices = indices[train_split:]

    x_scaled = scaler_x.transform(x_raw.reshape(-1, INPUT_DIM)).reshape(x_raw.shape)
    y_scaled = scaler_y.transform(y_raw.reshape(-1, OUTPUT_DIM)).reshape(y_raw.shape)

    train_loader = DataLoader(Subset(TrajectoryDataset(x_scaled, y_scaled), train_indices), 
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Subset(TrajectoryDataset(x_scaled, y_scaled), val_indices), 
                            batch_size=BATCH_SIZE, shuffle=False)

    model = CNN1DModel(INPUT_DIM, OUTPUT_DIM, WINDOW_SIZE, DROPOUT_RATE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_loss = float('inf')
    
    print(f"1D-CNN 학습 시작 (800m 환경)")
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
            status = "(New Best!)"
        else:
            status = ""
        
        print(f"Epoch {epoch+1:03} | Train Loss: {avg_train_loss:.8f} | Valid Loss: {avg_val_loss:.8f} {status}")

    print(f"[완료] 1D-CNN 모델이 '{MODEL_SAVE_PATH}'에 저장되었습니다.")