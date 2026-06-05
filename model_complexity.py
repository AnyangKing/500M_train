import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import numpy as np
import math
import time
import sys
import joblib
import importlib.util
from pathlib import Path

plt_import = True
try:
    import matplotlib
    matplotlib.use('Agg')
except:
    plt_import = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WINDOW_SIZE = 20
INPUT_DIM  = 25
OUTPUT_DIM = 3
SOUND_SPEED_CM_S = 150000.0
WARMUP = 100
ITER   = 2000   # 추론 시간 측정 반복 횟수

ROOT = Path(__file__).resolve().parent
FINAL_CODE_PATH = ROOT / "코드_최종.py"

def load_final_code():
    spec = importlib.util.spec_from_file_location("final_code", FINAL_CODE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

FINAL_CODE = load_final_code()

# ==============================================================================
# 모델 아키텍처 (코드_최종.py와 동일)
# ==============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=20):
        super().__init__()
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
    def __init__(self, input_dim, output_dim, d_model, nhead, nlayers, dropout=0.0534):
        super().__init__()
        self.d_model = d_model
        self.encoder_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=WINDOW_SIZE)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=nlayers)
        self.fc_out = nn.Linear(d_model, output_dim)
    def forward(self, src):
        src = self.encoder_embedding(src) * math.sqrt(self.d_model)
        return self.fc_out(self.transformer_encoder(self.pos_encoder(src)))

class LSTMModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)

class MLPModel(nn.Module):
    def __init__(self, input_dim, output_dim, window_size, hidden_dim, dropout=0.3):
        super().__init__()
        self.window_size = window_size
        self.net = nn.Sequential(
            nn.Linear(window_size * input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, window_size * output_dim)
        )
    def forward(self, x):
        bs = x.size(0)
        x = x.view(bs, -1)
        return self.net(x).view(bs, self.window_size, -1)

class CNN1DModel(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.3):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, 128, 3, padding=1), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(128, 256, 3, padding=1),       nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(256, 128, 3, padding=1),       nn.BatchNorm1d(128), nn.GELU()
        )
        self.output_layer = nn.Conv1d(128, output_dim, 1)
    def forward(self, x):
        x = x.transpose(1, 2)
        return self.output_layer(self.conv_layers(x)).transpose(1, 2)

# ==============================================================================
# MUSIC-LS 추론 함수
# ==============================================================================
# MUSIC 관련 (추론 시간 측정용)
MUSIC_F0     = 32000.0
MUSIC_N_SNAP = 64
MUSIC_SNR    = 316.0   # 25dB
MUSIC_AZ_RES = 1        # azimuth search resolution (deg)
MUSIC_EL_RES = 1        # elevation search resolution (deg)
_AZ_NP = np.radians(np.arange(-180, 180, MUSIC_AZ_RES))
_EL_NP = np.radians(np.arange(-90, 90, MUSIC_EL_RES))
_AZ_GRID, _EL_GRID = np.meshgrid(_AZ_NP, _EL_NP)
_AZ_FLAT = _AZ_GRID.ravel().astype(np.float64)
_EL_FLAT = _EL_GRID.ravel().astype(np.float64)
_D_MAT   = np.stack([np.cos(_EL_FLAT)*np.cos(_AZ_FLAT),
                      np.cos(_EL_FLAT)*np.sin(_AZ_FLAT),
                      np.sin(_EL_FLAT)], axis=1)

def music_doa_and_localize(raw_signal_t, feat_t, music_localizer):
    doa_vec = FINAL_CODE.music_doa_estimation_stable(FINAL_CODE.SENSORS_CM, raw_signal_t)
    return music_localizer.update(doa_vec, feat_t)

# ==============================================================================
# 유틸리티
# ==============================================================================
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_time_nn(model, dummy_input):
    """신경망 모델: 슬라이딩 윈도우 1개(batch=1) 기준"""
    model.eval()
    with torch.no_grad():
        for _ in range(WARMUP):
            _ = model(dummy_input)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(ITER):
            _ = model(dummy_input)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    return (time.perf_counter() - t0) / ITER * 1000  # ms

def measure_time_music(raw_dummy, feat_dummy):
    """MUSIC-LS: DOA estimation + TDOA/observed-DOA assisted localization, 1 step"""
    music_localizer = FINAL_CODE.MusicLocalizer(FINAL_CODE.SENSORS_CM)
    for _ in range(WARMUP):
        music_doa_and_localize(raw_dummy, feat_dummy, music_localizer)
    t0 = time.perf_counter()
    for _ in range(ITER):
        music_doa_and_localize(raw_dummy, feat_dummy, music_localizer)
    return (time.perf_counter() - t0) / ITER * 1000  # ms/step

# ==============================================================================
# 메인
# ==============================================================================
if __name__ == '__main__':
    MODEL_DIR = '.'   # 모델 파일들이 있는 폴더 (코드_최종.py와 같은 폴더에서 실행)

    PATHS = {
        'Proposed':  'model_td_0.0-15.0_doa_0.0-0.5_500M.pt',
        'LSTM':      'model_lstm_td_0.0-15.0_doa_0.0-0.5_500M.pt',
        'MLP':       'model_mlp_td_0.0-15.0_doa_0.0-0.5_500M.pt',
        '1D-CNN':    'model_cnn_td_0.0-15.0_doa_0.0-0.5_500M.pt',
    }

    # 모델 인스턴스 생성 (500M 학습 하이퍼파라미터 기준)
    models = {
        'Proposed': TransformerEncoderOnlyModel(INPUT_DIM, OUTPUT_DIM,
                        d_model=128, nhead=8, nlayers=9, dropout=0.0534).to(DEVICE),
        'LSTM':     LSTMModel(INPUT_DIM, OUTPUT_DIM,
                        hidden_dim=256, num_layers=3, dropout=0.3).to(DEVICE),
        'MLP':      MLPModel(INPUT_DIM, OUTPUT_DIM,
                        window_size=WINDOW_SIZE, hidden_dim=1024, dropout=0.3).to(DEVICE),
        '1D-CNN':   CNN1DModel(INPUT_DIM, OUTPUT_DIM, dropout=0.3).to(DEVICE),
    }

    # 가중치 로드
    print("모델 가중치 로드 중...")
    for name, path in PATHS.items():
        full_path = os.path.join(MODEL_DIR, path)
        if os.path.exists(full_path):
            models[name].load_state_dict(torch.load(full_path, map_location=DEVICE))
            models[name].eval()
            print(f"  [OK] {name}: {path}")
        else:
            print(f"  [WARN] {name}: 파일 없음 ({path}) - 파라미터 수는 계산 가능")

    # 더미 입력 생성
    dummy_nn     = torch.randn(1, WINDOW_SIZE, INPUT_DIM).to(DEVICE)
    dummy_tdoa   = np.random.randn(7).astype(np.float64) * 5.0  # cm 단위 TDOA
    dummy_feat   = np.zeros(25, dtype=np.float32)
    dummy_feat[0]   = 40000.0   # d0 (cm)
    dummy_feat[2:9] = dummy_tdoa
    dummy_feat[9:17] = 0.0      # observed azimuth
    dummy_feat[17:25] = 0.0     # observed elevation
    noise_std = 1.0 / np.sqrt(2.0 * 316.0)
    dummy_signal = (np.random.randn(8, MUSIC_N_SNAP)
                    + 1j * np.random.randn(8, MUSIC_N_SNAP)) * noise_std

    # ==============================================================================
    # 파라미터 수 측정
    # ==============================================================================
    print("\n파라미터 수 계산 중...")
    params = {}
    for name, model in models.items():
        params[name] = count_params(model)
    params['MUSIC-LS'] = None

    # ==============================================================================
    # 추론 시간 측정
    # ==============================================================================
    print(f"추론 시간 측정 중 (ITER={ITER}, device={DEVICE})...")
    times = {}
    for name, model in models.items():
        t = measure_time_nn(model, dummy_nn)
        times[name] = t
        print(f"  {name:<12}: {t:.4f} ms/step")

    t_music = measure_time_music(dummy_signal, dummy_feat)
    times['MUSIC-LS'] = t_music
    print(f"  {'MUSIC-LS':<12}: {t_music:.4f} ms/step")

    # ==============================================================================
    # 결과 출력 (논문 표 3 양식)
    # ==============================================================================
    order = ['Proposed', 'LSTM', 'MLP', '1D-CNN', 'MUSIC-LS']

    print(f"\n{'='*65}")
    print(f" [Table 3. Computational Complexity Comparison]")
    print(f"{'='*65}")
    print(f"{'Model':<14} | {'Parameters':>18} | {'Inference Time (ms/step)':>24}")
    print(f"{'-'*65}")
    for name in order:
        p = params.get(name)
        t = times.get(name)
        p_str = f"{p:,}" if p is not None else "N/A"
        t_str = f"{t:.4f}" if t is not None else "-"
        print(f"{name:<14} | {p_str:>18} | {t_str:>24}")
    print(f"{'='*65}")
    print(f"* Device: {DEVICE}")
    print(f"* Measurement: {ITER} iterations, batch size = 1, sequence length = {WINDOW_SIZE}")
    print(f"* MUSIC-LS: DOA estimation + TDOA/observed-DOA assisted localization per timestep")
