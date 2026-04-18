import torch
import torch.nn as nn
import numpy as np
import joblib
import math
import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.mplot3d import Axes3D

# Matplotlib 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LEN, WINDOW_SIZE = 20, 20
INPUT_DIM, OUTPUT_DIM = 25, 3
SOUND_SPEED_CM_S = 150000.0

# ==============================================================================
# 1. 모델 아키텍처 정의
# ==============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=20):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term); pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0); self.register_buffer('pe', pe)
    def forward(self, x): return self.dropout(x + self.pe[:, :x.size(1), :])

class TransformerEncoderOnlyModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, nhead, nlayers, dropout=0.0528):
        super(TransformerEncoderOnlyModel, self).__init__()
        self.d_model = d_model
        self.encoder_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=MAX_LEN)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=nlayers)
        self.fc_out = nn.Linear(d_model, output_dim)
    def forward(self, src):
        src = self.encoder_embedding(src) * math.sqrt(self.d_model)
        return self.fc_out(self.transformer_encoder(self.pos_encoder(src)))

class LSTMModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x): out, _ = self.lstm(x); return self.fc(out)

class MLPModel(nn.Module):
    def __init__(self, input_dim, output_dim, window_size, hidden_dim, dropout=0.3):
        super(MLPModel, self).__init__()
        self.window_size = window_size
        self.net = nn.Sequential(
            nn.Linear(window_size * input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim * 2), nn.BatchNorm1d(hidden_dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, window_size * output_dim)
        )
    def forward(self, x):
        bs = x.size(0); x = x.view(bs, -1); out = self.net(x); return out.view(bs, self.window_size, -1)

class CNN1DModel(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.3):
        super(CNN1DModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, 128, 3, padding=1), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(128, 256, 3, padding=1), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(256, 128, 3, padding=1), nn.BatchNorm1d(128), nn.GELU()
        ); self.output_layer = nn.Conv1d(128, output_dim, 1)
    def forward(self, x): x = x.transpose(1, 2); out = self.output_layer(self.conv_layers(x)); return out.transpose(1, 2)

class KalmanFilter:
    def __init__(self, init_pos):
        self.x = np.array([init_pos[0], init_pos[1], init_pos[2], 0, 0, 0])
        self.F = np.eye(6); self.F[0,3]=self.F[1,4]=self.F[2,5]=1.0
        self.H = np.zeros((3, 6)); self.H[0,0]=self.H[1,1]=self.H[2,2]=1.0
        self.P = np.eye(6)*500
        # Q: 궤적 특성에 맞춘 과정 잡음 공분산 (위치:속도 = 1:100)
        self.Q = np.diag([100, 100, 100, 10000, 10000, 10000])
        self.R = np.eye(3)*100
    def predict_and_update(self, z):
        self.x = self.F @ self.x; self.P = self.F @ self.P @ self.F.T + self.Q
        S = self.H @ self.P @ self.H.T + self.R; K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x); self.P = (np.eye(6) - K @ self.H) @ self.P
        return self.x[:3]

# ==============================================================================
# 2. 물리 기반 데이터 생성
# ==============================================================================
MUSIC_F0 = 32000.0          # 중심 주파수 (Hz)
MUSIC_N_SNAP = 64           # 스냅샷 수
MUSIC_AZ_RES = 2            # azimuth 탐색 해상도 (deg)
MUSIC_EL_RES = 2            # elevation 탐색 해상도 (deg)
MUSIC_SNR_LINEAR = 316.0     # SNR=25dB

# 탐색 격자 사전 계산
_AZ_NP = np.radians(np.arange(-180, 180, MUSIC_AZ_RES))
_EL_NP = np.radians(np.arange(-90,  90,  MUSIC_EL_RES))
_AZ_GRID, _EL_GRID = np.meshgrid(_AZ_NP, _EL_NP)
_AZ_FLAT_NP = _AZ_GRID.ravel().astype(np.float64)
_EL_FLAT_NP = _EL_GRID.ravel().astype(np.float64)
_D_MAT_NP = np.stack([np.cos(_EL_FLAT_NP)*np.cos(_AZ_FLAT_NP),
                       np.cos(_EL_FLAT_NP)*np.sin(_AZ_FLAT_NP),
                       np.sin(_EL_FLAT_NP)], axis=1)

r_cm, L_cm = 3.3, 7.9
def get_sensors_cm():
    S2 = np.sqrt(2)
    return np.array([[r_cm, 0, 0], [r_cm/S2, r_cm/S2, -L_cm], [0, r_cm, 0], [-r_cm/S2, r_cm/S2, -L_cm],
                     [-r_cm, 0, 0], [-r_cm/S2, -r_cm/S2, -L_cm], [0, -r_cm, 0], [r_cm/S2, -r_cm/S2, -L_cm]], dtype=np.float32)

SENSORS_CM = get_sensors_cm()
SENSOR_CENTER_CM = np.mean(SENSORS_CM, axis=0)
SENSORS_M_GPU = torch.tensor(SENSORS_CM.astype(np.float64) / 100.0, dtype=torch.float64, device=DEVICE)
_D_MAT_GPU = torch.tensor(_D_MAT_NP, dtype=torch.float64, device=DEVICE)
_MUSIC_LAMBDA_M = (SOUND_SPEED_CM_S / 100.0) / MUSIC_F0
_MUSIC_PROJ_GPU = _D_MAT_GPU @ SENSORS_M_GPU.T
_MUSIC_STEERING_GPU = torch.exp(-1j * 2 * math.pi * _MUSIC_PROJ_GPU / _MUSIC_LAMBDA_M).to(torch.complex128)
_MUSIC_STEERING_GPU = _MUSIC_STEERING_GPU / (torch.linalg.norm(_MUSIC_STEERING_GPU, dim=1, keepdim=True) + 1e-12)
_SW_COUNTS = np.minimum(np.arange(1, 201), WINDOW_SIZE)
_SW_COUNTS = np.minimum(_SW_COUNTS, _SW_COUNTS[::-1]).astype(np.float32).reshape(-1, 1)

def generate_controlled_traj_cm(td_noise_cm, doa_noise_deg, target_dist_cm=None, m_bias_cm=0.0):
    sensors = get_sensors_cm(); traj = np.zeros((200, 3), dtype=np.float32)
    direction = np.random.randn(3); direction /= (np.linalg.norm(direction) + 1e-9)
    traj[0] = direction * target_dist_cm; vec = np.random.randn(3); vec /= (np.linalg.norm(vec) + 1e-9)
    for i in range(1, 200):
        rv = np.random.randn(3); rv /= (np.linalg.norm(rv) + 1e-9)
        vec = 0.8*vec + 0.2*rv; vec /= (np.linalg.norm(vec)+1e-9); traj[i] = traj[i-1] + vec*100.0

    feats = np.zeros((200, 25), dtype=np.float32)
    raw_signals = np.zeros((200, 8, MUSIC_N_SNAP), dtype=np.complex128)
    td_std, doa_std = td_noise_cm / SOUND_SPEED_CM_S, np.radians(doa_noise_deg)
    sensor_specific_biases = np.random.normal(m_bias_cm, m_bias_cm * 0.5 + 1e-9, size=8) / SOUND_SPEED_CM_S
    noise_std = 1.0 / np.sqrt(2.0 * MUSIC_SNR_LINEAR)
    for i, p in enumerate(traj):
        d = np.linalg.norm(sensors - p, axis=1)
        toa = (d / SOUND_SPEED_CM_S) + sensor_specific_biases + np.random.normal(0, td_std, size=8)
        dp = p - sensors
        feats[i] = np.concatenate([toa[0:1]*SOUND_SPEED_CM_S, (toa-toa[0])*SOUND_SPEED_CM_S,
                                   np.arctan2(dp[:,1], dp[:,0]) + np.random.normal(0, doa_std, 8),
                                   np.arctan2(dp[:,2], np.sqrt(dp[:,0]**2 + dp[:,1]**2)+1e-9) + np.random.normal(0, doa_std, 8)])
        s = np.exp(1j * np.random.uniform(0, 2 * np.pi, MUSIC_N_SNAP))
        direction = (p - np.mean(sensors, axis=0)) / (np.linalg.norm(p - np.mean(sensors, axis=0)) + 1e-9)
        # DOA(방향)만 사용해 steering vector 생성
        # raw_signals는 MUSIC의 방향 추정용 입력으로만 사용
        steering = np.exp(-1j * 2 * np.pi * MUSIC_F0 / SOUND_SPEED_CM_S * (sensors @ direction))
        X = np.outer(steering, s)
        noise = noise_std * (np.random.randn(8, MUSIC_N_SNAP) + 1j * np.random.randn(8, MUSIC_N_SNAP))
        raw_signals[i] = X + noise
    return traj, feats, raw_signals

def music_doa_estimation_stable(sensors, raw_signals_t):
    n_signals = 1
    X = torch.tensor(raw_signals_t, dtype=torch.complex128, device=DEVICE)
    R = (X @ X.conj().T) / MUSIC_N_SNAP
    eigenvalues, eigenvectors = torch.linalg.eigh(R)
    idx = torch.argsort(eigenvalues, descending=True)
    eigenvectors = eigenvectors[:, idx]
    En = eigenvectors[:, n_signals:]
    if np.array_equal(sensors, SENSORS_CM):
        A = _MUSIC_STEERING_GPU
    else:
        sensors_m = torch.tensor(sensors.astype(np.float64) / 100.0, dtype=torch.float64, device=DEVICE)
        proj = _D_MAT_GPU @ sensors_m.T
        A = torch.exp(-1j * 2 * math.pi * proj / _MUSIC_LAMBDA_M).to(torch.complex128)
        A = A / (torch.linalg.norm(A, dim=1, keepdim=True) + 1e-12)
    En_A = A @ En
    denom = torch.sum(torch.real(En_A * En_A.conj()), dim=1)
    power = 1.0 / (denom + 1e-12)
    best_idx = int(torch.argmax(power).cpu())
    best_az = _AZ_FLAT_NP[best_idx]
    best_el = _EL_FLAT_NP[best_idx]
    est_vec = np.array([np.cos(best_el) * np.cos(best_az),
                        np.cos(best_el) * np.sin(best_az),
                        np.sin(best_el)], dtype=np.float64)
    return est_vec / (np.linalg.norm(est_vec) + 1e-9)

def compute_tdoa_feature_from_pos(pos_cm, sensors):
    dists = np.linalg.norm(sensors.astype(np.float64) - pos_cm.astype(np.float64), axis=1)
    return dists - dists[0]

def localize_music(sensors, estimated_doa, feat_t):
    sensor_center = SENSOR_CENTER_CM if np.array_equal(sensors, SENSORS_CM) else np.mean(sensors, axis=0)
    ref_sensor = sensors[0].astype(np.float64)
    music_range_cm = max(float(feat_t[0]), 0.0)
    doa_unit = estimated_doa / (np.linalg.norm(estimated_doa) + 1e-9)

    candidates = [
        sensor_center + doa_unit * music_range_cm,
        sensor_center - doa_unit * music_range_cm,
        ref_sensor + doa_unit * music_range_cm,
        ref_sensor - doa_unit * music_range_cm,
    ]

    meas_tdoa = feat_t[1:9].astype(np.float64)
    best_pos = candidates[0]
    best_cost = np.inf

    for cand in candidates:
        pred_tdoa = compute_tdoa_feature_from_pos(cand, sensors)
        tdoa_cost = np.mean((pred_tdoa - meas_tdoa) ** 2)
        range0_cost = (np.linalg.norm(cand - ref_sensor) - music_range_cm) ** 2
        center_cost = (np.linalg.norm(cand - sensor_center) - music_range_cm) ** 2
        cost = tdoa_cost + 0.1 * min(range0_cost, center_cost)
        if cost < best_cost:
            best_cost = cost
            best_pos = cand

    return best_pos

def solve_ls_localization(tdoa_values_cm, sensors):
    s0 = sensors[0].astype(np.float64)
    n = len(tdoa_values_cm)
    A = np.zeros((n, 4))
    b = np.zeros(n)
    for i in range(n):
        si = sensors[i + 1].astype(np.float64)
        di = tdoa_values_cm[i]
        A[i, :3] = 2.0 * (s0 - si)
        A[i, 3]  = -2.0 * di
        b[i] = di**2 + np.dot(s0, s0) - np.dot(si, si)
    result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return result[:3]

def sliding_window_inference_cm(model, sx, sy, x_raw_cm):
    model.eval(); x_scaled = sx.transform(x_raw_cm).astype(np.float32, copy=False)
    windows = np.lib.stride_tricks.sliding_window_view(x_scaled, window_shape=WINDOW_SIZE, axis=0)
    windows = np.swapaxes(windows, 1, 2)
    windows = torch.from_numpy(np.ascontiguousarray(windows)).to(DEVICE)
    final = np.zeros((200, 3), dtype=np.float32)
    with torch.no_grad():
        preds = model(windows).cpu().numpy()
        for i in range(len(preds)): final[i:i+WINDOW_SIZE, :] += preds[i]
    return sy.inverse_transform(final / (_SW_COUNTS + 1e-9))

def calculate_rmse(gt, pred, dims=[0, 1, 2]):
    return np.sqrt(np.mean(np.sum((gt[:, dims] - pred[:, dims])**2, axis=1)))

def get_axis_limits_from_tracks(tracks, dim, padding_ratio=0.08, min_padding=5.0):
    values = np.concatenate([track[:, dim] for track in tracks])
    vmin, vmax = float(np.min(values)), float(np.max(values))
    span = vmax - vmin
    pad = max(span * padding_ratio, min_padding)
    return vmin - pad, vmax + pad

# ==============================================================================
# 3. 메인 분석부
# ==============================================================================
if __name__ == '__main__':
    model_styles = {
        'Proposed': {'marker': 'o', 'color': 'r', 'ls': '-'}, 'MUSIC': {'marker': 'D', 'color': 'green', 'ls': '--'},
        'LSTM': {'marker': 's', 'color': 'm', 'ls': '-'}, 'MLP': {'marker': '^', 'color': 'b', 'ls': '-'},
        'KF': {'marker': 'P', 'color': 'orange', 'ls': '-'}, 'CNN': {'marker': 'x', 'color': 'c', 'ls': '-'}
    }
    CONFIG = {
        'proposed_path': 'model_td_0.0-15.0_doa_0.0-0.5_500M.pt', 'lstm_path': 'model_lstm_td_0.0-15.0_doa_0.0-0.5_500M.pt',
        'mlp_path': 'model_mlp_td_0.0-15.0_doa_0.0-0.5_500M.pt', 'cnn_path': 'model_cnn_td_0.0-15.0_doa_0.0-0.5_500M.pt',
        'scaler_x': 'scaler_x_td_0.0-15.0_doa_0.0-0.5_500M.pkl', 'scaler_y': 'scaler_y_td_0.0-15.0_doa_0.0-0.5_500M.pkl'
    }
    try:
        sx, sy = joblib.load(CONFIG['scaler_x']), joblib.load(CONFIG['scaler_y'])
        p_m = TransformerEncoderOnlyModel(25, 3, 128, 8, 9).to(DEVICE); p_m.load_state_dict(torch.load(CONFIG['proposed_path'], map_location=DEVICE))
        l_m = LSTMModel(25, 3, 256, 3, 0.3).to(DEVICE); l_m.load_state_dict(torch.load(CONFIG['lstm_path'], map_location=DEVICE))
        m_m = MLPModel(25, 3, 20, 1024, 0.3).to(DEVICE); m_m.load_state_dict(torch.load(CONFIG['mlp_path'], map_location=DEVICE))
        c_m = CNN1DModel(25, 3, 0.3).to(DEVICE); c_m.load_state_dict(torch.load(CONFIG['cnn_path'], map_location=DEVICE))
    except:
        print("모델 파일 경로를 확인해주세요.")
        sys.exit()

    ITER, sensors_loc_cm = 1000, SENSORS_CM

    def run_full_comparison(steps, type='dist'):
        res = {k: [] for k in model_styles.keys()}
        C_DIST, C_STD, C_DOA = 40000, 7.5, 0.5
        for i, val in enumerate(steps):
            errs_cm = {k: [] for k in res.keys()}
            t_dist  = (val if type == 'dist'     else C_DIST)
            t_td_std = (val if type == 'tdoa_std' else C_STD)
            t_td_m  = (val if type == 'tdoa'     else 0.0)
            t_doa   = (val if type == 'doa'      else C_DOA)
            for _ in range(ITER):
                gt_cm, feat_cm, raw_signals = generate_controlled_traj_cm(t_td_std, t_doa, t_dist, t_td_m)
                for k, m in zip(['Proposed', 'LSTM', 'MLP', 'CNN'], [p_m, l_m, m_m, c_m]):
                    errs_cm[k].append(calculate_rmse(gt_cm, sliding_window_inference_cm(m, sx, sy, feat_cm)))
                ls_init = solve_ls_localization(feat_cm[0, 2:9], sensors_loc_cm)
                kf = KalmanFilter(ls_init)
                kf_t = [ls_init.copy()]  # t=0은 초기 LS 값을 그대로 사용
                for t in range(1, 200):
                    ls_pos = solve_ls_localization(feat_cm[t, 2:9], sensors_loc_cm)
                    kf_t.append(kf.predict_and_update(ls_pos))
                errs_cm['KF'].append(calculate_rmse(gt_cm, np.array(kf_t)))
                p_mus = []
                for t in range(200):
                    m_vec = music_doa_estimation_stable(sensors_loc_cm, raw_signals[t])
                    p_mus.append(localize_music(sensors_loc_cm, m_vec, feat_cm[t]))
                errs_cm['MUSIC'].append(calculate_rmse(gt_cm, np.array(p_mus)))
            for k in res.keys(): res[k].append(np.mean(errs_cm[k]) / 100.0)
            sys.stdout.write(f'\r{type} 분석 중... ({i+1}/{len(steps)})'); sys.stdout.flush()
        return res

    # 분석 스텝 설정
    dist_steps        = np.linspace(0, 60000, 61)
    tdoa_m_steps_cm   = np.linspace(0, 15, 101)
    tdoa_std_steps_cm = np.linspace(0, 15, 101)
    doa_steps         = np.linspace(0, 1.2, 13)

    r_dist     = run_full_comparison(dist_steps,        'dist')
    r_tdoa     = run_full_comparison(tdoa_m_steps_cm,   'tdoa')
    r_tdoa_std = run_full_comparison(tdoa_std_steps_cm, 'tdoa_std')
    r_doa      = run_full_comparison(doa_steps,         'doa')

    # ==============================================================================
    # 4. 요약 결과 출력
    # ==============================================================================
    print(f"\n\n{'='*175}\n [ 종합 RMSE 비교 요약: 거리별(m) ]\n{'='*175}")
    print(f"{'Dist(m)':<10} | {'Prop':<16} | {'MUSIC':<16} | {'LSTM':<16} | {'MLP':<16} | {'KF':<16} | {'1D-CNN'}")
    print(f"{'-'*175}")
    for i in range(len(dist_steps)):
        val = int(round(dist_steps[i]/100.0))
        if val % 10 == 0:
            print(f"{val:>8} | {r_dist['Proposed'][i]:<16.4f} | {r_dist['MUSIC'][i]:<16.4f} | {r_dist['LSTM'][i]:<16.4f} | {r_dist['MLP'][i]:<16.4f} | {r_dist['KF'][i]:<16.4f} | {r_dist['CNN'][i]:.4f}")

    print(f"\n\n{'='*175}\n [ 종합 RMSE 비교 요약: TDOA 평균 바이어스별(us) ]\n{'='*175}")
    print(f"{'Bias(us)':<10} | {'Prop':<16} | {'MUSIC':<16} | {'LSTM':<16} | {'MLP':<16} | {'KF':<16} | {'1D-CNN'}")
    print(f"{'-'*175}")
    for i in range(len(tdoa_m_steps_cm)):
        s_str = f"{round(i*1.0):>8}"
        print(f"{s_str} | {r_tdoa['Proposed'][i]:<16.4f} | {r_tdoa['MUSIC'][i]:<16.4f} | {r_tdoa['LSTM'][i]:<16.4f} | {r_tdoa['MLP'][i]:<16.4f} | {r_tdoa['KF'][i]:<16.4f} | {r_tdoa['CNN'][i]:.4f}")

    print(f"\n\n{'='*175}\n [ 종합 RMSE 비교 요약: TDOA 노이즈 표준편차별(us) ]\n{'='*175}")
    print(f"{'Std(us)':<10} | {'Prop':<16} | {'MUSIC':<16} | {'LSTM':<16} | {'MLP':<16} | {'KF':<16} | {'1D-CNN'}")
    print(f"{'-'*175}")
    for i in range(len(tdoa_std_steps_cm)):
        s_str = f"{round(i*1.0):>8}"
        print(f"{s_str} | {r_tdoa_std['Proposed'][i]:<16.4f} | {r_tdoa_std['MUSIC'][i]:<16.4f} | {r_tdoa_std['LSTM'][i]:<16.4f} | {r_tdoa_std['MLP'][i]:<16.4f} | {r_tdoa_std['KF'][i]:<16.4f} | {r_tdoa_std['CNN'][i]:.4f}")

    print(f"\n\n{'='*175}\n [ 종합 RMSE 비교 요약: DOA 오차별(deg) ]\n{'='*175}")
    print(f"{'DOA(deg)':<10} | {'Prop':<16} | {'LSTM':<16} | {'MLP':<16} | {'KF':<16} | {'1D-CNN'}")
    print(f"{'-'*175}")
    for i in range(len(doa_steps)):
        s_str = f"{doa_steps[i]:>8.1f}"
        print(f"{s_str} | {r_doa['Proposed'][i]:<16.4f} | {r_doa['LSTM'][i]:<16.4f} | {r_doa['MLP'][i]:<16.4f} | {r_doa['KF'][i]:<16.4f} | {r_doa['CNN'][i]:.4f}")

    # ==============================================================================
    # 5. Figure 시각화
    # ==============================================================================
    gt_cm, feat_cm, raw_signals = generate_controlled_traj_cm(7.5, 0.5, target_dist_cm=40000, m_bias_cm=7.5)
    p_all = {k: sliding_window_inference_cm(m, sx, sy, feat_cm)/100.0 for k, m in zip(['Proposed', 'CNN', 'LSTM', 'MLP'], [p_m, c_m, l_m, m_m])}
    gt_m, music_m_list = gt_cm/100.0, []
    for t in range(200):
        m_v = music_doa_estimation_stable(sensors_loc_cm, raw_signals[t])
        music_m_list.append(localize_music(sensors_loc_cm, m_v, feat_cm[t]) / 100.0)
    p_all['MUSIC'] = np.array(music_m_list)
    ls_init_vis = solve_ls_localization(feat_cm[0, 2:9], sensors_loc_cm)
    kf_vis = KalmanFilter(ls_init_vis)
    kf_vis_traj = [ls_init_vis.copy() / 100.0]  # t=0은 초기 LS 값을 그대로 사용
    for t in range(1, 200):
        ls_pos_vis = solve_ls_localization(feat_cm[t, 2:9], sensors_loc_cm)
        kf_vis_traj.append(kf_vis.predict_and_update(ls_pos_vis) / 100.0)
    p_all['KF'] = np.array(kf_vis_traj)
    viz_tracks = [gt_m] + [p_all[k] for k in ['Proposed', 'MUSIC', 'LSTM', 'MLP', 'KF', 'CNN']]

    # Figure 1, 2, 3: 평면별 추정 결과
    planes = [('X', 'Y', [0, 1], 1), ('X', 'Z', [0, 2], 2), ('Y', 'Z', [1, 2], 3)]
    for n1, n2, dims, fig_n in planes:
        plt.figure(fig_n, figsize=(9, 7))
        plt.plot(gt_m[:, dims[0]], gt_m[:, dims[1]], 'k--', label='Ground Truth', lw=2)
        for k in ['Proposed', 'MUSIC', 'LSTM', 'MLP', 'KF', 'CNN']:
            plt.plot(p_all[k][:, dims[0]], p_all[k][:, dims[1]], label=k, color=model_styles[k]['color'],
                    marker=model_styles[k]['marker'], ls=model_styles[k]['ls'], lw=1.5, markevery=15)
        xlim = get_axis_limits_from_tracks(viz_tracks, dims[0], padding_ratio=0.06, min_padding=10.0)
        ylim = get_axis_limits_from_tracks(viz_tracks, dims[1], padding_ratio=0.06, min_padding=10.0)
        plt.xlim(*xlim)
        plt.ylim(*ylim)
        plt.title(f'{n1}-{n2} Plane Estimation (m)'); plt.grid(True, ls=':', alpha=0.6); plt.legend(); plt.tight_layout()

    # Figure 4: 거리 분석
    plt.figure(4, figsize=(10, 7)); plt.gca().set_xticks(np.arange(0, 601, 100))
    plt.gca().xaxis.grid(True, ls=':', alpha=0.5); plt.gca().yaxis.grid(True, which='both', ls=':', alpha=0.5)
    for k in ['Proposed', 'LSTM', 'MLP', 'KF', 'CNN', 'MUSIC']:
        plt.plot(dist_steps/100.0, r_dist[k], label=('1D-CNN' if k=='CNN' else k),
                color=model_styles[k]['color'], marker=model_styles[k]['marker'], ls=model_styles[k]['ls'], lw=1.5, markevery=10)
    plt.yscale('log'); plt.ylim(0.1, 100); plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    plt.title("Figure 4: Distance Error Analysis"); plt.xlabel("Distance (m)"); plt.ylabel("RMSE (m)"); plt.legend(); plt.tight_layout()

    # Figure 5: TDOA 바이어스 분석
    plt.figure(5, figsize=(10, 7)); td_us = (tdoa_m_steps_cm / SOUND_SPEED_CM_S) * 1000000
    plt.gca().set_xticks(np.arange(0, 101, 10))
    plt.gca().xaxis.grid(True, ls=':', alpha=0.5); plt.gca().yaxis.grid(True, which='both', ls=':', alpha=0.5)
    for k in model_styles.keys():
        plt.plot(td_us, r_tdoa[k], label=('1D-CNN' if k=='CNN' else k), color=model_styles[k]['color'],
                marker=model_styles[k]['marker'], ls=model_styles[k]['ls'], lw=1.5, markevery=10)
    plt.yscale('log'); plt.ylim(0.1, 100); plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    plt.title("TDOA Synchronization Error Analysis"); plt.xlabel(r"TDOA Bias ($\mu s$)"); plt.ylabel("RMSE (m)"); plt.legend(); plt.tight_layout()

    # Figure 6: DOA 검증 (MUSIC 제외)
    plt.figure(6, figsize=(10, 7)); plt.gca().set_xticks(doa_steps)
    plt.gca().xaxis.grid(True, ls=':', alpha=0.5); plt.gca().yaxis.grid(True, which='both', ls=':', alpha=0.5)
    for k in model_styles.keys():
        if k == 'MUSIC': continue
        plt.plot(doa_steps, r_doa[k], label=('1D-CNN' if k=='CNN' else k), color=model_styles[k]['color'],
                marker=model_styles[k]['marker'], ls=model_styles[k]['ls'], lw=1.5, markevery=1)
    plt.yscale('log'); plt.ylim(0.1, 100); plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    plt.title("DOA Validation"); plt.xlabel("DOA Deviation (deg)"); plt.ylabel("RMSE (m)"); plt.legend(); plt.tight_layout()

    # Figure 7: 3D 궤적
    fig7 = plt.figure(7, figsize=(10, 8)); ax7 = fig7.add_subplot(111, projection='3d')
    ax7.plot(gt_m[:, 0], gt_m[:, 1], gt_m[:, 2], 'k--', label='Ground Truth', lw=2)
    for k in ['Proposed', 'LSTM', 'MLP', 'KF', 'CNN', 'MUSIC']:
        ax7.plot(p_all[k][:, 0], p_all[k][:, 1], p_all[k][:, 2], label=k, color=model_styles[k]['color'],
                marker=model_styles[k]['marker'], ls=model_styles[k]['ls'], lw=1.5, markevery=20)
    ax7.set_xlim(*get_axis_limits_from_tracks(viz_tracks, 0, padding_ratio=0.06, min_padding=10.0))
    ax7.set_ylim(*get_axis_limits_from_tracks(viz_tracks, 1, padding_ratio=0.06, min_padding=10.0))
    ax7.set_zlim(*get_axis_limits_from_tracks(viz_tracks, 2, padding_ratio=0.06, min_padding=5.0))
    ax7.set_title('Figure 7: 3D Trajectory'); ax7.legend(); plt.tight_layout()

    # Figure 8: 거리 100-300m 상세
    plt.figure(8, figsize=(10, 7)); mask = (dist_steps >= 10000) & (dist_steps <= 30000); steps_sub = dist_steps[mask]/100.0
    plt.gca().set_xticks(np.arange(100, 301, 20)); plt.gca().xaxis.grid(True, ls=':', alpha=0.5); plt.gca().yaxis.grid(True, which='both', ls=':', alpha=0.5)
    for k in ['Proposed', 'LSTM', 'MLP', 'KF', 'CNN', 'MUSIC']:
        plt.plot(steps_sub, np.array(r_dist[k])[mask], label=('1D-CNN' if k=='CNN' else k),
                color=model_styles[k]['color'], marker=model_styles[k]['marker'], ls=model_styles[k]['ls'], lw=2.0, markevery=2)
    plt.yscale('log'); plt.ylim(0.1, 100); plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    plt.title("Figure 8: Distance Error (100~300m)"); plt.xlabel("Distance (m)"); plt.ylabel("RMSE (m)"); plt.legend(); plt.tight_layout()

    # Figure 9: TDOA 노이즈 표준편차 분석
    plt.figure(9, figsize=(10, 7)); td_std_us = (tdoa_std_steps_cm / SOUND_SPEED_CM_S) * 1000000
    plt.gca().set_xticks(np.arange(0, 101, 10))
    plt.gca().xaxis.grid(True, ls=':', alpha=0.5); plt.gca().yaxis.grid(True, which='both', ls=':', alpha=0.5)
    for k in model_styles.keys():
        plt.plot(td_std_us, r_tdoa_std[k], label=('1D-CNN' if k=='CNN' else k), color=model_styles[k]['color'],
                marker=model_styles[k]['marker'], ls=model_styles[k]['ls'], lw=1.5, markevery=10)
    plt.yscale('log'); plt.ylim(0.1, 100); plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    plt.title("TDOA Noise Std Analysis"); plt.xlabel(r"TDOA Noise Std ($\mu s$)"); plt.ylabel("RMSE (m)"); plt.legend(); plt.tight_layout()

    plt.show()
