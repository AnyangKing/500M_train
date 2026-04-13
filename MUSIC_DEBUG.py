import numpy as np
import torch
import math
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SOUND_SPEED_CM_S = 150000.0
MUSIC_F0 = 32000.0
MUSIC_N_SNAP = 64
MUSIC_AZ_RES = 2
MUSIC_EL_RES = 2
MUSIC_SNR_LINEAR = 31.6

# 탐색 격자
_AZ_NP = np.radians(np.arange(-180, 180, MUSIC_AZ_RES))
_EL_NP = np.radians(np.arange(-90, 90, MUSIC_EL_RES))
_AZ_GRID, _EL_GRID = np.meshgrid(_AZ_NP, _EL_NP)
_AZ_FLAT_NP = _AZ_GRID.ravel().astype(np.float64)
_EL_FLAT_NP = _EL_GRID.ravel().astype(np.float64)
_D_MAT_NP = np.stack([np.cos(_EL_FLAT_NP)*np.cos(_AZ_FLAT_NP),
                       np.cos(_EL_FLAT_NP)*np.sin(_AZ_FLAT_NP),
                       np.sin(_EL_FLAT_NP)], axis=1)
_D_MAT_GPU = torch.tensor(_D_MAT_NP, dtype=torch.float64, device=DEVICE)

r_cm, L_cm = 3.3, 7.9
def get_sensors_cm():
    S2 = np.sqrt(2)
    return np.array([[r_cm, 0, 0], [r_cm/S2, r_cm/S2, -L_cm], [0, r_cm, 0], [-r_cm/S2, r_cm/S2, -L_cm],
                     [-r_cm, 0, 0], [-r_cm/S2, -r_cm/S2, -L_cm], [0, -r_cm, 0], [r_cm/S2, -r_cm/S2, -L_cm]], dtype=np.float32)

def generate_test_traj():
    sensors = get_sensors_cm()
    # 단순 궤적: 0,0,0부터 100,100,100으로 이동
    traj = np.array([[i*100/10, i*100/10, i*100/10] for i in range(11)], dtype=np.float32)

    feats = np.zeros((len(traj), 25), dtype=np.float32)
    raw_signals = np.zeros((len(traj), 8, MUSIC_N_SNAP), dtype=np.complex128)

    td_std = 7.5 / SOUND_SPEED_CM_S
    doa_std = np.radians(0.5)
    noise_std = 1.0 / np.sqrt(2.0 * MUSIC_SNR_LINEAR)

    for i, p in enumerate(traj):
        d = np.linalg.norm(sensors - p, axis=1)
        toa = d / SOUND_SPEED_CM_S + np.random.normal(0, td_std, size=8)
        dp = p - sensors

        feats[i] = np.concatenate([toa[0:1]*SOUND_SPEED_CM_S,
                                   (toa-toa[0])*SOUND_SPEED_CM_S,
                                   np.arctan2(dp[:,1], dp[:,0]) + np.random.normal(0, doa_std, 8),
                                   np.arctan2(dp[:,2], np.sqrt(dp[:,0]**2 + dp[:,1]**2)+1e-9) + np.random.normal(0, doa_std, 8)])

        s = np.exp(1j * np.random.uniform(0, 2 * np.pi, MUSIC_N_SNAP))
        direction = (p - np.mean(sensors, axis=0)) / (np.linalg.norm(p - np.mean(sensors, axis=0)) + 1e-9)
        steering = np.exp(-1j * 2 * np.pi * MUSIC_F0 * (sensors @ direction) / SOUND_SPEED_CM_S)
        X = np.outer(steering, s)
        noise = noise_std * (np.random.randn(8, MUSIC_N_SNAP) + 1j * np.random.randn(8, MUSIC_N_SNAP))
        raw_signals[i] = X + noise

    return traj, feats, raw_signals

def music_doa_estimation_debug(sensors, raw_signals_t):
    n_signals = 1
    X = torch.tensor(raw_signals_t, dtype=torch.complex128, device=DEVICE)
    R = (X @ X.conj().T) / MUSIC_N_SNAP
    eigenvalues, eigenvectors = torch.linalg.eigh(R)
    idx = torch.argsort(eigenvalues, descending=True)
    eigenvectors = eigenvectors[:, idx]
    En = eigenvectors[:, n_signals:]

    sensors_m = torch.tensor(sensors.astype(np.float64) / 100.0, dtype=torch.float64, device=DEVICE)
    lam = (SOUND_SPEED_CM_S / 100.0) / MUSIC_F0
    proj = _D_MAT_GPU @ sensors_m.T
    A = torch.exp(-1j * 2 * math.pi * proj / lam).to(torch.complex128)
    norms = torch.linalg.norm(A, dim=1, keepdim=True)
    A = A / (norms + 1e-12)
    En_A = A @ En
    denom = torch.sum(torch.real(En_A * En_A.conj()), dim=1)
    power = 1.0 / (denom + 1e-12)

    best_idx = int(torch.argmax(power).cpu())
    best_az = _AZ_FLAT_NP[best_idx]
    best_el = _EL_FLAT_NP[best_idx]
    est_vec = np.array([np.cos(best_el) * np.cos(best_az),
                        np.cos(best_el) * np.sin(best_az),
                        np.sin(best_el)], dtype=np.float64)

    return est_vec / (np.linalg.norm(est_vec) + 1e-9), power.cpu().numpy()

def calc_true_doa(target_pos, sensors):
    """타겟에서 센서 어레이 중심으로의 방향 (true DOA)"""
    sensor_center = np.mean(sensors, axis=0)
    direction = target_pos - sensor_center
    return direction / (np.linalg.norm(direction) + 1e-9)

def localize_music(sensors, estimated_doa, feat_t):
    d0 = feat_t[0]
    d_all = np.concatenate([[d0], d0 + feat_t[2:9]])
    est_dist_cm = np.mean(d_all)
    return np.mean(sensors, axis=0) + estimated_doa * est_dist_cm

# ==== 테스트 실행 ====
print("="*100)
print("MUSIC DEBUG: DOA 추정 정확도 검증")
print("="*100)

sensors = get_sensors_cm()
traj, feats, raw_signals = generate_test_traj()

print(f"\n{'idx':<5} | {'Dist(cm)':<12} | {'True_DOA':<30} | {'Est_DOA':<30} | {'DOA_Error':<12} | {'pos_error(cm)':<12}")
print("-"*115)

for i in range(len(traj)):
    # True DOA
    true_doa = calc_true_doa(traj[i], sensors)

    # Estimated DOA
    est_doa, power = music_doa_estimation_debug(sensors, raw_signals[i])

    # MUSIC 방향 ambiguity 해결: 반대 방향이면 부호 반전
    if np.dot(true_doa, est_doa) < 0:
        est_doa = -est_doa

    # DOA error
    doa_error = np.arccos(np.clip(np.dot(true_doa, est_doa), -1, 1))

    # Distance
    true_dist = np.linalg.norm(traj[i] - np.mean(sensors, axis=0))

    # Position estimation
    est_pos = localize_music(sensors, est_doa, feats[i])
    pos_error = np.linalg.norm(traj[i] - est_pos)

    print(f"{i:<5} | {true_dist:<12.1f} | {str(true_doa):<30} | {str(est_doa):<30} | {np.degrees(doa_error):<12.2f}° | {pos_error:<12.1f}")

print("\n" + "="*100)
print("분석:")
print("- DOA_Error가 크면: MUSIC 알고리즘 자체 문제")
print("- DOA_Error는 작지만 pos_error가 크면: localize_music 함수 문제")
print("="*100)
