import numpy as np
import math
import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize

# Matplotlib 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

INPUT_DIM, OUTPUT_DIM = 25, 3
SOUND_SPEED_CM_S = 150000.0

# ==============================================================================
# MUSIC 설정
# ==============================================================================
MUSIC_F0 = 32000.0
MUSIC_N_SNAP = 64
MUSIC_AZ_RES = 2
MUSIC_EL_RES = 2
MUSIC_SNR_LINEAR = 316.0  # 코드_최종.py와 동일한 SNR=25dB 조건

# 탐색 격자 사전 계산
_AZ_NP = np.radians(np.arange(-180, 180, MUSIC_AZ_RES))
_EL_NP = np.radians(np.arange(-90,  90,  MUSIC_EL_RES))
_AZ_GRID, _EL_GRID = np.meshgrid(_AZ_NP, _EL_NP)
_AZ_FLAT_NP = _AZ_GRID.ravel().astype(np.float64)
_EL_FLAT_NP = _EL_GRID.ravel().astype(np.float64)
_D_MAT_NP = np.stack([np.cos(_EL_FLAT_NP)*np.cos(_AZ_FLAT_NP),
                       np.cos(_EL_FLAT_NP)*np.sin(_AZ_FLAT_NP),
                       np.sin(_EL_FLAT_NP)], axis=1)
_D_MAT_GPU = _D_MAT_NP

r_cm, L_cm = 3.3, 7.9

def get_sensors_cm():
    S2 = np.sqrt(2)
    return np.array([[r_cm, 0, 0], [r_cm/S2, r_cm/S2, -L_cm], [0, r_cm, 0], [-r_cm/S2, r_cm/S2, -L_cm],
                     [-r_cm, 0, 0], [-r_cm/S2, -r_cm/S2, -L_cm], [0, -r_cm, 0], [r_cm/S2, -r_cm/S2, -L_cm]], dtype=np.float32)

def generate_controlled_traj_cm(td_noise_cm, doa_noise_deg, target_dist_cm=None, m_bias_cm=0.0):
    sensors = get_sensors_cm()
    traj = np.zeros((200, 3), dtype=np.float32)
    direction = np.random.randn(3)
    direction /= (np.linalg.norm(direction) + 1e-9)
    traj[0] = direction * target_dist_cm
    vec = np.random.randn(3)
    vec /= (np.linalg.norm(vec) + 1e-9)

    for i in range(1, 200):
        rv = np.random.randn(3)
        rv /= (np.linalg.norm(rv) + 1e-9)
        vec = 0.8*vec + 0.2*rv
        vec /= (np.linalg.norm(vec)+1e-9)
        traj[i] = traj[i-1] + vec*100.0

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
        steering = np.exp(-1j * 2 * np.pi * MUSIC_F0 / SOUND_SPEED_CM_S * (sensors @ direction))
        X = np.outer(steering, s)
        noise = noise_std * (np.random.randn(8, MUSIC_N_SNAP) + 1j * np.random.randn(8, MUSIC_N_SNAP))
        raw_signals[i] = X + noise

    return traj, feats, raw_signals

def music_doa_estimation_stable(sensors, raw_signals_t):
    n_signals = 1
    X = raw_signals_t
    R = (X @ X.conj().T) / MUSIC_N_SNAP
    eigenvalues, eigenvectors = np.linalg.eigh(R)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    En = eigenvectors[:, n_signals:]
    sensors_m = sensors.astype(np.float64) / 100.0
    lam = (SOUND_SPEED_CM_S / 100.0) / MUSIC_F0
    proj = _D_MAT_GPU @ sensors_m.T
    A = np.exp(-1j * 2 * math.pi * proj / lam)
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    A = A / (norms + 1e-12)
    En_A = A @ En
    denom = np.sum(np.real(En_A * En_A.conj()), axis=1)
    power = 1.0 / (denom + 1e-12)
    best_idx = int(np.argmax(power))
    best_az = _AZ_FLAT_NP[best_idx]
    best_el = _EL_FLAT_NP[best_idx]
    est_vec = np.array([np.cos(best_el) * np.cos(best_az),
                        np.cos(best_el) * np.sin(best_az),
                        np.sin(best_el)], dtype=np.float64)
    return est_vec / (np.linalg.norm(est_vec) + 1e-9)

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

def compute_tdoa_from_pos_cm(pos, sensors):
    toa = np.linalg.norm(sensors - pos, axis=1) / SOUND_SPEED_CM_S
    return (toa[1:] - toa[0]) * SOUND_SPEED_CM_S

def localize_music_v0(sensors, estimated_doa, feat_t):
    """v0: 현재 방식 (기준)"""
    pos_ls = solve_ls_localization(feat_t[2:9], sensors)
    sensor_center = np.mean(sensors, axis=0)
    pos1 = pos_ls
    direction_to_ls = pos_ls - sensor_center
    pos2 = sensor_center - direction_to_ls
    dir1 = (pos1 - sensor_center) / (np.linalg.norm(pos1 - sensor_center) + 1e-9)
    dir2 = (pos2 - sensor_center) / (np.linalg.norm(pos2 - sensor_center) + 1e-9)
    match1 = np.dot(dir1, estimated_doa)
    match2 = np.dot(dir2, estimated_doa)
    return pos1 if match1 > match2 else pos2

def localize_music_v1(sensors, estimated_doa, feat_t):
    """v1: Option 1 - DOA 거리 직접 탐색"""
    sensor_center = np.mean(sensors, axis=0)
    best_pos = sensor_center
    best_error = float('inf')
    for r in np.linspace(0, 60000, 121):
        pos_candidate = sensor_center + estimated_doa * r
        tdoa = compute_tdoa_from_pos_cm(pos_candidate, sensors)
        error = np.mean((tdoa - feat_t[2:9])**2)
        if error < best_error:
            best_error = error
            best_pos = pos_candidate
    return best_pos

def localize_music_v2(sensors, estimated_doa, feat_t):
    """v2: Option 2 - DOA+TDOA 결합 최적화"""
    sensor_center = np.mean(sensors, axis=0)
    tdoa_target = feat_t[2:9]
    expected_radius = max(float(feat_t[0]), 1.0)
    def cost_function(pos):
        tdoa = compute_tdoa_from_pos_cm(pos, sensors)
        tdoa_error = np.mean((tdoa - tdoa_target)**2)
        direction = (pos - sensor_center) / (np.linalg.norm(pos - sensor_center) + 1e-9)
        doa_error = (1 - np.clip(np.dot(direction, estimated_doa), -1.0, 1.0)) ** 2
        radius_error = ((np.linalg.norm(pos - sensor_center) - expected_radius) / expected_radius) ** 2
        return tdoa_error + 2000.0 * doa_error + 10.0 * radius_error
    ls_init = solve_ls_localization(feat_t[2:9], sensors)
    mirrored_init = sensor_center - (ls_init - sensor_center)
    doa_line_init = localize_music_v1(sensors, estimated_doa, feat_t)
    candidate_inits = [ls_init, mirrored_init, doa_line_init, sensor_center + estimated_doa * expected_radius]
    best_pos = ls_init
    best_cost = cost_function(ls_init)
    try:
        for init in candidate_inits:
            result = minimize(cost_function, x0=init, method='Powell',
                              options={'maxiter': 200, 'xtol': 1e-3, 'ftol': 1e-3, 'disp': False})
            candidate_pos = result.x if result.success else init
            candidate_cost = cost_function(candidate_pos)
            if candidate_cost < best_cost:
                best_cost = candidate_cost
                best_pos = candidate_pos
        return best_pos
    except:
        return best_pos

def calculate_rmse(gt, pred, dims=[0, 1, 2]):
    return np.sqrt(np.mean(np.sum((gt[:, dims] - pred[:, dims])**2, axis=1)))

# ==============================================================================
# 메인: MUSIC 3개 버전 비교
# ==============================================================================
if __name__ == '__main__':
    np.random.seed(42)

    model_styles = {
        'GT': {'marker': '--', 'color': 'black', 'ls': '--', 'linewidth': 2},
        'MUSIC_v0': {'marker': 'D', 'color': 'green', 'ls': '-'},
        'MUSIC_v1': {'marker': 's', 'color': 'darkgreen', 'ls': '-.'},
        'MUSIC_v2': {'marker': '^', 'color': 'lightgreen', 'ls': ':'}
    }

    # 최종 반영 판단용 설정
    ITER = 40
    FAST_MODE = False
    sensors_loc_cm = get_sensors_cm()

    def run_music_comparison(steps, type='dist', fast=True):
        res = {k: [] for k in ['MUSIC_v0', 'MUSIC_v1', 'MUSIC_v2']}
        C_DIST, C_STD, C_DOA = 40000, 7.5, 0.5
        steps = steps[::5] if fast else steps  # fast 모드: 5개마다 샘플

        for i, val in enumerate(steps):
            errs_cm = {k: [] for k in res.keys()}
            t_dist  = (val if type == 'dist'     else C_DIST)
            t_td_std = (val if type == 'tdoa_std' else C_STD)
            t_td_m  = (val if type == 'tdoa'     else 0.0)
            t_doa   = (val if type == 'doa'      else C_DOA)

            for _ in range(ITER):
                gt_cm, feat_cm, raw_signals = generate_controlled_traj_cm(t_td_std, t_doa, t_dist, t_td_m)

                # MUSIC DOA 한 번만 계산, 3개 버전에 공용
                p_mus = {'v0': [], 'v1': [], 'v2': []}
                for t in range(200):
                    m_vec = music_doa_estimation_stable(sensors_loc_cm, raw_signals[t])
                    p_mus['v0'].append(localize_music_v0(sensors_loc_cm, m_vec, feat_cm[t]))
                    p_mus['v1'].append(localize_music_v1(sensors_loc_cm, m_vec, feat_cm[t]))
                    p_mus['v2'].append(localize_music_v2(sensors_loc_cm, m_vec, feat_cm[t]))

                for version in ['v0', 'v1', 'v2']:
                    errs_cm[f'MUSIC_{version}'].append(calculate_rmse(gt_cm, np.array(p_mus[version])))

            for k in res.keys():
                res[k].append(np.mean(errs_cm[k]) / 100.0)
            sys.stdout.write(f'\r{type} 분석... ({i+1}/{len(steps)})'); sys.stdout.flush()

        return res

    print("MUSIC 3개 버전 비교 시작")
    dist_steps = np.linspace(0, 60000, 61)
    tdoa_std_steps = np.linspace(0, 30, 31)
    doa_steps = np.linspace(0, 5, 26)

    r_dist = run_music_comparison(dist_steps, 'dist', fast=FAST_MODE)
    r_tdoa_std = run_music_comparison(tdoa_std_steps, 'tdoa_std', fast=FAST_MODE)
    r_doa = run_music_comparison(doa_steps, 'doa', fast=FAST_MODE)

    # 실제 사용된 스텝 (fast 모드에서 5개마다)
    dist_steps_plot = dist_steps[::5] if FAST_MODE else dist_steps
    tdoa_std_steps_plot = tdoa_std_steps[::5] if FAST_MODE else tdoa_std_steps
    doa_steps_plot = doa_steps[::5] if FAST_MODE else doa_steps

    # 그래프 1: 거리별 오차 (개선)
    fig, ax = plt.subplots(figsize=(11, 7))
    for k in ['MUSIC_v0', 'MUSIC_v1', 'MUSIC_v2']:
        ax.plot(dist_steps_plot/100.0, r_dist[k], label=k,
                color=model_styles[k]['color'], marker=model_styles[k]['marker'],
                ls=model_styles[k]['ls'], lw=2, markevery=6, markersize=6)

    ax.set_yscale('log')
    dist_data = [val for v in r_dist.values() for val in v]
    dist_min, dist_max = min(dist_data), max(dist_data)
    ax.set_ylim(dist_min * 0.5, dist_max * 2)
    ax.set_xlim(0, 600)
    ax.grid(True, which='both', ls=':', alpha=0.4)
    ax.set_xlabel("Distance (m)", fontsize=12)
    ax.set_ylabel("RMSE (m)", fontsize=12)
    ax.set_title("MUSIC 3개 버전 비교: 거리별 오차", fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    plt.tight_layout()
    plt.show()

    # 그래프 2: TDOA 표준편차별 오차
    fig, ax = plt.subplots(figsize=(11, 7))
    for k in ['MUSIC_v0', 'MUSIC_v1', 'MUSIC_v2']:
        ax.plot(tdoa_std_steps_plot, r_tdoa_std[k], label=k,
                color=model_styles[k]['color'], marker=model_styles[k]['marker'],
                ls=model_styles[k]['ls'], lw=2, markevery=3, markersize=6)

    ax.set_yscale('log')
    tdoa_data = [val for v in r_tdoa_std.values() for val in v]
    tdoa_min, tdoa_max = min(tdoa_data), max(tdoa_data)
    ax.set_ylim(tdoa_min * 0.5, tdoa_max * 2)
    ax.grid(True, which='both', ls=':', alpha=0.4)
    ax.set_xlabel("TDOA Noise Std (cm)", fontsize=12)
    ax.set_ylabel("RMSE (m)", fontsize=12)
    ax.set_title("MUSIC 3개 버전 비교: TDOA 잡음별 오차", fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    plt.tight_layout()
    plt.show()

    # 그래프 3: DOA 잡음별 오차
    fig, ax = plt.subplots(figsize=(11, 7))
    for k in ['MUSIC_v0', 'MUSIC_v1', 'MUSIC_v2']:
        ax.plot(doa_steps_plot, r_doa[k], label=k,
                color=model_styles[k]['color'], marker=model_styles[k]['marker'],
                ls=model_styles[k]['ls'], lw=2, markevery=3, markersize=6)

    ax.set_yscale('log')
    doa_data = [val for v in r_doa.values() for val in v]
    doa_min, doa_max = min(doa_data), max(doa_data)
    ax.set_ylim(doa_min * 0.5, doa_max * 2)
    ax.grid(True, which='both', ls=':', alpha=0.4)
    ax.set_xlabel("DOA Noise (deg)", fontsize=12)
    ax.set_ylabel("RMSE (m)", fontsize=12)
    ax.set_title("MUSIC 3개 버전 비교: DOA 잡음별 오차", fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    plt.tight_layout()
    plt.show()

    print("\n분석 완료!")
