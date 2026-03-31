import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import random

# Matplotlib 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ==============================================================================
# 1. 설정
# ==============================================================================
NUM_SEQUENCES = 5000000     
SEQUENCE_LENGTH = 20        
INITIAL_POS_RANGE_CM = 80000

STEP_MEAN_CM = 100.0          
DIRECTION_SMOOTHNESS = 0.8  

r = 3.3; L = 7.9
SOUND_SPEED_CM_S = 150000

SENSOR_POSITIONS = np.array([
    [r, 0, 0], [r/np.sqrt(2), r/np.sqrt(2), -L], [0, r, 0],
    [-r/np.sqrt(2), r/np.sqrt(2), -L], [-r, 0, 0],
    [-r/np.sqrt(2), -r/np.sqrt(2), -L], [0, -r, 0],
    [r/np.sqrt(2), -r/np.sqrt(2), -L]
], dtype=np.float32)

TDOA_NOISE_STD_RANGE_CM = (0.0, 15.0)
DOA_NOISE_STD_RANGE_DEG = (0.0, 0.5)

OUTPUT_FILENAME = "dataset_td_0.0-15.0_doa_0.0-0.5_500M.npz"

# ==============================================================================
# 2. 함수 정의
# ==============================================================================
def generate_trajectory(seq_length, initial_pos_range, step_mean, smoothness):
    start_pos = (np.random.rand(3) - 0.5) * initial_pos_range
    trajectory = np.zeros((seq_length, 3), dtype=np.float32)
    trajectory[0] = start_pos
    direction = np.random.randn(3); direction /= np.linalg.norm(direction)
    for i in range(1, seq_length):
        new_random_direction = np.random.randn(3); new_random_direction /= np.linalg.norm(new_random_direction)
        direction = smoothness * direction + (1 - smoothness) * new_random_direction
        direction /= np.linalg.norm(direction)
        step_distance = np.random.normal(loc=step_mean, scale=step_mean / 4)
        step_distance = max(0, step_distance) 
        delta = direction * step_distance
        trajectory[i] = trajectory[i-1] + delta
    return trajectory

def compute_features_from_trajectory(trajectory, sensors, sound_speed, td_std, doa_std):
    seq_length = len(trajectory); num_sensors = len(sensors)
    features_seq = np.zeros((seq_length, 25), dtype=np.float32)
    time_noise_std = td_std / sound_speed
    doa_std_rad = np.radians(doa_std)
    epsilon = 1e-9 

    for i, pos in enumerate(trajectory):
        d = np.linalg.norm(sensors - pos, axis=1) 
        noisy_toa = (d / sound_speed) + np.random.normal(0, time_noise_std, size=d.shape)
        noisy_toa0_cm = noisy_toa[0:1] * sound_speed
        noisy_tdoa_cm = (noisy_toa - noisy_toa[0]) * sound_speed
        delta_pos = pos - sensors
        az = np.arctan2(delta_pos[:, 1], delta_pos[:, 0]) + np.random.normal(0, doa_std_rad, size=d.shape)
        el = np.arctan2(delta_pos[:, 2], np.sqrt(delta_pos[:, 0]**2 + delta_pos[:, 1]**2) + epsilon) + np.random.normal(0, doa_std_rad, size=d.shape)
        features_seq[i] = np.concatenate([noisy_toa0_cm, noisy_tdoa_cm, az, el])
    return features_seq

# ==============================================================================
# 3. 메인 실행 및 시각화
# ==============================================================================
if __name__ == '__main__':
    print(f"범위 기반 노이즈 데이터셋 생성 시작...")
    y_list, x_list = [], []
    
    for _ in tqdm(range(NUM_SEQUENCES), desc="데이터 생성 중"):
        y_seq = generate_trajectory(SEQUENCE_LENGTH, INITIAL_POS_RANGE_CM, STEP_MEAN_CM, DIRECTION_SMOOTHNESS)
        td_std = np.random.uniform(*TDOA_NOISE_STD_RANGE_CM)
        doa_std = np.random.uniform(*DOA_NOISE_STD_RANGE_DEG)
        x_seq = compute_features_from_trajectory(y_seq, SENSOR_POSITIONS, SOUND_SPEED_CM_S, td_std, doa_std)
        y_list.append(y_seq); x_list.append(x_seq)

    Y_data, X_data = np.array(y_list), np.array(x_list)
    np.savez_compressed(OUTPUT_FILENAME, x_data=X_data, y_data=Y_data)
    print(f"\n저장 완료: {OUTPUT_FILENAME}")

    # --- 무작위 샘플 하나 시각화 ---
    print("샘플 궤적 시각화 중...")
    idx = random.randint(0, NUM_SEQUENCES - 1)
    sample_y = Y_data[idx]
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(sample_y[:, 0], sample_y[:, 1], sample_y[:, 2], 'b-o', label='Ground Truth Path')
    ax.scatter(sample_y[0, 0], sample_y[0, 1], sample_y[0, 2], c='g', s=100, label='Start')
    ax.scatter(sample_y[-1, 0], sample_y[-1, 1], sample_y[-1, 2], c='r', s=100, label='End')
    ax.set_title(f"Sample Trajectory #{idx}\n(Range Noise Dataset)")
    ax.set_xlabel('X (cm)'); ax.set_ylabel('Y (cm)'); ax.set_zlabel('Z (cm)')
    ax.legend(); plt.show()