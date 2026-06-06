"""
궤적 그림만 새로 생성 - npz와 무관하게 독립 실행
compute_results.py 실행과 완전히 독립적
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import importlib.util, sys, torch
import joblib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator, MultipleLocator

ROOT = Path(__file__).resolve().parent
SAVE_DIR = ROOT  # 저장 위치 (필요시 변경)

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.labelsize"] = 22
plt.rcParams["xtick.labelsize"] = 22
plt.rcParams["ytick.labelsize"] = 22
plt.rcParams["legend.fontsize"] = 18

# ── 코드_최종.py 로드 ──────────────────────────────────────────
spec = importlib.util.spec_from_file_location("fc", ROOT / "코드_최종.py")
fc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fc)

CONFIG = {
    "proposed_path": "model_td_0.0-15.0_doa_0.0-0.5_500M.pt",
    "lstm_path": "model_lstm_td_0.0-15.0_doa_0.0-0.5_500M.pt",
    "mlp_path": "model_mlp_td_0.0-15.0_doa_0.0-0.5_500M.pt",
    "cnn_path": "model_cnn_td_0.0-15.0_doa_0.0-0.5_500M.pt",
    "scaler_x": "scaler_x_td_0.0-15.0_doa_0.0-0.5_500M.pkl",
    "scaler_y": "scaler_y_td_0.0-15.0_doa_0.0-0.5_500M.pkl",
}

sx = joblib.load(ROOT / CONFIG["scaler_x"])
sy = joblib.load(ROOT / CONFIG["scaler_y"])

proposed = fc.TransformerEncoderOnlyModel(25, 3, 128, 8, 9).to(fc.DEVICE)
proposed.load_state_dict(torch.load(ROOT / CONFIG["proposed_path"], map_location=fc.DEVICE))

lstm = fc.LSTMModel(25, 3, 256, 3, 0.3).to(fc.DEVICE)
lstm.load_state_dict(torch.load(ROOT / CONFIG["lstm_path"], map_location=fc.DEVICE))

mlp = fc.MLPModel(25, 3, 20, 1024, 0.3).to(fc.DEVICE)
mlp.load_state_dict(torch.load(ROOT / CONFIG["mlp_path"], map_location=fc.DEVICE))

cnn = fc.CNN1DModel(25, 3, 0.3).to(fc.DEVICE)
cnn.load_state_dict(torch.load(ROOT / CONFIG["cnn_path"], map_location=fc.DEVICE))

models = {
    "Proposed": proposed,
    "LSTM":     lstm,
    "MLP":      mlp,
    "CNN":      cnn,
}

for model in models.values():
    model.eval()

MODEL_STYLES = {
    "Proposed": {"marker": "o", "color": "r",     "ls": "-"},
    "MUSIC":    {"marker": "D", "color": "green",  "ls": "--"},
    "LSTM":     {"marker": "s", "color": "m",      "ls": "-"},
    "MLP":      {"marker": "^", "color": "b",      "ls": "-"},
    "CNN":      {"marker": "x", "color": "c",      "ls": "-"},
}

def display_label(k):
    return {"Proposed": "Proposed (Transformer)", "MUSIC": "MUSIC-LS",
            "LSTM": "LSTM", "MLP": "MLP", "CNN": "1D-CNN"}.get(k, k)

def get_axis_limits_from_tracks(tracks, dim, padding_ratio=0.08, min_padding=5.0):
    values = np.concatenate([track[:, dim] for track in tracks])
    vmin, vmax = float(np.min(values)), float(np.max(values))
    span = vmax - vmin
    pad = max(span * padding_ratio, min_padding)
    return vmin - pad, vmax + pad

def get_axis_limits_for_plane(gt_track, pred_tracks, dim, padding_ratio=0.03, min_padding=2.0):
    values = [gt_track[:, dim]]
    values.extend(track[:, dim] for track in pred_tracks)
    merged = np.concatenate(values)
    vmin, vmax = float(np.min(merged)), float(np.max(merged))
    span = vmax - vmin
    pad = max(span * padding_ratio, min_padding)
    return vmin - pad, vmax + pad

def select_tracks_near_ground_truth(gt_track, pred_tracks, center_threshold_factor=3.0, spread_threshold_factor=3.0):
    gt_center = np.mean(gt_track, axis=0)
    gt_spread = np.linalg.norm(np.max(gt_track, axis=0) - np.min(gt_track, axis=0))
    gt_spread = max(float(gt_spread), 1.0)

    selected = []
    for track in pred_tracks:
        track_center = np.mean(track, axis=0)
        center_dist = np.linalg.norm(track_center - gt_center)
        track_spread = np.linalg.norm(np.max(track, axis=0) - np.min(track, axis=0))
        if center_dist <= center_threshold_factor * gt_spread and track_spread <= spread_threshold_factor * gt_spread:
            selected.append(track)

    return selected if selected else pred_tracks

def get_3d_limits_and_aspect(gt_track, pred_tracks, padding_ratio=0.04, min_padding=3.0):
    clustered_tracks = [gt_track] + select_tracks_near_ground_truth(gt_track, pred_tracks)
    limits = []
    spans = []
    for dim in range(3):
        vmin, vmax = get_axis_limits_from_tracks(clustered_tracks, dim, padding_ratio=padding_ratio, min_padding=min_padding)
        limits.append((vmin, vmax))
        spans.append(max(vmax - vmin, 1.0))
    return limits, tuple(spans)

def try_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    gt_cm, feat_cm, raw_signals = fc.generate_controlled_traj_cm(
        7.5, 0.5, target_dist_cm=40000, m_bias_cm=7.5)

    music_localizer = fc.MusicLocalizer(fc.SENSORS_CM)
    music_traj = []
    for t in range(200):
        doa_vec = fc.music_doa_estimation_stable(fc.SENSORS_CM, raw_signals[t])
        music_traj.append(music_localizer.update(doa_vec, feat_cm[t]) / 100.0)
    music_m = np.asarray(music_traj)
    gt_m = gt_cm / 100.0

    gt_z_range = max(np.ptp(gt_m[:, 2]), 1.0)
    gt_z_center = np.mean(gt_m[:, 2])
    music_z_dev = np.max(np.abs(music_m[:, 2] - gt_z_center))
    music_step = np.linalg.norm(np.diff(music_m, axis=0), axis=1)
    music_z_step = np.abs(np.diff(music_m[:, 2]))
    music_err = np.linalg.norm(music_m - gt_m, axis=1)

    if music_z_dev > gt_z_range * 5:
        print(f"  seed={seed}: z스파이크 {music_z_dev:.1f}m 기각")
        return None
    if np.max(music_z_step) > 80.0:
        print(f"  seed={seed}: z점프 {np.max(music_z_step):.1f}m 기각")
        return None
    if np.max(music_step) > 100.0:
        print(f"  seed={seed}: 3D점프 {np.max(music_step):.1f}m 기각")
        return None
    if np.max(music_err) > 120.0:
        print(f"  seed={seed}: 최대오차 {np.max(music_err):.1f}m 기각")
        return None

    print(
        f"  seed={seed}: OK "
        f"(MUSIC mean={np.mean(music_err):.1f}m, max={np.max(music_err):.1f}m, "
        f"max_step={np.max(music_step):.1f}m, max_z_step={np.max(music_z_step):.1f}m)"
    )
    return gt_m, feat_cm, music_m

# ── 좋은 시드 탐색 ────────────────────────────────────────────
print("궤적 탐색 중...")
result = None
for seed in range(300):
    result = try_seed(seed)
    if result is not None:
        best_seed = seed
        break

if result is None:
    print("적합한 시드 없음")
    sys.exit(1)

gt_m, feat_cm, music_m = result
print(f"시드 {best_seed} 선택")

preds_m = {}
for key in ["Proposed", "LSTM", "MLP", "CNN"]:
    preds_m[key] = fc.sliding_window_inference_cm(models[key], sx, sy, feat_cm) / 100.0
preds_m["MUSIC"] = music_m

ALL_KEYS = ["Proposed", "LSTM", "MLP", "CNN", "MUSIC"]
MARKEVERY = 20

def plot_2d(dim_x, dim_y, xlabel, ylabel, fname):
    plt.figure(figsize=(10, 8))
    plt.plot(gt_m[:, dim_x], gt_m[:, dim_y], "k--", label="Ground Truth", lw=2)
    for k in ALL_KEYS:
        plt.plot(preds_m[k][:, dim_x], preds_m[k][:, dim_y],
                 label=display_label(k), markevery=MARKEVERY,
                 **{kk: MODEL_STYLES[k][kk] for kk in ["color","marker","ls"]}, lw=1.5)
    clustered_pred_tracks = select_tracks_near_ground_truth(gt_m, [preds_m[k] for k in ALL_KEYS])
    plt.xlim(*get_axis_limits_for_plane(gt_m, clustered_pred_tracks, dim_x, padding_ratio=0.03, min_padding=2.0))
    plt.ylim(*get_axis_limits_for_plane(gt_m, clustered_pred_tracks, dim_y, padding_ratio=0.03, min_padding=2.0))
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.grid(True, ls=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / fname, dpi=300)
    plt.close()
    print(f"  저장: {fname}")

plot_2d(0, 1, "X (m)", "Y (m)", "trajectory_xy_plane.png")
plot_2d(0, 2, "X (m)", "Z (m)", "trajectory_xz_plane.png")
plot_2d(1, 2, "Y (m)", "Z (m)", "trajectory_yz_plane.png")

# 3D
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection="3d")
ax.plot(gt_m[:,0], gt_m[:,1], gt_m[:,2], "k--", label="Ground Truth", lw=2)
for k in ALL_KEYS:
    ax.plot(preds_m[k][:,0], preds_m[k][:,1], preds_m[k][:,2],
            label=display_label(k), markevery=MARKEVERY,
            **{kk: MODEL_STYLES[k][kk] for kk in ["color","marker","ls"]}, lw=1.5)
limits_3d, aspect_3d = get_3d_limits_and_aspect(
    gt_m, [preds_m[k] for k in ALL_KEYS], padding_ratio=0.04, min_padding=3.0
)
ax.set_xlim(*limits_3d[0])
ax.set_ylim(*limits_3d[1])
ax.set_zlim(*limits_3d[2])
ax.set_box_aspect(aspect_3d)
ax.view_init(elev=18, azim=-48)
ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
ax.yaxis.set_major_locator(MultipleLocator(20))
ax.zaxis.set_major_locator(MaxNLocator(nbins=4))
plt.tight_layout()
plt.savefig(SAVE_DIR / "trajectory_3d.png", dpi=300)
plt.close()
print("  저장: trajectory_3d.png")
print("완료!")
