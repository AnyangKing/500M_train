import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import argparse
import importlib.util
from pathlib import Path

import joblib
import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
FINAL_CODE_PATH = ROOT / "코드_최종.py"
DEFAULT_OUTPUT = ROOT / "comparison_results.npz"

MODEL_KEYS = ["Proposed", "LSTM", "MLP", "KF", "CNN", "MUSIC"]


def load_final_code():
    spec = importlib.util.spec_from_file_location("final_code_module", FINAL_CODE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_models(fc):
    config = {
        "proposed_path": "model_td_0.0-15.0_doa_0.0-0.5_500M.pt",
        "lstm_path": "model_lstm_td_0.0-15.0_doa_0.0-0.5_500M.pt",
        "mlp_path": "model_mlp_td_0.0-15.0_doa_0.0-0.5_500M.pt",
        "cnn_path": "model_cnn_td_0.0-15.0_doa_0.0-0.5_500M.pt",
        "scaler_x": "scaler_x_td_0.0-15.0_doa_0.0-0.5_500M.pkl",
        "scaler_y": "scaler_y_td_0.0-15.0_doa_0.0-0.5_500M.pkl",
    }

    sx = joblib.load(ROOT / config["scaler_x"])
    sy = joblib.load(ROOT / config["scaler_y"])

    proposed = fc.TransformerEncoderOnlyModel(25, 3, 128, 8, 9).to(fc.DEVICE)
    proposed.load_state_dict(torch.load(ROOT / config["proposed_path"], map_location=fc.DEVICE))

    lstm = fc.LSTMModel(25, 3, 256, 3, 0.3).to(fc.DEVICE)
    lstm.load_state_dict(torch.load(ROOT / config["lstm_path"], map_location=fc.DEVICE))

    mlp = fc.MLPModel(25, 3, 20, 1024, 0.3).to(fc.DEVICE)
    mlp.load_state_dict(torch.load(ROOT / config["mlp_path"], map_location=fc.DEVICE))

    cnn = fc.CNN1DModel(25, 3, 0.3).to(fc.DEVICE)
    cnn.load_state_dict(torch.load(ROOT / config["cnn_path"], map_location=fc.DEVICE))

    return sx, sy, {
        "Proposed": proposed,
        "LSTM": lstm,
        "MLP": mlp,
        "CNN": cnn,
    }


def run_full_comparison(fc, sx, sy, models, steps, sweep_type, iterations):
    results = {k: [] for k in MODEL_KEYS}
    c_dist, c_std, c_doa = 40000, 7.5, 0.5

    for i, val in enumerate(steps):
        errs_cm = {k: [] for k in MODEL_KEYS}
        t_dist = val if sweep_type == "dist" else c_dist
        t_td_std = val if sweep_type == "tdoa_std" else c_std
        t_td_m = val if sweep_type == "tdoa" else 0.0
        t_doa = val if sweep_type == "doa" else c_doa

        for _ in range(iterations):
            gt_cm, feat_cm, raw_signals = fc.generate_controlled_traj_cm(
                t_td_std, t_doa, t_dist, t_td_m
            )

            for key in ["Proposed", "LSTM", "MLP", "CNN"]:
                pred_cm = fc.sliding_window_inference_cm(models[key], sx, sy, feat_cm)
                errs_cm[key].append(fc.calculate_rmse(gt_cm, pred_cm))

            sensors = fc.SENSORS_CM
            ls_init = fc.compute_initial_kf_state(feat_cm, sensors)
            kf = fc.KalmanFilter(ls_init)
            kf_traj = [ls_init.copy()]
            for t in range(1, 200):
                ls_pos = fc.solve_ls_localization(feat_cm[t, 2:9], sensors)
                kf_traj.append(kf.predict_and_update(ls_pos))
            errs_cm["KF"].append(fc.calculate_rmse(gt_cm, np.array(kf_traj)))

            music_localizer = fc.MusicLocalizer(sensors)
            music_traj = []
            for t in range(200):
                doa_vec = fc.music_doa_estimation_stable(sensors, raw_signals[t])
                music_traj.append(music_localizer.update(doa_vec, feat_cm[t]))
            errs_cm["MUSIC"].append(fc.calculate_rmse(gt_cm, np.array(music_traj)))

        for key in MODEL_KEYS:
            results[key].append(np.mean(errs_cm[key]) / 100.0)

        print(f"\r{sweep_type} 계산 중... ({i + 1}/{len(steps)})", end="", flush=True)

    print()
    return {k: np.asarray(v, dtype=np.float32) for k, v in results.items()}


def build_visualization_bundle(fc, sx, sy, models):
    gt_cm, feat_cm, raw_signals = fc.generate_controlled_traj_cm(
        7.5, 0.5, target_dist_cm=40000, m_bias_cm=7.5
    )

    preds_m = {}
    for key in ["Proposed", "LSTM", "MLP", "CNN"]:
        preds_m[key] = fc.sliding_window_inference_cm(models[key], sx, sy, feat_cm) / 100.0

    music_localizer = fc.MusicLocalizer(fc.SENSORS_CM)
    music_traj = []
    for t in range(200):
        doa_vec = fc.music_doa_estimation_stable(fc.SENSORS_CM, raw_signals[t])
        music_traj.append(music_localizer.update(doa_vec, feat_cm[t]) / 100.0)
    preds_m["MUSIC"] = np.asarray(music_traj, dtype=np.float32)

    ls_init = fc.compute_initial_kf_state(feat_cm, fc.SENSORS_CM)
    kf = fc.KalmanFilter(ls_init)
    kf_traj = [ls_init.copy() / 100.0]
    for t in range(1, 200):
        ls_pos = fc.solve_ls_localization(feat_cm[t, 2:9], fc.SENSORS_CM)
        kf_traj.append(kf.predict_and_update(ls_pos) / 100.0)
    preds_m["KF"] = np.asarray(kf_traj, dtype=np.float32)

    bundle = {
        "viz_gt_m": (gt_cm / 100.0).astype(np.float32),
    }
    for key in MODEL_KEYS:
        bundle[f"viz_{key}_m"] = preds_m[key].astype(np.float32)
    return bundle


def main():
    parser = argparse.ArgumentParser(description="Run comparison metrics without plotting.")
    parser.add_argument("--iter", type=int, default=10000, help="Monte Carlo iterations per sweep point")
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Output .npz path",
    )
    args = parser.parse_args()

    fc = load_final_code()
    sx, sy, models = load_models(fc)

    dist_steps = np.linspace(0, 60000, 61, dtype=np.float32)
    tdoa_m_steps_cm = np.linspace(0, 15, 101, dtype=np.float32)
    tdoa_std_steps_cm = np.linspace(0, 15, 101, dtype=np.float32)
    doa_steps = np.linspace(0, 1.2, 13, dtype=np.float32)

    print(f"ITER={args.iter}")
    r_dist = run_full_comparison(fc, sx, sy, models, dist_steps, "dist", args.iter)
    r_tdoa = run_full_comparison(fc, sx, sy, models, tdoa_m_steps_cm, "tdoa", args.iter)
    r_tdoa_std = run_full_comparison(fc, sx, sy, models, tdoa_std_steps_cm, "tdoa_std", args.iter)
    r_doa = run_full_comparison(fc, sx, sy, models, doa_steps, "doa", args.iter)

    payload = {
        "iter_count": np.array([args.iter], dtype=np.int32),
        "dist_steps": dist_steps,
        "tdoa_m_steps_cm": tdoa_m_steps_cm,
        "tdoa_std_steps_cm": tdoa_std_steps_cm,
        "doa_steps": doa_steps,
    }
    for key in MODEL_KEYS:
        payload[f"r_dist_{key}"] = r_dist[key]
        payload[f"r_tdoa_{key}"] = r_tdoa[key]
        payload[f"r_tdoa_std_{key}"] = r_tdoa_std[key]
        payload[f"r_doa_{key}"] = r_doa[key]
    payload.update(build_visualization_bundle(fc, sx, sy, models))

    output_path = Path(args.output)
    np.savez_compressed(output_path, **payload)
    print(f"저장 완료: {output_path}")


if __name__ == "__main__":
    main()
