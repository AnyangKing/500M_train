import argparse
import importlib.util
import os
from pathlib import Path

import joblib
import numpy as np
import torch


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

ROOT = Path(__file__).resolve().parent
FINAL_CODE_PATH = ROOT / "코드_최종.py"
MODEL_KEYS = ["Proposed", "LSTM", "MLP", "CNN", "MUSIC"]
DISPLAY_NAMES = {
    "Proposed": "Proposed",
    "LSTM": "LSTM",
    "MLP": "MLP",
    "CNN": "1D-CNN",
    "MUSIC": "MUSIC-LS",
}


def load_final_code():
    spec = importlib.util.spec_from_file_location("final_code_module", FINAL_CODE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_models(fc):
    scaler_x = joblib.load(ROOT / "scaler_x_td_0.0-15.0_doa_0.0-0.5_500M.pkl")
    scaler_y = joblib.load(ROOT / "scaler_y_td_0.0-15.0_doa_0.0-0.5_500M.pkl")

    proposed = fc.TransformerEncoderOnlyModel(25, 3, 128, 8, 9).to(fc.DEVICE)
    proposed.load_state_dict(
        torch.load(ROOT / "model_td_0.0-15.0_doa_0.0-0.5_500M.pt", map_location=fc.DEVICE)
    )

    lstm = fc.LSTMModel(25, 3, 256, 3, 0.3).to(fc.DEVICE)
    lstm.load_state_dict(
        torch.load(ROOT / "model_lstm_td_0.0-15.0_doa_0.0-0.5_500M.pt", map_location=fc.DEVICE)
    )

    mlp = fc.MLPModel(25, 3, 20, 1024, 0.3).to(fc.DEVICE)
    mlp.load_state_dict(
        torch.load(ROOT / "model_mlp_td_0.0-15.0_doa_0.0-0.5_500M.pt", map_location=fc.DEVICE)
    )

    cnn = fc.CNN1DModel(25, 3, 0.3).to(fc.DEVICE)
    cnn.load_state_dict(
        torch.load(ROOT / "model_cnn_td_0.0-15.0_doa_0.0-0.5_500M.pt", map_location=fc.DEVICE)
    )

    models = {"Proposed": proposed, "LSTM": lstm, "MLP": mlp, "CNN": cnn}
    for model in models.values():
        model.eval()
    return scaler_x, scaler_y, models


def evaluate_condition(fc, sx, sy, models, iterations, *, target_dist_cm, tdoa_std_cm, doa_std_deg, tdoa_bias_cm):
    errs_cm = {key: [] for key in MODEL_KEYS}
    sensors = fc.SENSORS_CM

    with torch.no_grad():
        for i in range(iterations):
            gt_cm, feat_cm, raw_signals = fc.generate_controlled_traj_cm(
                tdoa_std_cm, doa_std_deg, target_dist_cm, tdoa_bias_cm
            )

            for key in ["Proposed", "LSTM", "MLP", "CNN"]:
                pred_cm = fc.sliding_window_inference_cm(models[key], sx, sy, feat_cm)
                errs_cm[key].append(fc.calculate_rmse(gt_cm, pred_cm))

            music_localizer = fc.MusicLocalizer(sensors)
            music_traj = []
            for t in range(200):
                doa_vec = fc.music_doa_estimation_stable(sensors, raw_signals[t])
                music_traj.append(music_localizer.update(doa_vec, feat_cm[t]))
            errs_cm["MUSIC"].append(fc.calculate_rmse(gt_cm, np.asarray(music_traj)))

            if (i + 1) % max(1, iterations // 20) == 0 or i + 1 == iterations:
                print(f"\r  progress: {i + 1}/{iterations}", end="", flush=True)
    print()

    return {
        key: np.asarray(values, dtype=np.float32) / 100.0
        for key, values in errs_cm.items()
    }


def summarize(values):
    return float(np.mean(values)), float(np.std(values, ddof=1))


def print_table(title, rows, data):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    print(f"{'Condition':<14} | {'Model':<10} | {'Mean RMSE (m)':>14} | {'Std (m)':>10}")
    print("-" * 78)
    for row_name in rows:
        for key in MODEL_KEYS:
            mean, std = summarize(data[row_name][key])
            print(f"{row_name:<14} | {DISPLAY_NAMES[key]:<10} | {mean:>14.4f} | {std:>10.4f}")
        print("-" * 78)


def print_latex_distance_rows(rows, data):
    print("\n" + "=" * 78)
    print("[LaTeX rows: mean $\\pm$ std, unit: m]")
    print("=" * 78)
    for row_name in rows:
        cells = [row_name.replace(" ", "~")]
        for key in MODEL_KEYS:
            mean, std = summarize(data[row_name][key])
            cells.append(f"{mean:.2f} $\\pm$ {std:.2f}")
        print("    " + " & ".join(cells) + r" \\")


def main():
    parser = argparse.ArgumentParser(
        description="Compute RMSE mean/std only for paper-summary conditions."
    )
    parser.add_argument("--iter", type=int, default=10000, help="Monte Carlo iterations per selected condition")
    parser.add_argument(
        "--distances-m",
        type=float,
        nargs="*",
        default=[0, 100, 200, 300, 400, 500, 600],
        help="Distance conditions for Table II-style summary",
    )
    parser.add_argument(
        "--noise-summary",
        action="store_true",
        help="Also compute representative endpoint conditions for TDOA bias, DOA input angular error, and TDOA random input error std",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional .npz output path. By default, no file is saved.",
    )
    args = parser.parse_args()

    fc = load_final_code()
    sx, sy, models = load_models(fc)

    print(f"Device: {fc.DEVICE}")
    print(f"Iterations per selected condition: {args.iter}")

    payload = None
    if args.save:
        payload = {
            "iter_count": np.array([args.iter], dtype=np.int32),
            "distances_m": np.asarray(args.distances_m, dtype=np.float32),
        }
    distance_data = {}

    for distance_m in args.distances_m:
        row_name = f"{distance_m:g} m"
        print(f"\nDistance condition: {row_name}")
        res = evaluate_condition(
            fc, sx, sy, models, args.iter,
            target_dist_cm=distance_m * 100.0,
            tdoa_std_cm=7.5,
            doa_std_deg=0.5,
            tdoa_bias_cm=0.0,
        )
        distance_data[row_name] = res
        if payload is not None:
            for key, values in res.items():
                payload[f"dist_{distance_m:g}m_{key}"] = values

    print_table("[Distance RMSE Mean/Std]", list(distance_data.keys()), distance_data)
    print_latex_distance_rows(list(distance_data.keys()), distance_data)

    if args.noise_summary:
        noise_conditions = {
            "TDOA bias 100us": dict(target_dist_cm=40000.0, tdoa_std_cm=7.5, doa_std_deg=0.5, tdoa_bias_cm=15.0),
            "DOA input angular error std 1.2deg": dict(target_dist_cm=40000.0, tdoa_std_cm=7.5, doa_std_deg=1.2, tdoa_bias_cm=0.0),
            "TDOA random input error std 100us": dict(target_dist_cm=40000.0, tdoa_std_cm=15.0, doa_std_deg=0.5, tdoa_bias_cm=0.0),
        }
        noise_data = {}
        for name, kwargs in noise_conditions.items():
            print(f"\nNoise condition: {name}")
            res = evaluate_condition(fc, sx, sy, models, args.iter, **kwargs)
            noise_data[name] = res
            if payload is not None:
                safe_name = name.replace(" ", "_").replace(".", "p")
                for key, values in res.items():
                    payload[f"noise_{safe_name}_{key}"] = values
        print_table("[Representative Input Error RMSE Mean/Std]", list(noise_data.keys()), noise_data)

    if payload is not None:
        output_path = Path(args.save)
        np.savez_compressed(output_path, **payload)
        print(f"\nSaved: {output_path}")
    else:
        print("\nNo output file saved. Copy the printed values if needed.")


if __name__ == "__main__":
    main()
