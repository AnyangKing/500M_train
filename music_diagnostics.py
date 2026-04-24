import argparse
import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
FINAL_CODE_PATH = ROOT / "코드_최종.py"


def load_final_code():
    spec = importlib.util.spec_from_file_location("final_code", FINAL_CODE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def music_track_rmse_m(final_code, td_noise_cm, doa_noise_deg, target_dist_cm, tdoa_bias_cm):
    gt_cm, feat_cm, raw_signals = final_code.generate_controlled_traj_cm(
        td_noise_cm, doa_noise_deg, target_dist_cm=target_dist_cm, m_bias_cm=tdoa_bias_cm
    )

    music_positions = []
    music_localizer = final_code.MusicLocalizer(final_code.SENSORS_CM)
    for t in range(200):
        m_vec = final_code.music_doa_estimation_stable(final_code.SENSORS_CM, raw_signals[t])
        music_positions.append(music_localizer.update(m_vec, feat_cm[t]))

    rmse_cm = final_code.calculate_rmse(gt_cm, np.array(music_positions))
    return rmse_cm / 100.0


def summarize(values, fail_threshold_m):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "fail_rate": float(np.mean(values > fail_threshold_m) * 100.0),
    }


def print_row(label, stats):
    print(
        f"{label:>10} | "
        f"{stats['mean']:>9.3f} | "
        f"{stats['median']:>9.3f} | "
        f"{stats['p90']:>9.3f} | "
        f"{stats['p95']:>9.3f} | "
        f"{stats['max']:>9.3f} | "
        f"{stats['fail_rate']:>8.2f}%"
    )


def run_sweep(final_code, sweep_name, steps, args):
    print(f"\n[{sweep_name}]")
    print("     value |      mean |    median |       p90 |       p95 |       max | fail>thr")
    print("-" * 83)

    for value in steps:
        rmses = []
        for _ in range(args.iter):
            target_dist_cm = args.base_dist_m * 100.0
            td_noise_cm = args.base_tdoa_std_cm
            tdoa_bias_cm = args.base_tdoa_bias_cm
            doa_noise_deg = args.base_doa_deg

            if sweep_name == "distance_m":
                target_dist_cm = value * 100.0
            elif sweep_name == "tdoa_std_us":
                td_noise_cm = value * final_code.SOUND_SPEED_CM_S / 1_000_000.0
            elif sweep_name == "tdoa_bias_us":
                tdoa_bias_cm = value * final_code.SOUND_SPEED_CM_S / 1_000_000.0

            rmses.append(
                music_track_rmse_m(
                    final_code,
                    td_noise_cm=td_noise_cm,
                    doa_noise_deg=doa_noise_deg,
                    target_dist_cm=target_dist_cm,
                    tdoa_bias_cm=tdoa_bias_cm,
                )
            )

        print_row(f"{value:.1f}", summarize(rmses, args.fail_threshold_m))


def main():
    parser = argparse.ArgumentParser(description="MUSIC-only RMSE distribution diagnostics.")
    parser.add_argument("--iter", type=int, default=50, help="Monte Carlo iterations per step.")
    parser.add_argument("--fail-threshold-m", type=float, default=100.0, help="Failure threshold for RMSE.")
    parser.add_argument("--base-dist-m", type=float, default=400.0)
    parser.add_argument("--base-tdoa-std-cm", type=float, default=7.5)
    parser.add_argument("--base-tdoa-bias-cm", type=float, default=0.0)
    parser.add_argument("--base-doa-deg", type=float, default=0.5)
    parser.add_argument(
        "--mode",
        choices=["distance", "tdoa_std", "tdoa_bias", "all"],
        default="all",
        help="Diagnostic sweep to run.",
    )
    args = parser.parse_args()

    final_code = load_final_code()
    np.random.seed(42)

    print("MUSIC-only diagnostics")
    print(f"Source: {FINAL_CODE_PATH}")
    print(f"ITER={args.iter}, fail threshold={args.fail_threshold_m:.1f} m")

    if args.mode in ("distance", "all"):
        run_sweep(final_code, "distance_m", np.array([0, 50, 100, 200, 300, 400, 500, 600], dtype=float), args)
    if args.mode in ("tdoa_std", "all"):
        run_sweep(final_code, "tdoa_std_us", np.array([0, 10, 20, 40, 60, 80, 100], dtype=float), args)
    if args.mode in ("tdoa_bias", "all"):
        run_sweep(final_code, "tdoa_bias_us", np.array([0, 10, 20, 40, 60, 80, 100], dtype=float), args)


if __name__ == "__main__":
    main()
