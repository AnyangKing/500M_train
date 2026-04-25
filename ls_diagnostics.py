import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import argparse
import importlib.util
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parent
FINAL_CODE_PATH = ROOT / "코드_최종.py"


def load_final_code():
    spec = importlib.util.spec_from_file_location("final_code", FINAL_CODE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def nonlinear_tdoa_ls(tdoa_values_cm, sensors, init_pos=None, prev_pos=None):
    sensors = sensors.astype(np.float64)
    tdoa_values_cm = np.asarray(tdoa_values_cm, dtype=np.float64)
    sensor_center = np.mean(sensors, axis=0).astype(np.float64)

    if init_pos is None:
        init_pos = sensor_center.copy()
    else:
        init_pos = np.asarray(init_pos, dtype=np.float64)

    def residuals(pos):
        dists = np.linalg.norm(sensors - pos, axis=1)
        pred = dists[1:] - dists[0]
        return pred - tdoa_values_cm

    lower = np.array([-100000.0, -100000.0, -100000.0], dtype=np.float64)
    upper = np.array([100000.0, 100000.0, 100000.0], dtype=np.float64)

    start_points = [init_pos, sensor_center]
    if prev_pos is not None:
        start_points.append(np.asarray(prev_pos, dtype=np.float64))
    start_points.extend([
        init_pos + np.array([100.0, 0.0, 0.0]),
        init_pos + np.array([-100.0, 0.0, 0.0]),
        init_pos + np.array([0.0, 100.0, 0.0]),
        init_pos + np.array([0.0, -100.0, 0.0]),
    ])

    best_x = init_pos
    best_cost = np.inf
    for x0 in start_points:
        x0 = np.clip(np.asarray(x0, dtype=np.float64), lower, upper)
        try:
            result = least_squares(
                residuals,
                x0=x0,
                bounds=(lower, upper),
                method="trf",
                loss="soft_l1",
                f_scale=10.0,
                max_nfev=200,
            )
            cost = float(np.mean(residuals(result.x) ** 2))
            if np.isfinite(cost) and cost < best_cost:
                best_cost = cost
                best_x = result.x
        except Exception:
            continue

    return best_x


def ls_track_rmse_m(final_code, td_noise_cm, doa_noise_deg, target_dist_cm, tdoa_bias_cm, mode):
    gt_cm, feat_cm, _ = final_code.generate_controlled_traj_cm(
        td_noise_cm, doa_noise_deg, target_dist_cm=target_dist_cm, m_bias_cm=tdoa_bias_cm
    )

    positions = []
    prev_pos = None
    for t in range(200):
        tdoa = feat_cm[t, 2:9]
        if mode == "linear":
            pos = final_code.solve_ls_localization(tdoa, final_code.SENSORS_CM)
        else:
            init_pos = prev_pos
            if init_pos is None:
                init_pos = final_code.solve_ls_localization(tdoa, final_code.SENSORS_CM)
            pos = nonlinear_tdoa_ls(tdoa, final_code.SENSORS_CM, init_pos=init_pos, prev_pos=prev_pos)
        positions.append(pos)
        prev_pos = pos

    rmse_cm = final_code.calculate_rmse(gt_cm, np.array(positions))
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


def run_sweep(final_code, sweep_name, steps, args, mode):
    print(f"\n[{mode}: {sweep_name}]")
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
                ls_track_rmse_m(
                    final_code,
                    td_noise_cm=td_noise_cm,
                    doa_noise_deg=doa_noise_deg,
                    target_dist_cm=target_dist_cm,
                    tdoa_bias_cm=tdoa_bias_cm,
                    mode=mode,
                )
            )

        print_row(f"{value:.1f}", summarize(rmses, args.fail_threshold_m))


def main():
    parser = argparse.ArgumentParser(description="LS-only RMSE distribution diagnostics.")
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

    print("LS diagnostics")
    print(f"Source: {FINAL_CODE_PATH}")
    print(f"ITER={args.iter}, fail threshold={args.fail_threshold_m:.1f} m")

    steps_distance = np.array([0, 50, 100, 200, 300, 400, 500, 600], dtype=float)
    steps_noise = np.array([0, 10, 20, 40, 60, 80, 100], dtype=float)

    for ls_mode in ("linear", "nonlinear"):
        if args.mode in ("distance", "all"):
            run_sweep(final_code, "distance_m", steps_distance, args, ls_mode)
        if args.mode in ("tdoa_std", "all"):
            run_sweep(final_code, "tdoa_std_us", steps_noise, args, ls_mode)
        if args.mode in ("tdoa_bias", "all"):
            run_sweep(final_code, "tdoa_bias_us", steps_noise, args, ls_mode)


if __name__ == "__main__":
    main()
