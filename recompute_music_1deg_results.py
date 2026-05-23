import argparse
import importlib.util
import shutil
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
FINAL_CODE_PATH = ROOT / "코드_최종.py"
DEFAULT_INPUT = ROOT / "comparison_results.npz"
DEFAULT_OUTPUT = ROOT / "comparison_results.npz"


def load_final_code():
    spec = importlib.util.spec_from_file_location("final_code_module", FINAL_CODE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run_music_sweep(fc, steps, sweep_type, iterations):
    results = []
    c_dist, c_std, c_doa = 40000, 7.5, 0.5
    started = time.time()

    for i, val in enumerate(steps):
        errs_cm = []
        t_dist = val if sweep_type == "dist" else c_dist
        t_td_std = val if sweep_type == "tdoa_std" else c_std
        t_td_m = val if sweep_type == "tdoa" else 0.0
        t_doa = val if sweep_type == "doa" else c_doa

        point_started = time.time()
        for _ in range(iterations):
            gt_cm, feat_cm, raw_signals = fc.generate_controlled_traj_cm(
                t_td_std, t_doa, t_dist, t_td_m
            )

            localizer = fc.MusicLocalizer(fc.SENSORS_CM)
            music_traj = []
            for t in range(200):
                doa_vec = fc.music_doa_estimation_stable(fc.SENSORS_CM, raw_signals[t])
                music_traj.append(localizer.update(doa_vec, feat_cm[t]))
            errs_cm.append(fc.calculate_rmse(gt_cm, np.asarray(music_traj)))

        results.append(float(np.mean(errs_cm)) / 100.0)
        elapsed = time.time() - started
        avg_per_point = elapsed / (i + 1)
        remain = avg_per_point * (len(steps) - i - 1)
        point_elapsed = time.time() - point_started
        print(
            f"\rMUSIC {sweep_type} 계산 중... ({i + 1}/{len(steps)}) "
            f"last={point_elapsed/60:.1f} min, ETA={remain/60:.1f} min",
            end="",
            flush=True,
        )

    print()
    return np.asarray(results, dtype=np.float32)


def build_music_visualization(fc):
    gt_cm, feat_cm, raw_signals = fc.generate_controlled_traj_cm(
        7.5, 0.5, target_dist_cm=40000, m_bias_cm=7.5
    )
    localizer = fc.MusicLocalizer(fc.SENSORS_CM)
    music_traj = []
    for t in range(200):
        doa_vec = fc.music_doa_estimation_stable(fc.SENSORS_CM, raw_signals[t])
        music_traj.append(localizer.update(doa_vec, feat_cm[t]) / 100.0)
    return (gt_cm / 100.0).astype(np.float32), np.asarray(music_traj, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Recompute only MUSIC-LS results with the current MUSIC grid resolution."
    )
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT), help="Existing .npz results path")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Updated .npz results path")
    parser.add_argument("--iter", type=int, default=None, help="Override Monte Carlo iterations")
    parser.add_argument(
        "--backup",
        action="store_true",
        default=True,
        help="Create a .bak copy before overwriting the input/output file",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    fc = load_final_code()
    print(f"MUSIC grid resolution: az={fc.MUSIC_AZ_RES} deg, el={fc.MUSIC_EL_RES} deg")

    loaded = np.load(input_path)
    payload = {key: loaded[key] for key in loaded.files}
    iterations = int(args.iter) if args.iter is not None else int(payload["iter_count"][0])
    print(f"ITER={iterations}")

    payload["r_dist_MUSIC"] = run_music_sweep(fc, payload["dist_steps"], "dist", iterations)
    payload["r_tdoa_MUSIC"] = run_music_sweep(fc, payload["tdoa_m_steps_cm"], "tdoa", iterations)
    payload["r_tdoa_std_MUSIC"] = run_music_sweep(fc, payload["tdoa_std_steps_cm"], "tdoa_std", iterations)

    viz_gt_m, viz_music_m = build_music_visualization(fc)
    payload["viz_gt_m"] = viz_gt_m
    payload["viz_MUSIC_m"] = viz_music_m
    payload["music_az_res_deg"] = np.array([fc.MUSIC_AZ_RES], dtype=np.float32)
    payload["music_el_res_deg"] = np.array([fc.MUSIC_EL_RES], dtype=np.float32)

    if args.backup and input_path.resolve() == output_path.resolve():
        backup_path = input_path.with_suffix(input_path.suffix + ".before_music_1deg.bak")
        shutil.copy2(input_path, backup_path)
        print(f"기존 결과 백업: {backup_path}")

    np.savez_compressed(output_path, **payload)
    print(f"저장 완료: {output_path}")


if __name__ == "__main__":
    main()
