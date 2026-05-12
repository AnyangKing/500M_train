import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter


plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "comparison_results.npz"

MODEL_KEYS = ["Proposed", "MUSIC", "LSTM", "MLP", "KF", "CNN"]
OUTPUT_MODEL_KEYS = ["Proposed", "MUSIC", "LSTM", "MLP", "CNN"]
DOA_OUTPUT_MODEL_KEYS = ["Proposed", "LSTM", "MLP", "CNN"]
MODEL_STYLES = {
    "Proposed": {"marker": "o", "color": "r", "ls": "-"},
    "MUSIC": {"marker": "D", "color": "green", "ls": "--"},
    "LSTM": {"marker": "s", "color": "m", "ls": "-"},
    "MLP": {"marker": "^", "color": "b", "ls": "-"},
    "KF": {"marker": "P", "color": "orange", "ls": "-"},
    "CNN": {"marker": "x", "color": "c", "ls": "-"},
}


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
    clustered_tracks = [gt_track] + list(pred_tracks)
    limits = []
    spans = []
    for dim in range(3):
        vmin, vmax = get_axis_limits_from_tracks(clustered_tracks, dim, padding_ratio=padding_ratio, min_padding=min_padding)
        limits.append((vmin, vmax))
        spans.append(max(vmax - vmin, 1.0))
    return limits, tuple(spans)


def load_results(path):
    data = np.load(path, allow_pickle=False)
    bundle = {k: data[k] for k in data.files}
    return bundle


def display_label(key):
    return "1D-CNN" if key == "CNN" else key


def print_summary_tables(dist_steps, tdoa_m_steps_cm, tdoa_std_steps_cm, doa_steps, r_dist, r_tdoa, r_tdoa_std, r_doa):
    print(f"\n\n{'='*175}\n [ 종합 RMSE 비교 요약: 거리별(m) ]\n{'='*175}")
    print(f"{'Dist(m)':<10} | {'Prop':<16} | {'MUSIC':<16} | {'LSTM':<16} | {'MLP':<16} | {'1D-CNN'}")
    print(f"{'-'*175}")
    for i in range(len(dist_steps)):
        val = int(round(dist_steps[i] / 100.0))
        if val % 10 == 0:
            print(
                f"{val:>8} | {r_dist['Proposed'][i]:<16.4f} | {r_dist['MUSIC'][i]:<16.4f} | "
                f"{r_dist['LSTM'][i]:<16.4f} | {r_dist['MLP'][i]:<16.4f} | "
                f"{r_dist['CNN'][i]:.4f}"
            )

    print(f"\n\n{'='*175}\n [ 종합 RMSE 비교 요약: TDOA 평균 바이어스별(us) ]\n{'='*175}")
    print(f"{'Bias(us)':<10} | {'Prop':<16} | {'MUSIC':<16} | {'LSTM':<16} | {'MLP':<16} | {'1D-CNN'}")
    print(f"{'-'*175}")
    for i in range(len(tdoa_m_steps_cm)):
        s_str = f"{round(i * 1.0):>8}"
        print(
            f"{s_str} | {r_tdoa['Proposed'][i]:<16.4f} | {r_tdoa['MUSIC'][i]:<16.4f} | "
            f"{r_tdoa['LSTM'][i]:<16.4f} | {r_tdoa['MLP'][i]:<16.4f} | "
            f"{r_tdoa['CNN'][i]:.4f}"
        )

    print(f"\n\n{'='*175}\n [ 종합 RMSE 비교 요약: TDOA 노이즈 표준편차별(us) ]\n{'='*175}")
    print(f"{'Std(us)':<10} | {'Prop':<16} | {'MUSIC':<16} | {'LSTM':<16} | {'MLP':<16} | {'1D-CNN'}")
    print(f"{'-'*175}")
    for i in range(len(tdoa_std_steps_cm)):
        s_str = f"{round(i * 1.0):>8}"
        print(
            f"{s_str} | {r_tdoa_std['Proposed'][i]:<16.4f} | {r_tdoa_std['MUSIC'][i]:<16.4f} | "
            f"{r_tdoa_std['LSTM'][i]:<16.4f} | {r_tdoa_std['MLP'][i]:<16.4f} | "
            f"{r_tdoa_std['CNN'][i]:.4f}"
        )

    print(f"\n\n{'='*175}\n [ 종합 RMSE 비교 요약: DOA 오차별(deg) ]\n{'='*175}")
    print(f"{'DOA(deg)':<10} | {'Prop':<16} | {'LSTM':<16} | {'MLP':<16} | {'1D-CNN'}")
    print(f"{'-'*175}")
    for i in range(len(doa_steps)):
        s_str = f"{doa_steps[i]:>8.1f}"
        print(
            f"{s_str} | {r_doa['Proposed'][i]:<16.4f} | {r_doa['LSTM'][i]:<16.4f} | "
            f"{r_doa['MLP'][i]:<16.4f} | {r_doa['CNN'][i]:.4f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Plot saved comparison results.")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT), help="Input .npz results path")
    parser.add_argument("--dpi", type=int, default=600, help="Saved figure DPI")
    parser.add_argument("--plane-padding-ratio", type=float, default=0.03, help="Padding ratio for XY/XZ/YZ plane figures")
    parser.add_argument("--plane-min-padding", type=float, default=2.0, help="Minimum padding for XY/XZ/YZ plane figures")
    args = parser.parse_args()

    data = load_results(Path(args.input))

    dist_steps = data["dist_steps"]
    tdoa_m_steps_cm = data["tdoa_m_steps_cm"]
    tdoa_std_steps_cm = data["tdoa_std_steps_cm"]
    doa_steps = data["doa_steps"]

    r_dist = {k: data[f"r_dist_{k}"] for k in MODEL_KEYS}
    r_tdoa = {k: data[f"r_tdoa_{k}"] for k in MODEL_KEYS}
    r_tdoa_std = {k: data[f"r_tdoa_std_{k}"] for k in MODEL_KEYS}
    r_doa = {k: data[f"r_doa_{k}"] for k in MODEL_KEYS}

    print_summary_tables(dist_steps, tdoa_m_steps_cm, tdoa_std_steps_cm, doa_steps, r_dist, r_tdoa, r_tdoa_std, r_doa)

    gt_m = data["viz_gt_m"]
    p_all = {k: data[f"viz_{k}_m"] for k in MODEL_KEYS}
    viz_tracks = [gt_m] + [p_all[k] for k in OUTPUT_MODEL_KEYS]
    clustered_pred_tracks = select_tracks_near_ground_truth(gt_m, [p_all[k] for k in OUTPUT_MODEL_KEYS])

    # Figure 1, 2, 3: 평면별 추정 결과
    planes = [("X", "Y", [0, 1], 1), ("X", "Z", [0, 2], 2), ("Y", "Z", [1, 2], 3)]
    for n1, n2, dims, fig_n in planes:
        plt.figure(fig_n, figsize=(9, 7))
        plt.plot(gt_m[:, dims[0]], gt_m[:, dims[1]], "k--", label="Ground Truth", lw=2)
        for key in OUTPUT_MODEL_KEYS:
            plt.plot(
                p_all[key][:, dims[0]],
                p_all[key][:, dims[1]],
                label=display_label(key),
                color=MODEL_STYLES[key]["color"],
                marker=MODEL_STYLES[key]["marker"],
                ls=MODEL_STYLES[key]["ls"],
                lw=1.5,
                markevery=15,
            )
        plt.xlim(*get_axis_limits_for_plane(
            gt_m, clustered_pred_tracks, dims[0], padding_ratio=args.plane_padding_ratio, min_padding=args.plane_min_padding
        ))
        plt.ylim(*get_axis_limits_for_plane(
            gt_m, clustered_pred_tracks, dims[1], padding_ratio=args.plane_padding_ratio, min_padding=args.plane_min_padding
        ))
        plt.grid(True, ls=":", alpha=0.6)
        plt.legend()
        plt.tight_layout()

    # Figure 4: 거리 분석
    plt.figure(4, figsize=(10, 7))
    plt.gca().set_xticks(np.arange(0, 601, 100))
    plt.gca().xaxis.grid(True, ls=":", alpha=0.5)
    plt.gca().yaxis.grid(True, which="both", ls=":", alpha=0.5)
    for key in ["Proposed", "LSTM", "MLP", "CNN", "MUSIC"]:
        plt.plot(
            dist_steps / 100.0,
            r_dist[key],
            label=display_label(key),
            color=MODEL_STYLES[key]["color"],
            marker=MODEL_STYLES[key]["marker"],
            ls=MODEL_STYLES[key]["ls"],
            lw=1.5,
            markevery=10,
        )
    plt.yscale("log")
    plt.ylim(0.1, 100)
    plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    plt.xlabel("Distance (m)")
    plt.ylabel("RMSE (m)")
    plt.legend()
    plt.tight_layout()

    # Figure 5: TDOA 바이어스 분석
    plt.figure(5, figsize=(10, 7))
    td_us = (tdoa_m_steps_cm / 150000.0) * 1000000.0
    plt.gca().set_xticks(np.arange(0, 101, 10))
    plt.gca().xaxis.grid(True, ls=":", alpha=0.5)
    plt.gca().yaxis.grid(True, which="both", ls=":", alpha=0.5)
    for key in OUTPUT_MODEL_KEYS:
        plt.plot(
            td_us,
            r_tdoa[key],
            label=display_label(key),
            color=MODEL_STYLES[key]["color"],
            marker=MODEL_STYLES[key]["marker"],
            ls=MODEL_STYLES[key]["ls"],
            lw=1.5,
            markevery=10,
        )
    plt.yscale("log")
    plt.ylim(0.1, 100)
    plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    plt.xlabel(r"TDOA Bias ($\mu s$)")
    plt.ylabel("RMSE (m)")
    plt.legend()
    plt.tight_layout()

    # Figure 6: DOA 검증 (MUSIC 제외)
    plt.figure(6, figsize=(10, 7))
    plt.gca().set_xticks(doa_steps)
    plt.gca().xaxis.grid(True, ls=":", alpha=0.5)
    plt.gca().yaxis.grid(True, which="both", ls=":", alpha=0.5)
    for key in DOA_OUTPUT_MODEL_KEYS:
        plt.plot(
            doa_steps,
            r_doa[key],
            label=display_label(key),
            color=MODEL_STYLES[key]["color"],
            marker=MODEL_STYLES[key]["marker"],
            ls=MODEL_STYLES[key]["ls"],
            lw=1.5,
            markevery=1,
        )
    plt.yscale("log")
    plt.ylim(0.1, 100)
    plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    plt.xlabel("DOA Deviation (deg)")
    plt.ylabel("RMSE (m)")
    plt.legend()
    plt.tight_layout()

    # Figure 7: 3D 궤적
    fig7 = plt.figure(7, figsize=(10, 8))
    ax7 = fig7.add_subplot(111, projection="3d")
    ax7.plot(gt_m[:, 0], gt_m[:, 1], gt_m[:, 2], "k--", label="Ground Truth", lw=2)
    for key in ["Proposed", "LSTM", "MLP", "CNN", "MUSIC"]:
        ax7.plot(
            p_all[key][:, 0],
            p_all[key][:, 1],
            p_all[key][:, 2],
            label=display_label(key),
            color=MODEL_STYLES[key]["color"],
            marker=MODEL_STYLES[key]["marker"],
            ls=MODEL_STYLES[key]["ls"],
            lw=1.5,
            markevery=20,
        )
    limits_3d, aspect_3d = get_3d_limits_and_aspect(
        gt_m, clustered_pred_tracks, padding_ratio=0.04, min_padding=3.0
    )
    ax7.set_xlim(*limits_3d[0])
    ax7.set_ylim(*limits_3d[1])
    ax7.set_zlim(*limits_3d[2])
    ax7.set_box_aspect(aspect_3d)
    ax7.view_init(elev=18, azim=-48)
    ax7.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    plt.tight_layout()

    # Figure 8: 거리 100-300m 상세
    plt.figure(8, figsize=(10, 7))
    mask = (dist_steps >= 10000) & (dist_steps <= 30000)
    steps_sub = dist_steps[mask] / 100.0
    plt.gca().set_xticks(np.arange(100, 301, 20))
    plt.gca().xaxis.grid(True, ls=":", alpha=0.5)
    plt.gca().yaxis.grid(True, which="both", ls=":", alpha=0.5)
    for key in ["Proposed", "LSTM", "MLP", "CNN", "MUSIC"]:
        plt.plot(
            steps_sub,
            np.array(r_dist[key])[mask],
            label=display_label(key),
            color=MODEL_STYLES[key]["color"],
            marker=MODEL_STYLES[key]["marker"],
            ls=MODEL_STYLES[key]["ls"],
            lw=2.0,
            markevery=2,
        )
    plt.yscale("log")
    plt.ylim(0.1, 100)
    plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    plt.xlabel("Distance (m)")
    plt.ylabel("RMSE (m)")
    plt.legend()
    plt.tight_layout()

    # Figure 9: TDOA 노이즈 표준편차 분석
    plt.figure(9, figsize=(10, 7))
    td_std_us = (tdoa_std_steps_cm / 150000.0) * 1000000.0
    plt.gca().set_xticks(np.arange(0, 101, 10))
    plt.gca().xaxis.grid(True, ls=":", alpha=0.5)
    plt.gca().yaxis.grid(True, which="both", ls=":", alpha=0.5)
    for key in OUTPUT_MODEL_KEYS:
        plt.plot(
            td_std_us,
            r_tdoa_std[key],
            label=display_label(key),
            color=MODEL_STYLES[key]["color"],
            marker=MODEL_STYLES[key]["marker"],
            ls=MODEL_STYLES[key]["ls"],
            lw=1.5,
            markevery=10,
        )
    plt.yscale("log")
    plt.ylim(0.1, 100)
    plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    plt.xlabel(r"TDOA Noise Std ($\mu s$)")
    plt.ylabel("RMSE (m)")
    plt.legend()
    plt.tight_layout()

    # 논문 제출용 고해상도 Figure 저장 (600 DPI)
    for fig_n in range(1, 10):
        plt.figure(fig_n)
        plt.savefig(ROOT / f"Figure_{fig_n}.png", dpi=args.dpi, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    main()
