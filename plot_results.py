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


def load_results(path):
    data = np.load(path, allow_pickle=False)
    bundle = {k: data[k] for k in data.files}
    return bundle


def main():
    parser = argparse.ArgumentParser(description="Plot saved comparison results.")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT), help="Input .npz results path")
    parser.add_argument("--dpi", type=int, default=600, help="Saved figure DPI")
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

    gt_m = data["viz_gt_m"]
    p_all = {k: data[f"viz_{k}_m"] for k in MODEL_KEYS}
    viz_tracks = [gt_m] + [p_all[k] for k in MODEL_KEYS]

    planes = [("X", "Y", [0, 1], 1), ("X", "Z", [0, 2], 2), ("Y", "Z", [1, 2], 3)]
    for n1, n2, dims, fig_n in planes:
        plt.figure(fig_n, figsize=(9, 7))
        plt.plot(gt_m[:, dims[0]], gt_m[:, dims[1]], "k--", label="Ground Truth", lw=2)
        for key in MODEL_KEYS:
            plt.plot(
                p_all[key][:, dims[0]],
                p_all[key][:, dims[1]],
                label=key,
                color=MODEL_STYLES[key]["color"],
                marker=MODEL_STYLES[key]["marker"],
                ls=MODEL_STYLES[key]["ls"],
                lw=1.5,
                markevery=15,
            )
        plt.xlim(*get_axis_limits_from_tracks(viz_tracks, dims[0], padding_ratio=0.06, min_padding=10.0))
        plt.ylim(*get_axis_limits_from_tracks(viz_tracks, dims[1], padding_ratio=0.06, min_padding=10.0))
        plt.title(f"{n1}-{n2} Plane Estimation (m)")
        plt.grid(True, ls=":", alpha=0.6)
        plt.legend()
        plt.tight_layout()

    plt.figure(4, figsize=(10, 7))
    plt.gca().set_xticks(np.arange(0, 601, 100))
    plt.gca().xaxis.grid(True, ls=":", alpha=0.5)
    plt.gca().yaxis.grid(True, which="both", ls=":", alpha=0.5)
    for key in ["Proposed", "LSTM", "MLP", "KF", "CNN", "MUSIC"]:
        plt.plot(
            dist_steps / 100.0,
            r_dist[key],
            label=("1D-CNN" if key == "CNN" else key),
            color=MODEL_STYLES[key]["color"],
            marker=MODEL_STYLES[key]["marker"],
            ls=MODEL_STYLES[key]["ls"],
            lw=1.5,
            markevery=10,
        )
    plt.yscale("log")
    plt.ylim(0.1, 100)
    plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    plt.title("Figure 4: Distance Error Analysis")
    plt.xlabel("Distance (m)")
    plt.ylabel("RMSE (m)")
    plt.legend()
    plt.tight_layout()

    plt.figure(5, figsize=(10, 7))
    td_us = (tdoa_m_steps_cm / 150000.0) * 1000000.0
    plt.gca().set_xticks(np.arange(0, 101, 10))
    plt.gca().xaxis.grid(True, ls=":", alpha=0.5)
    plt.gca().yaxis.grid(True, which="both", ls=":", alpha=0.5)
    for key in MODEL_KEYS:
        plt.plot(
            td_us,
            r_tdoa[key],
            label=("1D-CNN" if key == "CNN" else key),
            color=MODEL_STYLES[key]["color"],
            marker=MODEL_STYLES[key]["marker"],
            ls=MODEL_STYLES[key]["ls"],
            lw=1.5,
            markevery=10,
        )
    plt.yscale("log")
    plt.ylim(0.1, 100)
    plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    plt.title("TDOA Synchronization Error Analysis")
    plt.xlabel(r"TDOA Bias ($\mu s$)")
    plt.ylabel("RMSE (m)")
    plt.legend()
    plt.tight_layout()

    plt.figure(6, figsize=(10, 7))
    plt.gca().set_xticks(doa_steps)
    plt.gca().xaxis.grid(True, ls=":", alpha=0.5)
    plt.gca().yaxis.grid(True, which="both", ls=":", alpha=0.5)
    for key in MODEL_KEYS:
        if key == "MUSIC":
            continue
        plt.plot(
            doa_steps,
            r_doa[key],
            label=("1D-CNN" if key == "CNN" else key),
            color=MODEL_STYLES[key]["color"],
            marker=MODEL_STYLES[key]["marker"],
            ls=MODEL_STYLES[key]["ls"],
            lw=1.5,
            markevery=1,
        )
    plt.yscale("log")
    plt.ylim(0.1, 100)
    plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    plt.title("DOA Validation")
    plt.xlabel("DOA Deviation (deg)")
    plt.ylabel("RMSE (m)")
    plt.legend()
    plt.tight_layout()

    fig7 = plt.figure(7, figsize=(10, 8))
    ax7 = fig7.add_subplot(111, projection="3d")
    ax7.plot(gt_m[:, 0], gt_m[:, 1], gt_m[:, 2], "k--", label="Ground Truth", lw=2)
    for key in ["Proposed", "LSTM", "MLP", "KF", "CNN", "MUSIC"]:
        ax7.plot(
            p_all[key][:, 0],
            p_all[key][:, 1],
            p_all[key][:, 2],
            label=key,
            color=MODEL_STYLES[key]["color"],
            marker=MODEL_STYLES[key]["marker"],
            ls=MODEL_STYLES[key]["ls"],
            lw=1.5,
            markevery=20,
        )
    ax7.set_xlim(*get_axis_limits_from_tracks(viz_tracks, 0, padding_ratio=0.06, min_padding=10.0))
    ax7.set_ylim(*get_axis_limits_from_tracks(viz_tracks, 1, padding_ratio=0.06, min_padding=10.0))
    ax7.set_zlim(*get_axis_limits_from_tracks(viz_tracks, 2, padding_ratio=0.06, min_padding=5.0))
    ax7.set_title("Figure 7: 3D Trajectory")
    ax7.legend()
    plt.tight_layout()

    plt.figure(8, figsize=(10, 7))
    mask = (dist_steps >= 10000) & (dist_steps <= 30000)
    steps_sub = dist_steps[mask] / 100.0
    plt.gca().set_xticks(np.arange(100, 301, 20))
    plt.gca().xaxis.grid(True, ls=":", alpha=0.5)
    plt.gca().yaxis.grid(True, which="both", ls=":", alpha=0.5)
    for key in ["Proposed", "LSTM", "MLP", "KF", "CNN", "MUSIC"]:
        plt.plot(
            steps_sub,
            np.array(r_dist[key])[mask],
            label=("1D-CNN" if key == "CNN" else key),
            color=MODEL_STYLES[key]["color"],
            marker=MODEL_STYLES[key]["marker"],
            ls=MODEL_STYLES[key]["ls"],
            lw=2.0,
            markevery=2,
        )
    plt.yscale("log")
    plt.ylim(0.1, 100)
    plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    plt.title("Figure 8: Distance Error (100~300m)")
    plt.xlabel("Distance (m)")
    plt.ylabel("RMSE (m)")
    plt.legend()
    plt.tight_layout()

    plt.figure(9, figsize=(10, 7))
    td_std_us = (tdoa_std_steps_cm / 150000.0) * 1000000.0
    plt.gca().set_xticks(np.arange(0, 101, 10))
    plt.gca().xaxis.grid(True, ls=":", alpha=0.5)
    plt.gca().yaxis.grid(True, which="both", ls=":", alpha=0.5)
    for key in MODEL_KEYS:
        plt.plot(
            td_std_us,
            r_tdoa_std[key],
            label=("1D-CNN" if key == "CNN" else key),
            color=MODEL_STYLES[key]["color"],
            marker=MODEL_STYLES[key]["marker"],
            ls=MODEL_STYLES[key]["ls"],
            lw=1.5,
            markevery=10,
        )
    plt.yscale("log")
    plt.ylim(0.1, 100)
    plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    plt.title("TDOA Noise Std Analysis")
    plt.xlabel(r"TDOA Noise Std ($\mu s$)")
    plt.ylabel("RMSE (m)")
    plt.legend()
    plt.tight_layout()

    for fig_n in range(1, 10):
        plt.figure(fig_n)
        plt.savefig(ROOT / f"Figure_{fig_n}.png", dpi=args.dpi, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    main()
