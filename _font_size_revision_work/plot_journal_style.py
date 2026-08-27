"""Generate journal-style result and trajectory figures from saved NPZ data."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator


MODEL_ORDER = ["Proposed", "LSTM", "MLP", "CNN", "MUSIC"]
DOA_MODEL_ORDER = ["Proposed", "LSTM", "MLP", "CNN"]
MODEL_STYLES = {
    "Proposed": {
        "color": "#D55E00",
        "linestyle": "-",
        "marker": "o",
        "linewidth": 1.8,
        "zorder": 6,
    },
    "LSTM": {
        "color": "#CC79A7",
        "linestyle": "-.",
        "marker": "s",
        "linewidth": 1.25,
        "zorder": 4,
    },
    "MLP": {
        "color": "#0072B2",
        "linestyle": ":",
        "marker": "^",
        "linewidth": 1.35,
        "zorder": 3,
    },
    "CNN": {
        "color": "#56B4E9",
        "linestyle": (0, (5, 2)),
        "marker": "x",
        "linewidth": 1.2,
        "zorder": 2,
    },
    "MUSIC": {
        "color": "#009E73",
        "linestyle": "--",
        "marker": "D",
        "linewidth": 1.3,
        "zorder": 5,
    },
}

# Source font sizes are intentionally larger than 10 pt.  Each PDF is reduced
# when it is inserted in the review manuscript; these values compensate for
# that reduction so the final, placed text is approximately 10 pt.
DISTANCE_FONT_SIZE = 14.5
ROBUSTNESS_FONT_SIZE = 15.0
TRAJECTORY_2D_FONT_SIZE = 16.0
TRAJECTORY_YZ_FONT_SIZE = 12.0
TRAJECTORY_3D_FONT_SIZE = 17.1


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9.0,
            "axes.linewidth": 0.75,
            "axes.unicode_minus": True,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.minor.size": 2.0,
            "ytick.minor.size": 2.0,
            "savefig.facecolor": "white",
            "savefig.edgecolor": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def style_axes(ax: plt.Axes, *, log_y: bool = False) -> None:
    ax.tick_params(top=False, right=False)
    ax.grid(
        True,
        which="major",
        color="#D7D7D7",
        linewidth=0.55,
        linestyle=(0, (2, 2)),
        zorder=0,
    )
    if log_y:
        ax.grid(
            True,
            which="minor",
            color="#ECECEC",
            linewidth=0.35,
            linestyle=":",
            zorder=0,
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def set_axis_text_size(ax: plt.Axes, font_size: float) -> None:
    """Use one source size for labels, tick labels, and offset text."""
    ax.xaxis.label.set_size(font_size)
    ax.yaxis.label.set_size(font_size)
    ax.tick_params(axis="both", which="both", labelsize=font_size)
    ax.xaxis.get_offset_text().set_fontsize(font_size)
    ax.yaxis.get_offset_text().set_fontsize(font_size)


def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    filename: str,
    dpi: int,
    *,
    pad_inches: float = 0.035,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(filename).stem
    fig.savefig(
        output_dir / filename,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=pad_inches,
    )
    fig.savefig(
        output_dir / f"{stem}.pdf",
        bbox_inches="tight",
        pad_inches=pad_inches,
    )
    plt.close(fig)


def plot_model_line(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    key: str,
    *,
    markevery: int,
) -> None:
    style = MODEL_STYLES[key]
    marker_edge_width = 0.8
    marker_face = style["color"] if key == "Proposed" else "white"
    if key == "CNN":
        marker_face = "none"
    ax.plot(
        x,
        y,
        color=style["color"],
        linestyle=style["linestyle"],
        marker=style["marker"],
        linewidth=style["linewidth"],
        markersize=4.0,
        markeredgewidth=marker_edge_width,
        markerfacecolor=marker_face,
        markeredgecolor=style["color"],
        markevery=markevery,
        zorder=style["zorder"],
    )


def set_log_rmse_axis(ax: plt.Axes) -> None:
    ax.set_yscale("log")
    ax.set_ylim(0.1, 100.0)
    ax.yaxis.set_major_locator(FixedLocator([0.1, 1.0, 10.0, 100.0]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    style_axes(ax, log_y=True)


def plot_rmse_figures(
    results: dict[str, np.ndarray], output_dir: Path, dpi: int
) -> None:
    dist_m = results["dist_steps"] / 100.0
    static_toa_us = results["tdoa_m_steps_cm"] / 150000.0 * 1_000_000.0
    random_toa_us = results["tdoa_std_steps_cm"] / 150000.0 * 1_000_000.0
    doa_deg = results["doa_steps"]

    fig, ax = plt.subplots(figsize=(5.4, 3.45), constrained_layout=True)
    for key in MODEL_ORDER:
        plot_model_line(
            ax,
            dist_m,
            results[f"r_dist_{key}"],
            key,
            markevery=10,
        )
    ax.set_xlabel("Initial Target Range (m)")
    ax.set_ylabel("RMSE (m)")
    ax.set_xticks(np.arange(0, 601, 100))
    set_log_rmse_axis(ax)
    set_axis_text_size(ax, DISTANCE_FONT_SIZE)
    save_figure(fig, output_dir, "rmse_distance_0_600m.png", dpi)

    robustness_specs = [
        (
            "rmse_tdoa_bias_0_100us.png",
            static_toa_us,
            "r_tdoa_{}",
            MODEL_ORDER,
            "Sensor-wise Static TOA\n" r"Bias Level ($\mu$s)",
            10,
        ),
        (
            "rmse_doa_input_angular_error_std_0_1p2deg.png",
            doa_deg,
            "r_doa_{}",
            DOA_MODEL_ORDER,
            "DOA Angular Input\n" r"Error Std. ($^\circ$)",
            2,
        ),
        (
            "rmse_tdoa_random_input_error_std_0_100us.png",
            random_toa_us,
            "r_tdoa_std_{}",
            MODEL_ORDER,
            "Sensor-wise Random TOA\n" r"Measurement Error Std. ($\mu$s)",
            10,
        ),
    ]
    for filename, x, key_pattern, model_order, xlabel, markevery in robustness_specs:
        fig, ax = plt.subplots(figsize=(4.15, 2.8), constrained_layout=True)
        for key in model_order:
            plot_model_line(
                ax,
                x,
                results[key_pattern.format(key)],
                key,
                markevery=markevery,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("RMSE (m)")
        if len(x) > 20:
            ax.set_xticks(np.arange(0, 101, 20))
        else:
            ax.set_xticks(np.arange(0.0, 1.21, 0.2))
        set_log_rmse_axis(ax)
        set_axis_text_size(ax, ROBUSTNESS_FONT_SIZE)
        save_figure(fig, output_dir, filename, dpi)


def padded_limits(
    arrays: list[np.ndarray], dim: int, padding_ratio: float = 0.035
) -> tuple[float, float]:
    values = np.concatenate([array[:, dim] for array in arrays])
    lower = float(np.min(values))
    upper = float(np.max(values))
    padding = max((upper - lower) * padding_ratio, 1.0)
    return lower - padding, upper + padding


def plot_trajectory_line(
    ax: plt.Axes,
    track: np.ndarray,
    dims: tuple[int, int],
    key: str,
) -> None:
    style = MODEL_STYLES[key]
    marker_face = style["color"] if key == "Proposed" else "white"
    if key == "CNN":
        marker_face = "none"
    ax.plot(
        track[:, dims[0]],
        track[:, dims[1]],
        color=style["color"],
        linestyle=style["linestyle"],
        marker=style["marker"],
        linewidth=style["linewidth"],
        markersize=3.6,
        markeredgewidth=0.75,
        markerfacecolor=marker_face,
        markeredgecolor=style["color"],
        markevery=25,
        zorder=style["zorder"],
    )


def plot_trajectory_figures(
    trajectory: dict[str, np.ndarray], output_dir: Path, dpi: int
) -> None:
    gt = trajectory["viz_gt_m"]
    predictions = {key: trajectory[f"viz_{key}_m"] for key in MODEL_ORDER}
    all_tracks = [gt, *predictions.values()]

    plane_specs = [
        (
            (0, 1),
            "X (m)",
            "Y (m)",
            "trajectory_xy_plane.png",
            TRAJECTORY_2D_FONT_SIZE,
        ),
        (
            (0, 2),
            "X (m)",
            "Z (m)",
            "trajectory_xz_plane.png",
            TRAJECTORY_2D_FONT_SIZE,
        ),
        (
            (1, 2),
            "Y (m)",
            "Z (m)",
            "trajectory_yz_plane.png",
            TRAJECTORY_YZ_FONT_SIZE,
        ),
    ]
    for dims, xlabel, ylabel, filename, font_size in plane_specs:
        fig, ax = plt.subplots(figsize=(5.2, 4.0), constrained_layout=True)
        ax.plot(
            gt[:, dims[0]],
            gt[:, dims[1]],
            color="#1A1A1A",
            linestyle=(0, (5, 2)),
            linewidth=1.55,
            zorder=7,
        )
        for key in MODEL_ORDER:
            plot_trajectory_line(ax, predictions[key], dims, key)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(*padded_limits(all_tracks, dims[0]))
        ax.set_ylim(*padded_limits(all_tracks, dims[1]))
        ax.set_aspect("equal", adjustable="box")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        style_axes(ax)
        set_axis_text_size(ax, font_size)
        save_figure(fig, output_dir, filename, dpi)

    fig = plt.figure(figsize=(5.8, 4.6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        gt[:, 0],
        gt[:, 1],
        gt[:, 2],
        color="#1A1A1A",
        linestyle=(0, (5, 2)),
        linewidth=1.55,
        zorder=7,
    )
    for key in MODEL_ORDER:
        style = MODEL_STYLES[key]
        marker_face = style["color"] if key == "Proposed" else "white"
        if key == "CNN":
            marker_face = "none"
        ax.plot(
            predictions[key][:, 0],
            predictions[key][:, 1],
            predictions[key][:, 2],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=style["linewidth"],
            markersize=3.4,
            markeredgewidth=0.7,
            markerfacecolor=marker_face,
            markeredgecolor=style["color"],
            markevery=25,
            zorder=style["zorder"],
        )
    limits = [padded_limits(all_tracks, dim, 0.045) for dim in range(3)]
    spans = [upper - lower for lower, upper in limits]
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_zlim(*limits[2])
    ax.set_box_aspect(spans)
    ax.set_xlabel("X (m)", labelpad=12)
    ax.set_ylabel("Y (m)", labelpad=12)
    ax.set_zlabel("")
    ax.text2D(
        0.86,
        0.50,
        "Z (m)",
        transform=ax.transAxes,
        rotation=90,
        ha="center",
        va="center",
        fontsize=TRAJECTORY_3D_FONT_SIZE,
    )
    ax.view_init(elev=20, azim=-52)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.tick_params(axis="x", which="both", labelsize=TRAJECTORY_3D_FONT_SIZE)
    ax.tick_params(axis="y", which="both", labelsize=TRAJECTORY_3D_FONT_SIZE)
    ax.tick_params(axis="z", which="both", labelsize=TRAJECTORY_3D_FONT_SIZE)
    ax.xaxis.label.set_size(TRAJECTORY_3D_FONT_SIZE)
    ax.yaxis.label.set_size(TRAJECTORY_3D_FONT_SIZE)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"].update(
            {"color": "#D8D8D8", "linewidth": 0.45, "linestyle": ":"}
        )
    fig.subplots_adjust(left=0.02, right=0.96, bottom=0.02, top=0.98)
    save_figure(fig, output_dir, "trajectory_3d.png", dpi, pad_inches=0.15)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()
    results = load_npz(args.results)
    trajectory = load_npz(args.trajectory)
    plot_rmse_figures(results, args.output, args.dpi)
    plot_trajectory_figures(trajectory, args.output, args.dpi)
    print(f"Saved journal-style candidates to: {args.output}")


if __name__ == "__main__":
    main()
