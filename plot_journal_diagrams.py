"""Generate clean journal-style sensor-array and Transformer diagrams."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "savefig.edgecolor": "none",
        }
    )


def save_both(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / f"{stem}.png",
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.025,
    )
    fig.savefig(
        output_dir / f"{stem}.pdf",
        bbox_inches="tight",
        pad_inches=0.025,
    )
    plt.close(fig)


def plot_sensor_array(output_dir: Path, dpi: int) -> None:
    radius_cm = 3.3
    spacing_cm = 7.9
    root_two = np.sqrt(2.0)
    sensors = np.array(
        [
            [radius_cm, 0.0, 0.0],
            [radius_cm / root_two, radius_cm / root_two, -spacing_cm],
            [0.0, radius_cm, 0.0],
            [-radius_cm / root_two, radius_cm / root_two, -spacing_cm],
            [-radius_cm, 0.0, 0.0],
            [-radius_cm / root_two, -radius_cm / root_two, -spacing_cm],
            [0.0, -radius_cm, 0.0],
            [radius_cm / root_two, -radius_cm / root_two, -spacing_cm],
        ]
    )
    upper = [0, 2, 4, 6]
    lower = [1, 3, 5, 7]

    fig = plt.figure(figsize=(4.25, 3.75))
    ax = fig.add_subplot(111, projection="3d")
    theta = np.linspace(0.0, 2.0 * np.pi, 240)
    ring_x = radius_cm * np.cos(theta)
    ring_y = radius_cm * np.sin(theta)
    ax.plot(
        ring_x,
        ring_y,
        np.zeros_like(theta),
        color="#777777",
        linewidth=0.8,
        linestyle=(0, (3, 2)),
    )
    ax.plot(
        ring_x,
        ring_y,
        np.full_like(theta, -spacing_cm),
        color="#777777",
        linewidth=0.8,
        linestyle=(0, (3, 2)),
    )
    for angle in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
        x = radius_cm * np.cos(angle)
        y = radius_cm * np.sin(angle)
        ax.plot(
            [x, x],
            [y, y],
            [0.0, -spacing_cm],
            color="#C5C5C5",
            linewidth=0.45,
            linestyle=":",
        )

    ax.scatter(
        sensors[upper, 0],
        sensors[upper, 1],
        sensors[upper, 2],
        s=38,
        marker="o",
        facecolor="#0072B2",
        edgecolor="white",
        linewidth=0.7,
        depthshade=False,
        zorder=5,
    )
    ax.scatter(
        sensors[lower, 0],
        sensors[lower, 1],
        sensors[lower, 2],
        s=36,
        marker="D",
        facecolor="#D55E00",
        edgecolor="white",
        linewidth=0.7,
        depthshade=False,
        zorder=5,
    )
    label_offsets = {
        0: (0.35, 0.15, 0.35),
        1: (-0.45, 0.25, -0.30),
        2: (0.10, 0.35, 0.35),
        3: (-0.35, 0.25, -0.25),
        4: (-0.35, 0.10, 0.35),
        5: (-0.35, -0.20, -0.25),
        6: (0.05, -0.45, 0.35),
        7: (0.35, -0.25, -0.25),
    }
    for index, position in enumerate(sensors):
        dx, dy, dz = label_offsets[index]
        ax.text(
            position[0] + dx,
            position[1] + dy,
            position[2] + dz,
            rf"$S_{{{index}}}$",
            fontsize=7.8,
            color="#222222",
            ha="center",
            va="center",
        )

    # Radius and inter-plane spacing annotations.
    ax.plot(
        [0.0, radius_cm],
        [0.0, 0.0],
        [0.0, 0.0],
        color="#333333",
        linewidth=0.8,
    )
    ax.text(
        radius_cm * 0.48,
        -0.35,
        0.35,
        r"$r=3.3$ cm",
        fontsize=7.5,
        ha="center",
    )
    dimension_x = radius_cm + 1.45
    ax.plot(
        [dimension_x, dimension_x],
        [0.0, 0.0],
        [0.0, -spacing_cm],
        color="#333333",
        linewidth=0.8,
    )
    ax.plot(
        [dimension_x - 0.25, dimension_x + 0.25],
        [0.0, 0.0],
        [0.0, 0.0],
        color="#333333",
        linewidth=0.8,
    )
    ax.plot(
        [dimension_x - 0.25, dimension_x + 0.25],
        [0.0, 0.0],
        [-spacing_cm, -spacing_cm],
        color="#333333",
        linewidth=0.8,
    )
    ax.text(
        dimension_x + 0.45,
        0.0,
        -spacing_cm / 2.0,
        r"$h_s=7.9$ cm",
        fontsize=7.5,
        rotation=90,
        va="center",
    )
    ax.text(-0.1, 0.0, 0.65, r"$z=0$", fontsize=7.2, color="#555555")
    ax.text(
        -0.2,
        0.0,
        -spacing_cm - 0.65,
        r"$z=-h_s$",
        fontsize=7.2,
        color="#555555",
    )

    ax.set_xlabel("X (cm)", labelpad=1)
    ax.set_ylabel("Y (cm)", labelpad=1)
    ax.set_zlabel("Z (cm)", labelpad=2)
    ax.set_xlim(-4.8, 5.8)
    ax.set_ylim(-4.8, 4.8)
    ax.set_zlim(-9.7, 1.7)
    ax.set_xticks([-3, 0, 3])
    ax.set_yticks([-3, 0, 3])
    ax.set_zticks([-8, -4, 0])
    ax.set_box_aspect((1.0, 1.0, 1.15))
    ax.view_init(elev=22, azim=-48)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("#E0E0E0")
        axis._axinfo["grid"].update(
            {"color": "#E1E1E1", "linewidth": 0.4, "linestyle": ":"}
        )
    fig.subplots_adjust(left=0.0, right=0.96, bottom=0.0, top=0.99)
    save_both(fig, output_dir, "sensor_array", dpi)


def add_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    subtitle: str | None = None,
    *,
    facecolor: str = "white",
    edgecolor: str = "#666666",
    linewidth: float = 0.8,
    title_size: float = 8.0,
    subtitle_size: float = 6.8,
) -> None:
    patch = FancyBboxPatch(
        (x - width / 2.0, y - height / 2.0),
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.035",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=3,
    )
    ax.add_patch(patch)
    if subtitle:
        ax.text(
            x,
            y + 0.11 * height,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            fontweight="semibold",
            color="#222222",
            zorder=4,
        )
        ax.text(
            x,
            y - 0.22 * height,
            subtitle,
            ha="center",
            va="center",
            fontsize=subtitle_size,
            color="#444444",
            zorder=4,
        )
    else:
        ax.text(
            x,
            y,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            fontweight="semibold",
            color="#222222",
            zorder=4,
        )


def arrow(ax: plt.Axes, x: float, y0: float, y1: float) -> None:
    ax.annotate(
        "",
        xy=(x, y1),
        xytext=(x, y0),
        arrowprops={
            "arrowstyle": "-|>",
            "color": "#555555",
            "linewidth": 0.85,
            "mutation_scale": 8,
        },
        zorder=5,
    )


def plot_transformer(output_dir: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(3.8, 6.1))
    ax.set_xlim(0.0, 3.8)
    ax.set_ylim(0.0, 9.0)
    ax.axis("off")

    center = 2.05
    width = 2.70
    inner_width = 2.25
    height = 0.66
    positions = {
        "input": 0.55,
        "embedding": 1.55,
        "position": 2.55,
        "attention": 3.80,
        "norm1": 4.75,
        "ffn": 5.70,
        "norm2": 6.65,
        "output": 7.75,
        "estimate": 8.55,
    }

    encoder_bottom = 3.20
    encoder_top = 7.15
    encoder = FancyBboxPatch(
        (0.12, encoder_bottom),
        3.55,
        encoder_top - encoder_bottom,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor="#F2F6FA",
        edgecolor="#4C72B0",
        linewidth=1.0,
        zorder=1,
    )
    ax.add_patch(encoder)
    ax.text(
        0.29,
        (encoder_bottom + encoder_top) / 2.0,
        "Transformer encoder layer",
        rotation=90,
        ha="center",
        va="center",
        fontsize=7.0,
        color="#355A85",
        fontweight="semibold",
    )
    ax.text(
        3.42,
        (encoder_bottom + encoder_top) / 2.0,
        r"$\times\,9$",
        ha="center",
        va="center",
        fontsize=8.0,
        color="#355A85",
        fontweight="semibold",
    )

    add_box(
        ax,
        center,
        positions["input"],
        width,
        height,
        "Input feature sequence",
        r"$\mathbf{X}\in\mathbb{R}^{T\times F_{\mathrm{in}}}$; "
        r"$T=20$, $F_{\mathrm{in}}=25$ in simulation",
        facecolor="#F7F7F7",
    )
    add_box(
        ax,
        center,
        positions["embedding"],
        width,
        height,
        "Linear embedding",
        r"$F_{\mathrm{in}}\rightarrow d_{\mathrm{model}}=128$",
        facecolor="#EAF2F8",
    )
    add_box(
        ax,
        center,
        positions["position"],
        width,
        height,
        "Sinusoidal positional encoding",
        r"sequence length $T=20$",
        facecolor="#EAF2F8",
    )
    add_box(
        ax,
        center,
        positions["attention"],
        inner_width,
        height,
        "Multi-head self-attention",
        r"$N_h=8,\ d_k=d_v=16$",
        facecolor="white",
    )
    add_box(
        ax,
        center,
        positions["norm1"],
        inner_width,
        height * 0.82,
        "Add & layer normalization",
        facecolor="white",
    )
    add_box(
        ax,
        center,
        positions["ffn"],
        inner_width,
        height,
        "Feed-forward network",
        r"$d_{\mathrm{ff}}=512$, ReLU, dropout $=0.0534$",
        facecolor="white",
    )
    add_box(
        ax,
        center,
        positions["norm2"],
        inner_width,
        height * 0.82,
        "Add & layer normalization",
        facecolor="white",
    )
    add_box(
        ax,
        center,
        positions["output"],
        width,
        height,
        "Linear output layer",
        r"$d_{\mathrm{model}}=128\rightarrow 3$",
        facecolor="#EAF2F8",
    )
    add_box(
        ax,
        center,
        positions["estimate"],
        width,
        height,
        "Estimated position sequence",
        r"$\widehat{\mathbf{P}}\in\mathbb{R}^{T\times 3}$",
        facecolor="#F7F7F7",
    )

    ordered = [
        "input",
        "embedding",
        "position",
        "attention",
        "norm1",
        "ffn",
        "norm2",
        "output",
        "estimate",
    ]
    for lower_key, upper_key in zip(ordered[:-1], ordered[1:]):
        lower_height = height * (0.82 if "norm" in lower_key else 1.0)
        upper_height = height * (0.82 if "norm" in upper_key else 1.0)
        arrow(
            ax,
            center,
            positions[lower_key] + lower_height / 2.0 + 0.04,
            positions[upper_key] - upper_height / 2.0 - 0.04,
        )

    # Residual connections inside a single encoder layer.
    residual_x = center - inner_width / 2.0 - 0.25
    target_x = center - inner_width / 2.0 - 0.035
    residual_style = {
        "color": "#888888",
        "linewidth": 0.7,
        "linestyle": (0, (3, 2)),
        "zorder": 4,
    }
    for source_key, target_key in [("attention", "norm1"), ("ffn", "norm2")]:
        source_y = positions[source_key] - height / 2.0 - 0.11
        target_y = positions[target_key]
        ax.plot([center, residual_x], [source_y, source_y], **residual_style)
        ax.plot([residual_x, residual_x], [source_y, target_y], **residual_style)
        ax.annotate(
            "",
            xy=(target_x, target_y),
            xytext=(residual_x, target_y),
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#888888",
                "linewidth": 0.7,
                "linestyle": (0, (3, 2)),
                "mutation_scale": 7,
            },
            zorder=5,
        )
        ax.plot(
            center,
            source_y,
            marker="o",
            markersize=2.3,
            color="#777777",
            zorder=6,
        )
        ax.text(
            residual_x - 0.11,
            target_y + 0.16,
            "+",
            ha="center",
            va="center",
            fontsize=8.5,
            color="#888888",
            fontweight="semibold",
            zorder=6,
        )

    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    save_both(fig, output_dir, "transformer_blockdiagram", dpi)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()
    plot_sensor_array(args.output, args.dpi)
    plot_transformer(args.output, args.dpi)
    print(f"Saved journal-style diagrams to: {args.output}")


if __name__ == "__main__":
    main()
