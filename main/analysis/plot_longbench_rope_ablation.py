#!/usr/bin/env python3
"""
Plot LongBench ablation bars for RoPE-on-K experiments.

Output:
  - data/fig/longbench_rope_ablation.png
  - data/fig/longbench_rope_ablation.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_longbench_rope_ablation(out_dir: Path) -> None:
    # Reported scores from user.
    original_score = 46.06
    no_rope_k_score = 3.55
    ab8_score = 42.89
    ab8_no_beacon_k_rope_score = 41.56
    ab8_no_raw_k_rope_score = 5.00

    labels = [
        "Original\nModel",
        "No RoPE on K\n(No AB)",
        "AB 8x\n(Standard)",
        "AB 8x\n(No Beacon-K RoPE)",
        "AB 8x\n(No Raw-K RoPE)",
    ]
    values = [
        original_score,
        no_rope_k_score,
        ab8_score,
        ab8_no_beacon_k_rope_score,
        ab8_no_raw_k_rope_score,
    ]

    # Clustered x positions to emphasize the comparison logic.
    # Group 1: no AB baseline. Group 2: AB 8x family.
    x = [0.0, 1.0, 3.0, 4.0, 5.0]

    # Colorblind-safe scientific palette with semantic emphasis.
    colors = [
        "#1f3b73",  # deep blue: baseline
        "#c43c39",  # red: severe degradation
        "#1b9e77",  # teal-green: AB standard
        "#4daf4a",  # green: mild degradation
        "#e76f51",  # orange-red: severe degradation
    ]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6.8))

    bars = ax.bar(
        x,
        values,
        width=0.72,
        color=colors,
        edgecolor="#202020",
        linewidth=1.1,
        zorder=3,
    )

    # Divider between no-AB and AB experiments.
    ax.axvline(2.0, color="#7f7f7f", linestyle="--", linewidth=1.2, alpha=0.9, zorder=2)

    # Group labels.
    ax.text(0.5, 50.0, "No Compression (No AB)", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.text(4.0, 50.0, "Activation Beacon (8x Compression)", ha="center", va="bottom", fontsize=12, fontweight="bold")

    # Numeric labels.
    for rect, val in zip(bars, values):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 0.85,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    # Delta annotations for interpretability.
    delta_no_ab = no_rope_k_score - original_score
    delta_ab_beacon = ab8_no_beacon_k_rope_score - ab8_score
    delta_ab_raw = ab8_no_raw_k_rope_score - ab8_score

    ax.text(1.0, 8.0, f"Δ vs Original: {delta_no_ab:+.2f}", ha="center", va="center", fontsize=10, color="#4d4d4d")
    ax.text(4.0, 44.8, f"Δ vs AB Std: {delta_ab_beacon:+.2f}", ha="center", va="center", fontsize=10, color="#4d4d4d")
    ax.text(5.0, 9.0, f"Δ vs AB Std: {delta_ab_raw:+.2f}", ha="center", va="center", fontsize=10, color="#4d4d4d")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("LongBench Score (Higher is better)", fontsize=12)
    ax.set_title("LongBench RoPE-on-K Ablation (Llama3.1-8B)", fontsize=14, fontweight="bold", pad=10)
    ax.set_ylim(0, 52)
    ax.grid(axis="y", color="#d0d0d0", linestyle="-", linewidth=0.8, alpha=0.9, zorder=0)
    ax.grid(axis="x", visible=False)

    # Minimal frame style for paper-quality look.
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#4d4d4d")
    ax.spines["bottom"].set_color("#4d4d4d")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "longbench_rope_ablation.png"
    out_pdf = out_dir / "longbench_rope_ablation.pdf"
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot LongBench RoPE ablation bar chart.")
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("data/fig"),
        help="Directory to save figures (default: data/fig).",
    )
    args = parser.parse_args()
    plot_longbench_rope_ablation(args.out_dir)


if __name__ == "__main__":
    main()
