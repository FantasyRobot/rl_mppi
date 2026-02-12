#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os

import numpy as np


def _plot_series(ax, ys: np.ndarray, label_prefix: str, max_dims: int | None = None) -> None:
    ys = np.asarray(ys)
    if ys.ndim == 1:
        ax.plot(ys, label=label_prefix)
        return

    if ys.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array, got shape={ys.shape}")

    d = ys.shape[1]
    if max_dims is not None:
        d = min(d, int(max_dims))

    for j in range(d):
        ax.plot(ys[:, j], label=f"{label_prefix}[{j}]")


def main() -> None:
    p = argparse.ArgumentParser(description="Plot curves from t12a_14_compare.npz (MPPI vs SAC vs RL-MPPI)")
    p.add_argument(
        "--npz",
        type=str,
        default=os.path.join("t12a_14_compare.npz"),
        help="Path to .npz produced by compare_t12a_14_methods.py",
    )
    p.add_argument("--max_joints", type=int, default=14, help="Max joints to plot for qpos/qvel/qacc")
    p.add_argument("--save_dir", type=str, default="", help="If set, save PNGs into this directory")
    p.add_argument("--no_show", action="store_true", help="Do not show interactive windows")
    args = p.parse_args()

    npz_path = os.path.abspath(os.path.expandvars(os.path.expanduser(str(args.npz))))
    if not os.path.exists(npz_path):
        raise SystemExit(f"NPZ not found: {npz_path}")

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        raise SystemExit(
            "matplotlib is required for plotting. Install it in your env:\n"
            "  pip install matplotlib"
        )

    data = np.load(npz_path)

    def g(key: str) -> np.ndarray:
        if key not in data:
            raise KeyError(f"Missing key in npz: {key}")
        return np.asarray(data[key])

    save_dir = str(args.save_dir).strip()
    if save_dir:
        save_dir = os.path.abspath(os.path.expandvars(os.path.expanduser(save_dir)))
        os.makedirs(save_dir, exist_ok=True)

    # --- dist ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(g("mppi_dist"), label="MPPI")
    ax.plot(g("sac_dist"), label="SAC")
    ax.plot(g("rlmppi_dist"), label="RL-MPPI")
    ax.set_title("EEF distance to goal")
    ax.set_xlabel("step")
    ax.set_ylabel("dist (m)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if save_dir:
        fig.savefig(os.path.join(save_dir, "compare_dist.png"), dpi=150)

    # --- reward ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(g("mppi_reward"), label="MPPI")
    ax.plot(g("sac_reward"), label="SAC")
    ax.plot(g("rlmppi_reward"), label="RL-MPPI")
    ax.set_title("Reward")
    ax.set_xlabel("step")
    ax.set_ylabel("reward")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if save_dir:
        fig.savefig(os.path.join(save_dir, "compare_reward.png"), dpi=150)

    # --- qpos/qvel/qacc ---
    for key, title, ylab in [
        ("qpos", "Joint position", "qpos"),
        ("qvel", "Joint velocity", "qvel"),
        ("qacc", "Joint acceleration (finite-diff)", "qacc"),
    ]:
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        for ax, algo, prefix in [
            (axes[0], "MPPI", "mppi"),
            (axes[1], "SAC", "sac"),
            (axes[2], "RL-MPPI", "rlmppi"),
        ]:
            ys = g(f"{prefix}_{key}")
            _plot_series(ax, ys, label_prefix=f"{algo} ", max_dims=int(args.max_joints))
            ax.set_ylabel(ylab)
            ax.set_title(f"{algo}: {title}")
            ax.grid(True, alpha=0.3)
            # Too many lines -> keep legend off by default for readability.
        axes[-1].set_xlabel("step")
        fig.tight_layout()
        if save_dir:
            fig.savefig(os.path.join(save_dir, f"compare_{key}.png"), dpi=150)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
