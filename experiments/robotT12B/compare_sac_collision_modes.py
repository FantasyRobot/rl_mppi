#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

try:
    from scipy.io import savemat  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    savemat = None

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover
    plt = None

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)


def _resolve_out_path(path: str, *, default_dir: str) -> str:
    path = os.path.expanduser(os.path.expandvars(str(path)))
    if not os.path.isabs(path) and os.path.dirname(path) == "":
        path = os.path.join(default_dir, path)
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    return path


def _load_log(npz_path: str) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _as_1d(x) -> np.ndarray:
    x = np.asarray(x)
    return x.reshape(-1)


def _save_mat(*, out_mat: str, log_a: dict, label_a: str, log_b: dict, label_b: str) -> None:
    if savemat is None:
        raise SystemExit(
            "scipy is required to export .mat files. Install it in your env:\n"
            "  pip install scipy"
        )

    payload: dict[str, object] = {
        "label_a": str(label_a),
        "label_b": str(label_b),
        # --- A ---
        "a_episode_end_step": _as_1d(log_a.get("episode_end_step", [])),
        "a_episode_return": _as_1d(log_a.get("episode_return", [])),
        "a_eval_step": _as_1d(log_a.get("eval_step", [])),
        "a_eval_mean_dist": _as_1d(log_a.get("eval_mean_dist", [])),
        "a_eval_std_dist": _as_1d(log_a.get("eval_std_dist", [])),
        "a_eval_success_rate": _as_1d(log_a.get("eval_success_rate", [])),
        "a_eval_success_no_collision_rate": _as_1d(log_a.get("eval_success_no_collision_rate", [])),
        "a_eval_collision_rate": _as_1d(log_a.get("eval_collision_rate", [])),
        "a_eval_alpha": _as_1d(log_a.get("eval_alpha", [])),
        # --- B ---
        "b_episode_end_step": _as_1d(log_b.get("episode_end_step", [])),
        "b_episode_return": _as_1d(log_b.get("episode_return", [])),
        "b_eval_step": _as_1d(log_b.get("eval_step", [])),
        "b_eval_mean_dist": _as_1d(log_b.get("eval_mean_dist", [])),
        "b_eval_std_dist": _as_1d(log_b.get("eval_std_dist", [])),
        "b_eval_success_rate": _as_1d(log_b.get("eval_success_rate", [])),
        "b_eval_success_no_collision_rate": _as_1d(log_b.get("eval_success_no_collision_rate", [])),
        "b_eval_collision_rate": _as_1d(log_b.get("eval_collision_rate", [])),
        "b_eval_alpha": _as_1d(log_b.get("eval_alpha", [])),
    }

    # Optional meta (kept as strings/numbers when possible)
    for prefix, log in [("a", log_a), ("b", log_b)]:
        for k in [
            "collision_mode",
            "terminate_on_collision",
            "collision_penalty",
            "cdf_sigma",
            "cdf_margin",
            "cdf_scale",
            "reach_tol",
            "max_ep_steps",
            "action_repeat",
            "seed",
            "total_steps",
            "obstacle_prefix",
            "xml_path",
        ]:
            if k not in log:
                continue
            v = log.get(k)
            if isinstance(v, np.ndarray) and v.shape == ():
                try:
                    v = v.item()
                except Exception:
                    pass
            if isinstance(v, bytes):
                try:
                    v = v.decode("utf-8")
                except Exception:
                    v = str(v)
            payload[f"{prefix}_meta_{k}"] = v

    savemat(out_mat, payload, do_compression=True)
    print("[MAT] saved:", out_mat)


def _plot_compare(*, log_a: dict, label_a: str, log_b: dict, label_b: str, out_png: str) -> None:
    if plt is None:
        raise SystemExit("matplotlib is required for plotting. Install with: pip install matplotlib")

    fig = plt.figure(figsize=(13, 9))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    # Episode return
    ax1.plot(log_a.get("episode_end_step", []), log_a.get("episode_return", []), label=label_a, linewidth=1.2)
    ax1.plot(log_b.get("episode_end_step", []), log_b.get("episode_return", []), label=label_b, linewidth=1.2)
    ax1.set_title("Episode return")
    ax1.set_xlabel("env step")
    ax1.set_ylabel("return")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    # Eval dist
    xa = np.asarray(log_a.get("eval_step", []), dtype=np.float64)
    xb = np.asarray(log_b.get("eval_step", []), dtype=np.float64)
    ya = np.asarray(log_a.get("eval_mean_dist", []), dtype=np.float64)
    yb = np.asarray(log_b.get("eval_mean_dist", []), dtype=np.float64)
    sa = np.asarray(log_a.get("eval_std_dist", np.zeros_like(ya)), dtype=np.float64)
    sb = np.asarray(log_b.get("eval_std_dist", np.zeros_like(yb)), dtype=np.float64)

    ax2.plot(xa, ya, label=f"{label_a} mean", linewidth=1.6)
    if len(xa) == len(sa) and np.any(sa > 0):
        ax2.fill_between(xa, ya - sa, ya + sa, alpha=0.15)

    ax2.plot(xb, yb, label=f"{label_b} mean", linewidth=1.6)
    if len(xb) == len(sb) and np.any(sb > 0):
        ax2.fill_between(xb, yb - sb, yb + sb, alpha=0.15)

    ax2.set_title("Eval mean dist (±1 std)")
    ax2.set_xlabel("env step")
    ax2.set_ylabel("dist")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    # Rates
    ax3.plot(xa, np.asarray(log_a.get("eval_success_rate", [])) * 100.0, label=f"{label_a} success", linewidth=1.6)
    ax3.plot(
        xa,
        np.asarray(log_a.get("eval_success_no_collision_rate", [])) * 100.0,
        label=f"{label_a} success_no_coll",
        linewidth=1.6,
    )
    ax3.plot(xa, np.asarray(log_a.get("eval_collision_rate", [])) * 100.0, label=f"{label_a} collision", linewidth=1.6)

    ax3.plot(xb, np.asarray(log_b.get("eval_success_rate", [])) * 100.0, label=f"{label_b} success", linewidth=1.6)
    ax3.plot(
        xb,
        np.asarray(log_b.get("eval_success_no_collision_rate", [])) * 100.0,
        label=f"{label_b} success_no_coll",
        linewidth=1.6,
    )
    ax3.plot(xb, np.asarray(log_b.get("eval_collision_rate", [])) * 100.0, label=f"{label_b} collision", linewidth=1.6)

    ax3.set_title("Eval rates")
    ax3.set_xlabel("env step")
    ax3.set_ylabel("rate (%)")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="best", ncol=2, fontsize=9)

    # Alpha
    ax4.plot(xa, np.asarray(log_a.get("eval_alpha", [])), label=label_a, linewidth=1.6)
    ax4.plot(xb, np.asarray(log_b.get("eval_alpha", [])), label=label_b, linewidth=1.6)
    ax4.set_title("Alpha")
    ax4.set_xlabel("env step")
    ax4.set_ylabel("alpha")
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print("[PLOT] saved:", out_png)


def main() -> None:
    default_xml = os.path.join(_THIS_DIR, "urdf", "t12a_14_normal.xml")
    results_dir = os.path.join(_ROOT_DIR, "experiments", "results")

    p = argparse.ArgumentParser(description="Train+compare SAC collision modes: stop vs cdf")
    p.add_argument("--xml", type=str, default=default_xml)
    p.add_argument("--goal_site", type=str, default="goal")
    p.add_argument("--eef_site", type=str, default="end_effector")
    p.add_argument("--action_repeat", type=int, default=5)
    p.add_argument("--max_steps", type=int, default=2500)
    p.add_argument("--reach_tol", type=float, default=0.03)

    p.add_argument("--total_steps", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_every", type=int, default=20_000)

    p.add_argument("--collision_penalty", type=float, default=50.0)
    p.add_argument("--cdf_sigma", type=float, default=0.05)
    p.add_argument("--cdf_margin", type=float, default=0.0)
    p.add_argument("--cdf_scale", type=float, default=5.0)

    p.add_argument("--out_dir", type=str, default=results_dir)
    p.add_argument("--prefix", type=str, default="sac_collision_compare")
    p.add_argument(
        "--out_mat",
        type=str,
        default="",
        help="If set, export the compared curves into a .mat file for MATLAB plotting",
    )

    args = p.parse_args()

    from train_sac_t12a_14_online import train_sac_t12a_14_online

    out_dir = _resolve_out_path(str(args.out_dir), default_dir=results_dir)

    save_stop = os.path.join(out_dir, f"{args.prefix}_stop.pth")
    save_cdf = os.path.join(out_dir, f"{args.prefix}_cdf.pth")

    log_stop = os.path.splitext(save_stop)[0] + "_train_log.npz"
    log_cdf = os.path.splitext(save_cdf)[0] + "_train_log.npz"

    plot_stop = os.path.splitext(save_stop)[0] + "_train.png"
    plot_cdf = os.path.splitext(save_cdf)[0] + "_train.png"

    print("=== Train mode=stop (terminate on collision) ===")
    train_sac_t12a_14_online(
        xml_path=str(args.xml),
        goal_site=str(args.goal_site),
        eef_site=str(args.eef_site),
        save_path=str(save_stop),
        total_steps=int(args.total_steps),
        seed=int(args.seed),
        eval_every=int(args.eval_every),
        action_repeat=int(args.action_repeat),
        max_ep_steps=int(args.max_steps),
        reach_tol=float(args.reach_tol),
        collision_mode="stop",
        collision_penalty=float(args.collision_penalty),
        terminate_on_collision=True,
        log_path=str(log_stop),
        plot_path=str(plot_stop),
        show_plot=False,
    )

    print("\n=== Train mode=cdf (shaping, do NOT terminate on collision) ===")
    train_sac_t12a_14_online(
        xml_path=str(args.xml),
        goal_site=str(args.goal_site),
        eef_site=str(args.eef_site),
        save_path=str(save_cdf),
        total_steps=int(args.total_steps),
        seed=int(args.seed),
        eval_every=int(args.eval_every),
        action_repeat=int(args.action_repeat),
        max_ep_steps=int(args.max_steps),
        reach_tol=float(args.reach_tol),
        collision_mode="cdf",
        collision_penalty=float(args.collision_penalty),
        terminate_on_collision=False,
        cdf_sigma=float(args.cdf_sigma),
        cdf_margin=float(args.cdf_margin),
        cdf_scale=float(args.cdf_scale),
        log_path=str(log_cdf),
        plot_path=str(plot_cdf),
        show_plot=False,
    )

    out_png = os.path.join(out_dir, f"{args.prefix}_overlay.png")
    out_mat = str(args.out_mat).strip()
    if out_mat:
        out_mat = _resolve_out_path(out_mat, default_dir=out_dir)
    else:
        out_mat = os.path.join(out_dir, f"{args.prefix}_overlay.mat")

    log_a = _load_log(log_stop)
    log_b = _load_log(log_cdf)

    _save_mat(out_mat=out_mat, log_a=log_a, label_a="stop", log_b=log_b, label_b="cdf")
    _plot_compare(
        log_a=log_a,
        label_a="stop",
        log_b=log_b,
        label_b="cdf",
        out_png=out_png,
    )


if __name__ == "__main__":
    main()
