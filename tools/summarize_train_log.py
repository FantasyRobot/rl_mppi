import argparse
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np


def _as_1d_float(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    arr = np.asarray(x)
    if arr.size == 0:
        return arr.astype(float)
    return arr.reshape(-1).astype(float)


def _first_step_where(steps: np.ndarray, mask: np.ndarray) -> Optional[float]:
    if steps is None or mask is None:
        return None
    if len(steps) == 0:
        return None
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return None
    return float(steps[int(idx[0])])


def _maybe_item_dict(v: Any) -> Optional[Dict[str, Any]]:
    try:
        if isinstance(v, np.ndarray) and v.shape == ():
            v = v.item()
        if isinstance(v, dict):
            return v
    except Exception:
        return None
    return None


def _pick(meta: Dict[str, Any], key: str) -> Any:
    if key in meta:
        return meta[key]
    # tolerate different naming conventions
    alts = {
        "violation_tol": ["violation_tol", "constraint_violation_tol"],
        "violation_agg": ["violation_agg", "constraint_violation_agg"],
        "tdcd_tau_c": ["tdcd_tau_c", "tau_c"],
        "constraint_discount_use_amount": ["constraint_discount_use_amount"],
        "seed": ["seed", "random_seed"],
    }
    for alt in alts.get(key, []):
        if alt in meta:
            return meta[alt]
    return None


def summarize(npz_path: str) -> str:
    if not os.path.exists(npz_path):
        raise FileNotFoundError(npz_path)

    with np.load(npz_path, allow_pickle=True) as data:
        keys = set(data.files)

        # Eval series (preferred)
        eval_steps = _as_1d_float(data["eval_steps"]) if "eval_steps" in keys else None
        eval_success = _as_1d_float(data["eval_success_rate"]) if "eval_success_rate" in keys else None
        eval_collision = _as_1d_float(data["eval_collision_rate"]) if "eval_collision_rate" in keys else None
        eval_mean_dist = _as_1d_float(data["eval_mean_dist"]) if "eval_mean_dist" in keys else None
        eval_alpha = _as_1d_float(data["eval_alpha"]) if "eval_alpha" in keys else None

        # Training series (optional)
        steps = _as_1d_float(data["steps"]) if "steps" in keys else None
        ep_return = _as_1d_float(data["episode_return"]) if "episode_return" in keys else None
        ep_len = _as_1d_float(data["episode_len"]) if "episode_len" in keys else None

        meta = None
        for mk in ("meta", "config", "args"):
            if mk in keys:
                meta = _maybe_item_dict(data[mk])
                if meta is not None:
                    break
        if meta is None:
            meta = {}

    lines = []
    lines.append(f"npz: {npz_path}")

    def _fmt(v: Any) -> str:
        if v is None:
            return "-"
        if isinstance(v, (float, np.floating)):
            return f"{float(v):.6g}"
        return str(v)

    # Meta (paper-relevant)
    if meta:
        lines.append(
            "meta: "
            + ", ".join(
                [
                    f"violation_agg={_fmt(_pick(meta, 'violation_agg'))}",
                    f"violation_tol={_fmt(_pick(meta, 'violation_tol'))}",
                    f"constraint_discount_use_amount={_fmt(_pick(meta, 'constraint_discount_use_amount'))}",
                    f"tdcd_tau_c={_fmt(_pick(meta, 'tdcd_tau_c'))}",
                    f"seed={_fmt(_pick(meta, 'seed'))}",
                ]
            )
        )

    # Eval summary
    if eval_steps is not None and eval_steps.size > 0:
        lines.append("eval:")

        if eval_success is not None:
            lines.append(f"  final success={eval_success[-1]:.4f}, best success={np.nanmax(eval_success):.4f}")
            step_100 = _first_step_where(eval_steps, eval_success >= 1.0 - 1e-12)
            lines.append(f"  step_to_success_100={_fmt(step_100)}")

        if eval_collision is not None:
            lines.append(
                f"  final collision={eval_collision[-1]:.4f}, best(min) collision={np.nanmin(eval_collision):.4f}"
            )
            step_col_0 = _first_step_where(eval_steps, eval_collision <= 1e-12)
            lines.append(f"  step_to_collision_0={_fmt(step_col_0)}")

        if eval_mean_dist is not None:
            lines.append(
                f"  final mean_dist={eval_mean_dist[-1]:.6f}, best(min) mean_dist={np.nanmin(eval_mean_dist):.6f}"
            )
            for thr in (0.03, 0.02, 0.01):
                st = _first_step_where(eval_steps, eval_mean_dist <= thr)
                lines.append(f"  step_to_mean_dist<={thr:g}={_fmt(st)}")

        if eval_alpha is not None:
            lines.append(f"  final alpha={eval_alpha[-1]:.6g}")

    else:
        lines.append("eval: (missing eval_steps; cannot compute eval milestones)")

    # Training tail stats (optional)
    if steps is not None and steps.size > 0:
        lines.append("train:")
        lines.append(f"  total_steps={int(steps[-1])}")
        if ep_return is not None and ep_return.size > 0:
            lines.append(f"  final episode_return={ep_return[-1]:.6g}, best={np.nanmax(ep_return):.6g}")
        if ep_len is not None and ep_len.size > 0:
            lines.append(f"  final episode_len={ep_len[-1]:.6g}")

    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--npz",
        default=r"experiments/robotT12B/models/cd_sac_t12a_14_model_online_train_log.npz",
        help="Path to training log .npz",
    )
    p.add_argument(
        "--out",
        default="",
        help="Optional output text file path",
    )
    args = p.parse_args()

    txt = summarize(args.npz)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(txt)
    else:
        print(txt, end="")


if __name__ == "__main__":
    main()
