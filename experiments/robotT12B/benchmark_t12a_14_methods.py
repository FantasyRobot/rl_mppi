#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)


@contextmanager
def _pushd(path: str):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


@dataclass
class TrialMetrics:
    method: str
    trial_index: int
    success: bool
    steps: int
    total_plan_time_ms: float
    mean_plan_time_ms: float
    std_plan_time_ms: float
    max_plan_time_ms: float
    final_dist: float
    mean_dist: float
    rms_dist: float
    min_dist: float
    path_length: float
    max_abs_qvel: float
    rms_qvel: float
    max_abs_qacc: float
    rms_qacc: float
    max_abs_qjerk: float
    rms_qjerk: float


def _parse_args() -> argparse.Namespace:
    default_xml = os.path.join(_THIS_DIR, "urdf", "t12a_14_normal.xml")
    default_sac = os.path.join(_THIS_DIR, "models", "sac_t12a_14_model.pth")
    default_out = os.path.join(_ROOT_DIR, "experiments", "results", "t12a_14_benchmark.npz")

    p = argparse.ArgumentParser(
        description=(
            "Benchmark MPPI, SAC, and RL-MPPI on t12a_14 with planning time, "
            "goal-tracking error, and joint impact metrics."
        )
    )
    p.add_argument("--xml", type=str, default=default_xml)
    p.add_argument("--goal_site", type=str, default="goal")
    p.add_argument("--eef_site", type=str, default="end_effector")
    p.add_argument("--sac_model", type=str, default=default_sac)

    p.add_argument("--num_trials", type=int, default=20, help="Number of trials to evaluate")
    p.add_argument("--steps", type=int, default=400, help="Max rollout length per trial")
    p.add_argument("--action_repeat", type=int, default=5)
    p.add_argument("--tol", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--start_margin", type=float, default=0.10, help="Avoid joint limits by this fraction of ctrlrange")
    p.add_argument(
        "--init_qpos",
        type=float,
        nargs="+",
        default=None,
        help="Optional fixed initial qpos. If set, the same start is used for all trials.",
    )

    p.add_argument("--horizon", type=int, default=25)
    p.add_argument("--num_samples", type=int, default=96)
    p.add_argument("--lambda_coeff", type=float, default=1.0)
    p.add_argument("--noise_std", type=float, default=0.06)
    p.add_argument("--pos_cost", type=float, default=200.0)
    p.add_argument("--action_cost", type=float, default=0.02)
    p.add_argument("--smooth_cost", type=float, default=0.2)

    p.add_argument("--save_npz", type=str, default=default_out)
    p.add_argument("--save_json", type=str, default="", help="Optional summary JSON path")
    p.add_argument("--save_summary_csv", type=str, default="", help="Optional summary CSV path")
    p.add_argument("--save_trials_csv", type=str, default="", help="Optional per-trial CSV path")
    return p.parse_args()


def _resolve_out_path(path: str, *, default_dir: str) -> str:
    path = os.path.expanduser(os.path.expandvars(str(path)))
    if not os.path.isabs(path) and os.path.dirname(path) == "":
        path = os.path.join(default_dir, path)
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    return path


def _derive_sidecar(path: str, suffix: str, ext: str) -> str:
    root, _ = os.path.splitext(str(path))
    if ext and not ext.startswith("."):
        ext = f".{ext}"
    return f"{root}{suffix}{ext}"


def _effective_dt(env) -> float:
    return float(env.model.opt.timestep) * float(env.action_repeat)


def _sample_qpos(rng: np.random.Generator, env, *, margin: float) -> np.ndarray:
    lo = np.asarray(env.ctrl_min, dtype=np.float64)
    hi = np.asarray(env.ctrl_max, dtype=np.float64)
    span = hi - lo
    lo2 = lo + float(margin) * span
    hi2 = hi - float(margin) * span
    return rng.uniform(lo2, hi2).astype(np.float64)


def _safe_mean(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.mean(x))


def _safe_std(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.std(x))


def _safe_max_abs(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.max(np.abs(x)))


def _safe_rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x))))


def _path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    diff = np.diff(points, axis=0)
    return float(np.sum(np.linalg.norm(diff, axis=1)))


def _summarize_trial(
    method: str,
    trial_index: int,
    plan_times_ms,
    dist_traj,
    eef_traj,
    qvel_traj,
    qacc_traj,
    qjerk_traj,
    success: bool,
) -> TrialMetrics:
    plan_arr = np.asarray(plan_times_ms, dtype=np.float64)
    dist_arr = np.asarray(dist_traj, dtype=np.float64)
    eef_arr = np.asarray(eef_traj, dtype=np.float64)
    qvel_arr = np.asarray(qvel_traj, dtype=np.float64)
    qacc_arr = np.asarray(qacc_traj, dtype=np.float64)
    qjerk_arr = np.asarray(qjerk_traj, dtype=np.float64)

    final_dist = float(dist_arr[-1]) if dist_arr.size else float("inf")
    min_dist = float(np.min(dist_arr)) if dist_arr.size else float("inf")

    return TrialMetrics(
        method=method,
        trial_index=int(trial_index),
        success=bool(success),
        steps=int(dist_arr.size),
        total_plan_time_ms=float(np.sum(plan_arr)),
        mean_plan_time_ms=_safe_mean(plan_arr),
        std_plan_time_ms=_safe_std(plan_arr),
        max_plan_time_ms=float(np.max(plan_arr)) if plan_arr.size else 0.0,
        final_dist=final_dist,
        mean_dist=_safe_mean(dist_arr),
        rms_dist=_safe_rms(dist_arr),
        min_dist=min_dist,
        path_length=_path_length(eef_arr),
        max_abs_qvel=_safe_max_abs(qvel_arr),
        rms_qvel=_safe_rms(qvel_arr),
        max_abs_qacc=_safe_max_abs(qacc_arr),
        rms_qacc=_safe_rms(qacc_arr),
        max_abs_qjerk=_safe_max_abs(qjerk_arr),
        rms_qjerk=_safe_rms(qjerk_arr),
    )


def _trial_to_row(trial: TrialMetrics) -> dict[str, float | int | str]:
    return {
        "method": trial.method,
        "trial_index": int(trial.trial_index),
        "success": int(trial.success),
        "steps": int(trial.steps),
        "total_plan_time_ms": float(trial.total_plan_time_ms),
        "mean_plan_time_ms": float(trial.mean_plan_time_ms),
        "std_plan_time_ms": float(trial.std_plan_time_ms),
        "max_plan_time_ms": float(trial.max_plan_time_ms),
        "final_dist": float(trial.final_dist),
        "mean_dist": float(trial.mean_dist),
        "rms_dist": float(trial.rms_dist),
        "min_dist": float(trial.min_dist),
        "path_length": float(trial.path_length),
        "max_abs_qvel": float(trial.max_abs_qvel),
        "rms_qvel": float(trial.rms_qvel),
        "max_abs_qacc": float(trial.max_abs_qacc),
        "rms_qacc": float(trial.rms_qacc),
        "max_abs_qjerk": float(trial.max_abs_qjerk),
        "rms_qjerk": float(trial.rms_qjerk),
    }


def _aggregate_trials(method: str, trials: list[TrialMetrics]) -> dict[str, float | int | str]:
    rows = [_trial_to_row(x) for x in trials]
    numeric_keys = [
        "steps",
        "total_plan_time_ms",
        "mean_plan_time_ms",
        "std_plan_time_ms",
        "max_plan_time_ms",
        "final_dist",
        "mean_dist",
        "rms_dist",
        "min_dist",
        "path_length",
        "max_abs_qvel",
        "rms_qvel",
        "max_abs_qacc",
        "rms_qacc",
        "max_abs_qjerk",
        "rms_qjerk",
    ]

    out: dict[str, float | int | str] = {
        "method": method,
        "num_trials": int(len(trials)),
        "success_rate": float(np.mean([1.0 if t.success else 0.0 for t in trials])) if trials else 0.0,
    }

    for key in numeric_keys:
        arr = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        out[f"{key}_mean"] = _safe_mean(arr)
        out[f"{key}_std"] = _safe_std(arr)

    # Calculate pooled planning time statistics (across all steps of all trials)
    # Reconstruct sum and sum_sq from mean/std/count to perform pooled variance calculation.
    total_steps = 0
    total_plan_sum = 0.0
    total_plan_sq_sum = 0.0

    for t in trials:
        n = t.steps
        if n > 0:
            total_steps += n
            # Sum of X
            total_plan_sum += t.mean_plan_time_ms * n
            # Sum of X^2. Since Var = E[X^2] - (E[X])^2 => X_sq_sum = n * (var + mean^2) = n * (std^2 + mean^2)
            total_plan_sq_sum += n * (t.std_plan_time_ms**2 + t.mean_plan_time_ms**2)

    if total_steps > 0:
        pooled_mean = total_plan_sum / total_steps
        # Var = E[X^2] - (E[X])^2
        pooled_var = (total_plan_sq_sum / total_steps) - (pooled_mean**2)
        # Numerical stability clip
        if pooled_var < 0:
            pooled_var = 0.0
        pooled_std = float(np.sqrt(pooled_var))
        out["pooled_plan_mean"] = float(pooled_mean)
        out["pooled_plan_std"] = pooled_std
    else:
        out["pooled_plan_mean"] = 0.0
        out["pooled_plan_std"] = 0.0

    return out


def _write_csv(path: str, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_summary_table(summary_rows: list[dict[str, float | int | str]]) -> None:
    print("\n=== Benchmark Summary ===")
    # Define headers and widths
    # Format: (Header Name, Width)
    cols = [
        ("Method", 10),
        ("Succ(%)", 8),
        ("Plan(ms)", 20),
        ("FinalErr", 20),
        ("RMSErr", 20),
        ("Max|qvel|", 12),
        ("Max|qacc|", 12),
        ("Max|jerk|", 12),
    ]

    header_str = " ".join([f"{{:>{w}s}}" if i > 0 else f"{{:<{w}s}}" for i, (_, w) in enumerate(cols)])
    print(header_str.format(*[c[0] for c in cols]))

    for row in summary_rows:
        method = str(row["method"])
        succ = "{:.1f}".format(100.0 * float(row["success_rate"]))

        def fmt_stat(key_base, prec=3):
            m = float(row[f"{key_base}_mean"])
            s = float(row[f"{key_base}_std"])
            return f"{m:.{prec}f} ± {s:.{prec}f}"

        # Use pooled statistics for planning time to reflect per-step variability
        plan_mean = float(row.get("pooled_plan_mean", 0.0))
        plan_std = float(row.get("pooled_plan_std", 0.0))
        plan = f"{plan_mean:.2f} ± {plan_std:.2f}"

        ferr = fmt_stat("final_dist", prec=4)
        rerr = fmt_stat("rms_dist", prec=4)

        # For kinematic limits, we just show the mean of the maxes for brevity, or we can expand.
        # Given the space, I'll stick to mean for these unless requested otherwise.
        qvel = "{:.4f}".format(float(row["max_abs_qvel_mean"]))
        qacc = "{:.4f}".format(float(row["max_abs_qacc_mean"]))
        qjerk = "{:.4f}".format(float(row["max_abs_qjerk_mean"]))

        print(
            "{:<10s} {:>8s} {:>20s} {:>20s} {:>20s} {:>12s} {:>12s} {:>12s}".format(
                method, succ, plan, ferr, rerr, qvel, qacc, qjerk
            )
        )


def main() -> None:
    args = _parse_args()

    try:
        import mujoco
    except ModuleNotFoundError:
        raise SystemExit("Install mujoco first: pip install mujoco")

    from algorithms.mppi.mppi_mujoco_arm import MuJoCoArmMPPI
    from algorithms.rl_mppi.rl_mppi_mujoco_arm import RLMuJoCoArmMPPI, load_sac_policy as load_rl_policy
    from env.envmujoco_t12a_14 import T12A14MuJoCoEnv
    from test_sac_t12a_14 import load_sac_policy as load_sac_agent

    xml_path = os.path.abspath(os.path.expandvars(os.path.expanduser(str(args.xml))))
    if not os.path.exists(xml_path):
        raise SystemExit(f"XML not found: {xml_path}")

    sac_model = os.path.abspath(os.path.expandvars(os.path.expanduser(str(args.sac_model))))
    if not os.path.exists(sac_model):
        raise SystemExit(f"SAC model not found: {sac_model}")

    xml_dir = os.path.dirname(xml_path)
    with _pushd(xml_dir):
        model_for_ctrl = mujoco.MjModel.from_xml_path(xml_path)

    env_mppi = T12A14MuJoCoEnv(
        xml_path=xml_path,
        goal_site=str(args.goal_site),
        eef_site=str(args.eef_site),
        max_steps=int(args.steps),
        reach_tol=float(args.tol),
        action_repeat=int(args.action_repeat),
    )
    env_sac = T12A14MuJoCoEnv(
        xml_path=xml_path,
        goal_site=str(args.goal_site),
        eef_site=str(args.eef_site),
        max_steps=int(args.steps),
        reach_tol=float(args.tol),
        action_repeat=int(args.action_repeat),
    )
    env_rl = T12A14MuJoCoEnv(
        xml_path=xml_path,
        goal_site=str(args.goal_site),
        eef_site=str(args.eef_site),
        max_steps=int(args.steps),
        reach_tol=float(args.tol),
        action_repeat=int(args.action_repeat),
    )

    mppi = MuJoCoArmMPPI(
        model_for_ctrl,
        eef_site=str(args.eef_site),
        horizon=int(args.horizon),
        num_samples=int(args.num_samples),
        lambda_coeff=float(args.lambda_coeff),
        noise_std=float(args.noise_std),
        pos_cost_coeff=float(args.pos_cost),
        action_cost_coeff=float(args.action_cost),
        smooth_cost_coeff=float(args.smooth_cost),
        seed=int(args.seed),
    )

    sac_agent, _ = load_sac_agent(sac_model)
    rl_policy = load_rl_policy(sac_model)
    rl_mppi = RLMuJoCoArmMPPI(
        model_for_ctrl,
        rl_policy,
        eef_site=str(args.eef_site),
        goal_site=str(args.goal_site),
        horizon=int(args.horizon),
        num_samples=int(args.num_samples),
        lambda_coeff=float(args.lambda_coeff),
        noise_std=float(args.noise_std),
        pos_cost_coeff=float(args.pos_cost),
        action_cost_coeff=float(args.action_cost),
        smooth_cost_coeff=float(args.smooth_cost),
        seed=int(args.seed),
    )

    dt = _effective_dt(env_mppi)
    rng = np.random.default_rng(int(args.seed))

    if args.init_qpos is not None:
        init_qpos = np.asarray(args.init_qpos, dtype=np.float64)
        if init_qpos.shape[0] != int(env_mppi.nu):
            raise SystemExit(
                f"init_qpos length mismatch: got {init_qpos.shape[0]}, expected {int(env_mppi.nu)}"
            )
        starts = [np.clip(init_qpos, env_mppi.ctrl_min, env_mppi.ctrl_max).astype(np.float64) for _ in range(int(args.num_trials))]
    else:
        starts = [_sample_qpos(rng, env_mppi, margin=float(args.start_margin)) for _ in range(int(args.num_trials))]

    trials_by_method: dict[str, list[TrialMetrics]] = {"mppi": [], "sac": [], "rl_mppi": []}

    def run_mppi_trial(trial_index: int, q0: np.ndarray) -> TrialMetrics:
        mppi.reset()
        env_mppi.reset(init_qpos=q0)
        prev_qvel = np.asarray(env_mppi.data.qvel[: env_mppi.nu], dtype=np.float64).copy()
        prev_qacc = np.zeros_like(prev_qvel)

        plan_times_ms = []
        dist_traj = []
        eef_traj = []
        qvel_traj = []
        qacc_traj = []
        qjerk_traj = []
        success = False

        for _ in range(int(args.steps)):
            t0 = time.perf_counter()
            ctrl = mppi.get_action(env_mppi.data.qpos, env_mppi.data.qvel, env_mppi.goal_pos)
            plan_times_ms.append((time.perf_counter() - t0) * 1000.0)

            _obs, _reward, done, info = env_mppi.step(env_mppi.ctrl_to_action(ctrl))
            qvel_now = np.asarray(env_mppi.data.qvel[: env_mppi.nu], dtype=np.float64).copy()
            qacc_now = (qvel_now - prev_qvel) / (dt if dt > 0.0 else 1.0)
            qjerk_now = (qacc_now - prev_qacc) / (dt if dt > 0.0 else 1.0)
            prev_qvel = qvel_now
            prev_qacc = qacc_now

            dist_traj.append(float(info["dist"]))
            eef_traj.append(np.asarray(info["eef"], dtype=np.float64))
            qvel_traj.append(qvel_now)
            qacc_traj.append(qacc_now)
            qjerk_traj.append(qjerk_now)
            success = bool(info.get("success", False))
            if done:
                break

        return _summarize_trial("mppi", trial_index, plan_times_ms, dist_traj, eef_traj, qvel_traj, qacc_traj, qjerk_traj, success)

    def run_sac_trial(trial_index: int, q0: np.ndarray) -> TrialMetrics:
        obs = env_sac.reset(init_qpos=q0)
        prev_qvel = np.asarray(env_sac.data.qvel[: env_sac.nu], dtype=np.float64).copy()
        prev_qacc = np.zeros_like(prev_qvel)

        plan_times_ms = []
        dist_traj = []
        eef_traj = []
        qvel_traj = []
        qacc_traj = []
        qjerk_traj = []
        success = False

        for _ in range(int(args.steps)):
            t0 = time.perf_counter()
            action = sac_agent.select_action(obs, evaluate=True)
            plan_times_ms.append((time.perf_counter() - t0) * 1000.0)

            obs, _reward, done, info = env_sac.step(action)
            qvel_now = np.asarray(env_sac.data.qvel[: env_sac.nu], dtype=np.float64).copy()
            qacc_now = (qvel_now - prev_qvel) / (dt if dt > 0.0 else 1.0)
            qjerk_now = (qacc_now - prev_qacc) / (dt if dt > 0.0 else 1.0)
            prev_qvel = qvel_now
            prev_qacc = qacc_now

            dist_traj.append(float(info["dist"]))
            eef_traj.append(np.asarray(info["eef"], dtype=np.float64))
            qvel_traj.append(qvel_now)
            qacc_traj.append(qacc_now)
            qjerk_traj.append(qjerk_now)
            success = bool(info.get("success", False))
            if done:
                break

        return _summarize_trial("sac", trial_index, plan_times_ms, dist_traj, eef_traj, qvel_traj, qacc_traj, qjerk_traj, success)

    def run_rl_mppi_trial(trial_index: int, q0: np.ndarray) -> TrialMetrics:
        rl_mppi.reset()
        env_rl.reset(init_qpos=q0)
        prev_qvel = np.asarray(env_rl.data.qvel[: env_rl.nu], dtype=np.float64).copy()
        prev_qacc = np.zeros_like(prev_qvel)

        plan_times_ms = []
        dist_traj = []
        eef_traj = []
        qvel_traj = []
        qacc_traj = []
        qjerk_traj = []
        success = False

        for _ in range(int(args.steps)):
            t0 = time.perf_counter()
            ctrl = rl_mppi.get_action(env_rl.data.qpos, env_rl.data.qvel)
            plan_times_ms.append((time.perf_counter() - t0) * 1000.0)

            _obs, _reward, done, info = env_rl.step(env_rl.ctrl_to_action(ctrl))
            qvel_now = np.asarray(env_rl.data.qvel[: env_rl.nu], dtype=np.float64).copy()
            qacc_now = (qvel_now - prev_qvel) / (dt if dt > 0.0 else 1.0)
            qjerk_now = (qacc_now - prev_qacc) / (dt if dt > 0.0 else 1.0)
            prev_qvel = qvel_now
            prev_qacc = qacc_now

            dist_traj.append(float(info["dist"]))
            eef_traj.append(np.asarray(info["eef"], dtype=np.float64))
            qvel_traj.append(qvel_now)
            qacc_traj.append(qacc_now)
            qjerk_traj.append(qjerk_now)
            success = bool(info.get("success", False))
            if done:
                break

        return _summarize_trial("rl_mppi", trial_index, plan_times_ms, dist_traj, eef_traj, qvel_traj, qacc_traj, qjerk_traj, success)

    for trial_index, q0 in enumerate(starts):
        trials_by_method["mppi"].append(run_mppi_trial(trial_index, q0))
        trials_by_method["sac"].append(run_sac_trial(trial_index, q0))
        trials_by_method["rl_mppi"].append(run_rl_mppi_trial(trial_index, q0))

        if (trial_index + 1) % 5 == 0 or (trial_index + 1) == len(starts):
            print(f"[progress] finished {trial_index + 1}/{len(starts)} trials")

    summary_rows = [
        _aggregate_trials("mppi", trials_by_method["mppi"]),
        _aggregate_trials("sac", trials_by_method["sac"]),
        _aggregate_trials("rl_mppi", trials_by_method["rl_mppi"]),
    ]
    _print_summary_table(summary_rows)

    save_npz = _resolve_out_path(str(args.save_npz), default_dir=os.path.join(_ROOT_DIR, "experiments", "results"))
    save_json = _resolve_out_path(
        str(args.save_json) if str(args.save_json).strip() else _derive_sidecar(save_npz, "_summary", ".json"),
        default_dir=os.path.dirname(save_npz),
    )
    save_summary_csv = _resolve_out_path(
        str(args.save_summary_csv) if str(args.save_summary_csv).strip() else _derive_sidecar(save_npz, "_summary", ".csv"),
        default_dir=os.path.dirname(save_npz),
    )
    save_trials_csv = _resolve_out_path(
        str(args.save_trials_csv) if str(args.save_trials_csv).strip() else _derive_sidecar(save_npz, "_trials", ".csv"),
        default_dir=os.path.dirname(save_npz),
    )

    all_trial_rows: list[dict[str, float | int | str]] = []
    payload: dict[str, object] = {
        "starts": np.asarray(starts, dtype=np.float64),
        "dt": float(dt),
        "num_trials": int(args.num_trials),
        "steps": int(args.steps),
        "action_repeat": int(args.action_repeat),
        "tol": float(args.tol),
        "horizon": int(args.horizon),
        "num_samples": int(args.num_samples),
        "lambda_coeff": float(args.lambda_coeff),
        "noise_std": float(args.noise_std),
        "pos_cost": float(args.pos_cost),
        "action_cost": float(args.action_cost),
        "smooth_cost": float(args.smooth_cost),
    }

    for method, trials in trials_by_method.items():
        method_rows = [_trial_to_row(x) for x in trials]
        all_trial_rows.extend(method_rows)
        for key in method_rows[0].keys():
            if key == "method":
                continue
            payload[f"{method}_{key}"] = np.asarray([row[key] for row in method_rows])

    np.savez(save_npz, **payload)
    _write_csv(save_summary_csv, summary_rows)
    _write_csv(save_trials_csv, all_trial_rows)

    with open(save_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "xml": xml_path,
                    "sac_model": sac_model,
                    "num_trials": int(args.num_trials),
                    "steps": int(args.steps),
                    "action_repeat": int(args.action_repeat),
                    "tol": float(args.tol),
                    "seed": int(args.seed),
                    "start_margin": float(args.start_margin),
                    "horizon": int(args.horizon),
                    "num_samples": int(args.num_samples),
                    "lambda_coeff": float(args.lambda_coeff),
                    "noise_std": float(args.noise_std),
                    "pos_cost": float(args.pos_cost),
                    "action_cost": float(args.action_cost),
                    "smooth_cost": float(args.smooth_cost),
                },
                "summary": summary_rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\n[DONE] NPZ: {save_npz}")
    print(f"[DONE] JSON: {save_json}")
    print(f"[DONE] Summary CSV: {save_summary_csv}")
    print(f"[DONE] Trials CSV: {save_trials_csv}")


if __name__ == "__main__":
    main()
