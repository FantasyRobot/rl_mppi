#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)


@dataclass
class TrialResult:
    reached: bool
    collided: bool
    success: bool
    steps: int
    final_dist: float


@contextmanager
def _pushd(path: str):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _resolve_out_path(path: str, *, default_dir: str) -> str:
    path = os.path.expanduser(os.path.expandvars(str(path)))
    if not os.path.isabs(path) and os.path.dirname(path) == "":
        path = os.path.join(default_dir, path)
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    return path


def _obstacle_geom_ids(mujoco, model, *, name_prefix: str = "obstacle") -> set[int]:
    out: set[int] = set()
    for gid in range(int(model.ngeom)):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(gid))
        if not name:
            continue
        if str(name).startswith(str(name_prefix)):
            out.add(int(gid))
    return out


def _has_obstacle_contact(env, obstacle_geoms: set[int]) -> bool:
    if not obstacle_geoms:
        return False

    data = env.data
    ncon = int(getattr(data, "ncon", 0))
    if ncon <= 0:
        return False

    for i in range(ncon):
        c = data.contact[i]
        g1 = int(getattr(c, "geom1", -1))
        g2 = int(getattr(c, "geom2", -1))
        if g1 in obstacle_geoms or g2 in obstacle_geoms:
            return True
    return False


def _set_env_qpos(env, qpos: np.ndarray) -> np.ndarray:
    # Reset counters and set state deterministically.
    if hasattr(env, "_step_count"):
        env._step_count = 0
    env.data.qpos[:] = 0.0
    env.data.qvel[:] = 0.0
    env.data.qpos[: int(env.nu)] = np.asarray(qpos, dtype=np.float64).reshape(int(env.nu))

    env._mujoco.mj_forward(env.model, env.data)

    # Update goal (site position).
    env.goal_pos = np.asarray(env.data.site_xpos[env.goal_sid], dtype=np.float64).copy()
    return env.get_obs()


def _sample_qpos(rng: np.random.Generator, env, *, margin: float) -> np.ndarray:
    lo = np.asarray(env.ctrl_min, dtype=np.float64)
    hi = np.asarray(env.ctrl_max, dtype=np.float64)
    span = hi - lo
    lo2 = lo + float(margin) * span
    hi2 = hi - float(margin) * span
    if np.any(hi2 <= lo2):
        lo2 = lo
        hi2 = hi
    return rng.uniform(lo2, hi2).astype(np.float64)


def _rollout_sac(*, env, sac_agent, obstacle_geoms: set[int], max_steps: int) -> TrialResult:
    obs = env.get_obs()
    collided = _has_obstacle_contact(env, obstacle_geoms)
    last_info: dict[str, Any] = {"success": False, "dist": float("inf"), "step": 0}

    for _ in range(int(max_steps)):
        a = sac_agent.select_action(obs, evaluate=True)
        obs, _r, done, info = env.step(a)
        last_info = dict(info)

        if _has_obstacle_contact(env, obstacle_geoms):
            collided = True
            break
        if bool(done):
            break

    reached = bool(last_info.get("success", False))
    steps = int(last_info.get("step", 0))
    final_dist = float(last_info.get("dist", float("inf")))
    success = bool(reached) and not bool(collided)
    return TrialResult(reached=reached, collided=collided, success=success, steps=steps, final_dist=final_dist)


def _rollout_mppi(*, env, mppi, obstacle_geoms: set[int], max_steps: int) -> TrialResult:
    collided = _has_obstacle_contact(env, obstacle_geoms)
    last_info: dict[str, Any] = {"success": False, "dist": float("inf"), "step": 0}

    for _ in range(int(max_steps)):
        u = mppi.get_action(env.data.qpos, env.data.qvel, env.goal_pos)
        a_norm = env.ctrl_to_action(u)
        _obs, _r, done, info = env.step(a_norm)
        last_info = dict(info)

        if _has_obstacle_contact(env, obstacle_geoms):
            collided = True
            break
        if bool(done):
            break

    reached = bool(last_info.get("success", False))
    steps = int(last_info.get("step", 0))
    final_dist = float(last_info.get("dist", float("inf")))
    success = bool(reached) and not bool(collided)
    return TrialResult(reached=reached, collided=collided, success=success, steps=steps, final_dist=final_dist)


def _rollout_rl_mppi(*, env, rl_mppi, obstacle_geoms: set[int], max_steps: int) -> TrialResult:
    collided = _has_obstacle_contact(env, obstacle_geoms)
    last_info: dict[str, Any] = {"success": False, "dist": float("inf"), "step": 0}

    for _ in range(int(max_steps)):
        u = rl_mppi.get_action(env.data.qpos, env.data.qvel)
        a_norm = env.ctrl_to_action(u)
        _obs, _r, done, info = env.step(a_norm)
        last_info = dict(info)

        if _has_obstacle_contact(env, obstacle_geoms):
            collided = True
            break
        if bool(done):
            break

    reached = bool(last_info.get("success", False))
    steps = int(last_info.get("step", 0))
    final_dist = float(last_info.get("dist", float("inf")))
    success = bool(reached) and not bool(collided)
    return TrialResult(reached=reached, collided=collided, success=success, steps=steps, final_dist=final_dist)


def main() -> None:
    default_xml = os.path.join(_THIS_DIR, "urdf", "t12a_14_dyn.xml")
    default_sac = os.path.join(_THIS_DIR, "models", "sac_t12a_14_model.pth")
    default_out = os.path.join(_ROOT_DIR, "experiments", "results", "t12a_14_random_starts_success.npz")

    p = argparse.ArgumentParser(
        description="Random-start evaluation for MPPI / SAC / RL-MPPI: success = reach goal AND no obstacle collision"
    )
    p.add_argument("--xml", type=str, default=default_xml)
    p.add_argument("--goal_site", type=str, default="goal")
    p.add_argument("--eef_site", type=str, default="end_effector")
    p.add_argument("--action_repeat", type=int, default=5)
    p.add_argument("--max_steps", type=int, default=400)
    p.add_argument("--tol", type=float, default=0.05)

    p.add_argument("--num_starts", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--start_margin", type=float, default=0.10, help="Avoid joint limits by this fraction of ctrlrange")
    p.add_argument("--max_resample", type=int, default=2000, help="Max attempts to find collision-free starts")

    p.add_argument("--sac_model", type=str, default=default_sac)
    p.add_argument("--out", type=str, default=default_out)
    p.add_argument("--save_json", type=str, default="", help="Optional: also save summary json next to npz")

    args = p.parse_args()

    try:
        import mujoco
    except ModuleNotFoundError:
        raise SystemExit("Install mujoco first: pip install mujoco")

    xml_path = os.path.abspath(os.path.expandvars(os.path.expanduser(str(args.xml))))
    if not os.path.exists(xml_path):
        raise SystemExit(f"XML not found: {xml_path}")

    from env.envmujoco_t12a_14 import T12A14MuJoCoEnv

    # Create envs (separate instance per method).
    env_mppi = T12A14MuJoCoEnv(
        xml_path=xml_path,
        goal_site=str(args.goal_site),
        eef_site=str(args.eef_site),
        max_steps=int(args.max_steps),
        reach_tol=float(args.tol),
        action_repeat=int(args.action_repeat),
    )
    env_sac = T12A14MuJoCoEnv(
        xml_path=xml_path,
        goal_site=str(args.goal_site),
        eef_site=str(args.eef_site),
        max_steps=int(args.max_steps),
        reach_tol=float(args.tol),
        action_repeat=int(args.action_repeat),
    )
    env_rl = T12A14MuJoCoEnv(
        xml_path=xml_path,
        goal_site=str(args.goal_site),
        eef_site=str(args.eef_site),
        max_steps=int(args.max_steps),
        reach_tol=float(args.tol),
        action_repeat=int(args.action_repeat),
    )

    obstacle_mppi = _obstacle_geom_ids(mujoco, env_mppi.model)
    obstacle_sac = _obstacle_geom_ids(mujoco, env_sac.model)
    obstacle_rl = _obstacle_geom_ids(mujoco, env_rl.model)

    if not obstacle_mppi:
        print("[WARN] No obstacle geoms found by prefix 'obstacle'. Collision check will always be false.")

    # Controllers
    # MPPI / RL-MPPI use a standalone model object.
    xml_dir = os.path.dirname(xml_path)
    with _pushd(xml_dir):
        model_for_ctrl = mujoco.MjModel.from_xml_path(xml_path)

    from algorithms.mppi.mppi_mujoco_arm import MuJoCoArmMPPI
    from algorithms.rl_mppi.rl_mppi_mujoco_arm import RLMuJoCoArmMPPI
    from algorithms.rl_mppi.rl_mppi_mujoco_arm import load_sac_policy as load_rl_mppi_policy

    # For SAC evaluation, reuse the existing loader used in other scripts in this folder.
    from test_sac_t12a_14 import load_sac_policy as load_sac_policy_eval

    if not os.path.exists(str(args.sac_model)):
        raise SystemExit(
            f"SAC model not found: {args.sac_model}\n"
            "Train it first via: python experiments/robotT12B/sac_t12a_14_cli.py train"
        )

    sac_agent, _ckpt = load_sac_policy_eval(str(args.sac_model))

    mppi = MuJoCoArmMPPI(model_for_ctrl, eef_site=str(args.eef_site))
    rl_policy = load_rl_mppi_policy(str(args.sac_model))
    rl_mppi = RLMuJoCoArmMPPI(
        model_for_ctrl,
        rl_policy,
        eef_site=str(args.eef_site),
        goal_site=str(args.goal_site),
    )

    rng = np.random.default_rng(int(args.seed))

    starts: list[np.ndarray] = []
    tries = 0
    while len(starts) < int(args.num_starts) and tries < int(args.max_resample):
        tries += 1
        q0 = _sample_qpos(rng, env_sac, margin=float(args.start_margin))
        _set_env_qpos(env_sac, q0)
        if _has_obstacle_contact(env_sac, obstacle_sac):
            continue
        starts.append(q0.astype(np.float64))

    if len(starts) < int(args.num_starts):
        print(f"[WARN] Only collected {len(starts)}/{int(args.num_starts)} collision-free starts (tries={tries}).")

    # Evaluate
    res_mppi: list[TrialResult] = []
    res_sac: list[TrialResult] = []
    res_rl: list[TrialResult] = []

    for i, q0 in enumerate(starts):
        # Reset controller internal sequences for a fair per-start evaluation.
        try:
            mppi.reset()
        except Exception:
            pass
        try:
            rl_mppi.reset()
        except Exception:
            pass

        _set_env_qpos(env_mppi, q0)
        _set_env_qpos(env_sac, q0)
        _set_env_qpos(env_rl, q0)

        r_mppi = _rollout_mppi(env=env_mppi, mppi=mppi, obstacle_geoms=obstacle_mppi, max_steps=int(args.max_steps))
        r_sac = _rollout_sac(env=env_sac, sac_agent=sac_agent, obstacle_geoms=obstacle_sac, max_steps=int(args.max_steps))
        r_rl = _rollout_rl_mppi(env=env_rl, rl_mppi=rl_mppi, obstacle_geoms=obstacle_rl, max_steps=int(args.max_steps))

        res_mppi.append(r_mppi)
        res_sac.append(r_sac)
        res_rl.append(r_rl)

        if (i + 1) % 10 == 0 or (i + 1) == len(starts):
            def _rate(xs: list[TrialResult]) -> float:
                return float(sum(1 for x in xs if x.success) / max(1, len(xs)))

            print(
                f"[{i+1:3d}/{len(starts):3d}] "
                f"success_rate: MPPI={_rate(res_mppi)*100:5.1f}% "
                f"SAC={_rate(res_sac)*100:5.1f}% "
                f"RL-MPPI={_rate(res_rl)*100:5.1f}%"
            )

    def _summ(xs: list[TrialResult]) -> dict[str, float]:
        n = max(1, len(xs))
        return {
            "n": float(len(xs)),
            "reached_rate": float(sum(1 for x in xs if x.reached) / n),
            "collision_rate": float(sum(1 for x in xs if x.collided) / n),
            "success_rate": float(sum(1 for x in xs if x.success) / n),
            "avg_steps": float(np.mean([x.steps for x in xs])) if xs else 0.0,
            "avg_final_dist": float(np.mean([x.final_dist for x in xs])) if xs else float("inf"),
        }

    summary = {
        "xml": xml_path,
        "goal_site": str(args.goal_site),
        "eef_site": str(args.eef_site),
        "action_repeat": int(args.action_repeat),
        "max_steps": int(args.max_steps),
        "tol": float(args.tol),
        "num_starts": int(len(starts)),
        "seed": int(args.seed),
        "start_margin": float(args.start_margin),
        "mppi": _summ(res_mppi),
        "sac": _summ(res_sac),
        "rl_mppi": _summ(res_rl),
    }

    out_path = _resolve_out_path(str(args.out), default_dir=os.path.join(_ROOT_DIR, "experiments", "results"))
    np.savez(
        out_path,
        qpos0=np.asarray(starts, dtype=np.float64),
        mppi_success=np.asarray([int(r.success) for r in res_mppi], dtype=np.int8),
        sac_success=np.asarray([int(r.success) for r in res_sac], dtype=np.int8),
        rlmppi_success=np.asarray([int(r.success) for r in res_rl], dtype=np.int8),
        mppi_reached=np.asarray([int(r.reached) for r in res_mppi], dtype=np.int8),
        sac_reached=np.asarray([int(r.reached) for r in res_sac], dtype=np.int8),
        rlmppi_reached=np.asarray([int(r.reached) for r in res_rl], dtype=np.int8),
        mppi_collided=np.asarray([int(r.collided) for r in res_mppi], dtype=np.int8),
        sac_collided=np.asarray([int(r.collided) for r in res_sac], dtype=np.int8),
        rlmppi_collided=np.asarray([int(r.collided) for r in res_rl], dtype=np.int8),
        mppi_steps=np.asarray([int(r.steps) for r in res_mppi], dtype=np.int32),
        sac_steps=np.asarray([int(r.steps) for r in res_sac], dtype=np.int32),
        rlmppi_steps=np.asarray([int(r.steps) for r in res_rl], dtype=np.int32),
        mppi_final_dist=np.asarray([float(r.final_dist) for r in res_mppi], dtype=np.float32),
        sac_final_dist=np.asarray([float(r.final_dist) for r in res_sac], dtype=np.float32),
        rlmppi_final_dist=np.asarray([float(r.final_dist) for r in res_rl], dtype=np.float32),
        summary_json=json.dumps(summary, ensure_ascii=False),
    )

    print("\n=== Summary (reach && no-collision) ===")
    print(f"MPPI    success_rate={summary['mppi']['success_rate']*100:.1f}% collision_rate={summary['mppi']['collision_rate']*100:.1f}% reached_rate={summary['mppi']['reached_rate']*100:.1f}%")
    print(f"SAC     success_rate={summary['sac']['success_rate']*100:.1f}% collision_rate={summary['sac']['collision_rate']*100:.1f}% reached_rate={summary['sac']['reached_rate']*100:.1f}%")
    print(f"RL-MPPI success_rate={summary['rl_mppi']['success_rate']*100:.1f}% collision_rate={summary['rl_mppi']['collision_rate']*100:.1f}% reached_rate={summary['rl_mppi']['reached_rate']*100:.1f}%")
    print("Saved:", out_path)

    json_path = str(args.save_json).strip()
    if json_path:
        json_path = _resolve_out_path(json_path, default_dir=os.path.dirname(out_path))
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print("Saved:", json_path)


if __name__ == "__main__":
    main()
