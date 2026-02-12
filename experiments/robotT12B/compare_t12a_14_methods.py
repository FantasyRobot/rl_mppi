#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from contextlib import contextmanager

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


def main() -> None:
    this_dir = os.path.dirname(os.path.abspath(__file__))
    default_xml = os.path.join(this_dir, "urdf", "t12a_14.xml")

    p = argparse.ArgumentParser(description="Compare MPPI vs SAC vs RL-MPPI on t12a_14")
    p.add_argument("--xml", type=str, default=default_xml)
    p.add_argument("--goal_site", type=str, default="goal")
    p.add_argument("--eef_site", type=str, default="end_effector")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--action_repeat", type=int, default=5)
    p.add_argument("--tol", type=float, default=0.05)

    p.add_argument("--sac_model", type=str, default=os.path.join(this_dir, "models", "sac_t12a_14_model.pth"))

    p.add_argument("--save_npz", type=str, default=os.path.join(_ROOT_DIR, "experiments", "results", "t12a_14_compare.npz"))

    args = p.parse_args()

    try:
        import mujoco
    except ModuleNotFoundError:
        raise SystemExit("Install mujoco first: pip install mujoco")

    xml_path = os.path.abspath(os.path.expandvars(os.path.expanduser(str(args.xml))))
    if not os.path.exists(xml_path):
        raise SystemExit(f"XML not found: {xml_path}")

    xml_dir = os.path.dirname(xml_path)
    with _pushd(xml_dir):
        model = mujoco.MjModel.from_xml_path(xml_path)

    from algorithms.mppi.mppi_mujoco_arm import MuJoCoArmMPPI
    from algorithms.rl_mppi.rl_mppi_mujoco_arm import RLMuJoCoArmMPPI, load_sac_policy
    from env.envmujoco_t12a_14 import T12A14MuJoCoEnv

    def _effective_dt(env: T12A14MuJoCoEnv) -> float:
        # One env.step() advances action_repeat physics steps.
        return float(env.model.opt.timestep) * float(env.action_repeat)

    def _slice_q(env: T12A14MuJoCoEnv) -> slice:
        # Log only actuated joints (same convention as env obs).
        return slice(0, int(env.nu))

    # --- MPPI ---
    env_mppi = T12A14MuJoCoEnv(
        xml_path=xml_path,
        goal_site=str(args.goal_site),
        eef_site=str(args.eef_site),
        max_steps=int(args.steps),
        reach_tol=float(args.tol),
        action_repeat=int(args.action_repeat),
    )
    obs = env_mppi.reset()
    mppi = MuJoCoArmMPPI(model, eef_site=str(args.eef_site))

    mppi_eef = []
    mppi_dist = []
    mppi_reward = []
    mppi_qpos = []
    mppi_qvel = []
    mppi_qacc = []
    dt_mppi = _effective_dt(env_mppi)
    qsl_mppi = _slice_q(env_mppi)
    prev_qvel_mppi = np.asarray(env_mppi.data.qvel[qsl_mppi], dtype=np.float64).copy()
    for _ in range(int(args.steps)):
        # use internal data from env for state
        qpos = env_mppi.data.qpos
        qvel = env_mppi.data.qvel
        u = mppi.get_action(qpos, qvel, env_mppi.goal_pos)
        # map ctrl -> action_norm then step env
        a_norm = env_mppi.ctrl_to_action(u)
        obs, r, done, info = env_mppi.step(a_norm)

        qpos_now = np.asarray(env_mppi.data.qpos[qsl_mppi], dtype=np.float64).copy()
        qvel_now = np.asarray(env_mppi.data.qvel[qsl_mppi], dtype=np.float64).copy()
        qacc_now = (qvel_now - prev_qvel_mppi) / (dt_mppi if dt_mppi > 0 else 1.0)
        prev_qvel_mppi = qvel_now

        mppi_eef.append(info["eef"])
        mppi_dist.append(info["dist"])
        mppi_reward.append(float(r))
        mppi_qpos.append(qpos_now)
        mppi_qvel.append(qvel_now)
        mppi_qacc.append(qacc_now)
        if done:
            break

    # --- SAC ---
    env_sac = T12A14MuJoCoEnv(
        xml_path=xml_path,
        goal_site=str(args.goal_site),
        eef_site=str(args.eef_site),
        max_steps=int(args.steps),
        reach_tol=float(args.tol),
        action_repeat=int(args.action_repeat),
    )
    obs = env_sac.reset()

    try:
        from test_sac_t12a_14 import load_sac_policy as _load_sac_agent

        sac_agent, _ckpt = _load_sac_agent(str(args.sac_model))
    except Exception as e:
        raise SystemExit(
            f"Failed to load SAC model: {args.sac_model}\n"
            f"Error: {e}\n"
            "Train first via: python experiments/robotT12B/sac_t12a_14_cli.py train"
        )

    sac_eef = []
    sac_dist = []
    sac_reward = []
    sac_qpos = []
    sac_qvel = []
    sac_qacc = []
    dt_sac = _effective_dt(env_sac)
    qsl_sac = _slice_q(env_sac)
    prev_qvel_sac = np.asarray(env_sac.data.qvel[qsl_sac], dtype=np.float64).copy()
    for _ in range(int(args.steps)):
        a = sac_agent.select_action(obs, evaluate=True)
        obs, r, done, info = env_sac.step(a)

        qpos_now = np.asarray(env_sac.data.qpos[qsl_sac], dtype=np.float64).copy()
        qvel_now = np.asarray(env_sac.data.qvel[qsl_sac], dtype=np.float64).copy()
        qacc_now = (qvel_now - prev_qvel_sac) / (dt_sac if dt_sac > 0 else 1.0)
        prev_qvel_sac = qvel_now

        sac_eef.append(info["eef"])
        sac_dist.append(info["dist"])
        sac_reward.append(float(r))
        sac_qpos.append(qpos_now)
        sac_qvel.append(qvel_now)
        sac_qacc.append(qacc_now)
        if done:
            break

    # --- RL-MPPI ---
    env_rl = T12A14MuJoCoEnv(
        xml_path=xml_path,
        goal_site=str(args.goal_site),
        eef_site=str(args.eef_site),
        max_steps=int(args.steps),
        reach_tol=float(args.tol),
        action_repeat=int(args.action_repeat),
    )
    obs = env_rl.reset()

    policy = load_sac_policy(str(args.sac_model))
    rl_mppi = RLMuJoCoArmMPPI(
        model,
        policy,
        eef_site=str(args.eef_site),
        goal_site=str(args.goal_site),
    )

    rlmppi_eef = []
    rlmppi_dist = []
    rlmppi_reward = []
    rlmppi_qpos = []
    rlmppi_qvel = []
    rlmppi_qacc = []
    dt_rlmppi = _effective_dt(env_rl)
    qsl_rlmppi = _slice_q(env_rl)
    prev_qvel_rlmppi = np.asarray(env_rl.data.qvel[qsl_rlmppi], dtype=np.float64).copy()
    for _ in range(int(args.steps)):
        u = rl_mppi.get_action(env_rl.data.qpos, env_rl.data.qvel)
        a_norm = env_rl.ctrl_to_action(u)
        obs, r, done, info = env_rl.step(a_norm)

        qpos_now = np.asarray(env_rl.data.qpos[qsl_rlmppi], dtype=np.float64).copy()
        qvel_now = np.asarray(env_rl.data.qvel[qsl_rlmppi], dtype=np.float64).copy()
        qacc_now = (qvel_now - prev_qvel_rlmppi) / (dt_rlmppi if dt_rlmppi > 0 else 1.0)
        prev_qvel_rlmppi = qvel_now

        rlmppi_eef.append(info["eef"])
        rlmppi_dist.append(info["dist"])
        rlmppi_reward.append(float(r))
        rlmppi_qpos.append(qpos_now)
        rlmppi_qvel.append(qvel_now)
        rlmppi_qacc.append(qacc_now)
        if done:
            break

    out_dir = os.path.dirname(os.path.abspath(str(args.save_npz)))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    np.savez(
        str(args.save_npz),
        dt=float(dt_mppi),
        action_repeat=int(args.action_repeat),
        tol=float(args.tol),
        mppi_eef=np.asarray(mppi_eef, dtype=np.float32),
        mppi_dist=np.asarray(mppi_dist, dtype=np.float32),
        mppi_reward=np.asarray(mppi_reward, dtype=np.float32),
        mppi_qpos=np.asarray(mppi_qpos, dtype=np.float32),
        mppi_qvel=np.asarray(mppi_qvel, dtype=np.float32),
        mppi_qacc=np.asarray(mppi_qacc, dtype=np.float32),
        sac_eef=np.asarray(sac_eef, dtype=np.float32),
        sac_dist=np.asarray(sac_dist, dtype=np.float32),
        sac_reward=np.asarray(sac_reward, dtype=np.float32),
        sac_qpos=np.asarray(sac_qpos, dtype=np.float32),
        sac_qvel=np.asarray(sac_qvel, dtype=np.float32),
        sac_qacc=np.asarray(sac_qacc, dtype=np.float32),
        rlmppi_eef=np.asarray(rlmppi_eef, dtype=np.float32),
        rlmppi_dist=np.asarray(rlmppi_dist, dtype=np.float32),
        rlmppi_reward=np.asarray(rlmppi_reward, dtype=np.float32),
        rlmppi_qpos=np.asarray(rlmppi_qpos, dtype=np.float32),
        rlmppi_qvel=np.asarray(rlmppi_qvel, dtype=np.float32),
        rlmppi_qacc=np.asarray(rlmppi_qacc, dtype=np.float32),
    )

    print("[DONE] Saved:", str(args.save_npz))
    print(f"MPPI   final_dist={float(mppi_dist[-1]) if mppi_dist else float('inf'):.4f} steps={len(mppi_dist)}")
    print(f"SAC    final_dist={float(sac_dist[-1]) if sac_dist else float('inf'):.4f} steps={len(sac_dist)}")
    print(f"RLMPPI final_dist={float(rlmppi_dist[-1]) if rlmppi_dist else float('inf'):.4f} steps={len(rlmppi_dist)}")


if __name__ == "__main__":
    main()
