#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
import time
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


def load_sac_policy(model_path: str):
    try:
        import torch
    except ModuleNotFoundError as e:
        raise SystemExit(
            "Torch is required for SAC policy loading. Install it in your env, e.g. (CPU):\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
        ) from e

    from algorithms.sac.sac_utils import SACAgent

    ckpt = torch.load(str(model_path), map_location="cpu")
    obs_dim = int(ckpt["obs_dim"])
    action_dim = int(ckpt["action_dim"])

    agent = SACAgent(
        state_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=256,
        learning_rate=3e-4,
        alpha=float(ckpt.get("alpha", 0.2)),
        gamma=0.99,
        tau=0.005,
        auto_entropy_tuning=bool(ckpt.get("auto_entropy_tuning", False)),
        use_lr_scheduler=False,
    )

    agent.policy_net.load_state_dict(ckpt["policy_state_dict"])
    agent.q_net1.load_state_dict(ckpt["q1_state_dict"])
    agent.q_net2.load_state_dict(ckpt["q2_state_dict"])
    agent.target_q_net1.load_state_dict(ckpt["target_q1_state_dict"])
    agent.target_q_net2.load_state_dict(ckpt["target_q2_state_dict"])

    return agent, ckpt


def test_sac_t12a_14(
    *,
    model_path: str,
    xml_path: str,
    goal_site: str = "goal",
    eef_site: str = "end_effector",
    max_steps: int = 2500,
    action_repeat: int = 5,
    init_qpos=None,
) -> dict:
    from env.envmujoco_t12a_14 import T12A14MuJoCoEnv

    agent, ckpt = load_sac_policy(str(model_path))

    env = T12A14MuJoCoEnv(
        xml_path=str(xml_path),
        goal_site=str(goal_site),
        eef_site=str(eef_site),
        max_steps=int(max_steps),
        action_repeat=int(action_repeat),
    )

    obs = env.reset(init_qpos=init_qpos)
    eef_traj: list[np.ndarray] = []
    dist_traj: list[float] = []

    for _ in range(int(max_steps)):
        a = agent.select_action(obs, evaluate=True)
        obs, _r, done, info = env.step(a)
        eef_traj.append(np.asarray(info["eef"], dtype=np.float32))
        dist_traj.append(float(info["dist"]))
        if done:
            break

    return {
        "eef": np.asarray(eef_traj, dtype=np.float32),
        "dist": np.asarray(dist_traj, dtype=np.float32),
        "success": bool(info.get("success", False)),
        "final_dist": float(dist_traj[-1]) if dist_traj else float("inf"),
        "ckpt": ckpt,
    }


def test_sac_t12a_14_viewer(
    *,
    model_path: str,
    xml_path: str,
    goal_site: str = "goal",
    eef_site: str = "end_effector",
    max_steps: int = 2500,
    action_repeat: int = 5,
    exit_on_done: bool = False,
    no_draw_traj: bool = False,
    traj_max_points: int = 400,
    traj_stride: int = 1,
    traj_width: float = 4.0,
    init_qpos=None,
) -> dict:
    """Run SAC policy in a MuJoCo viewer and draw the end-effector trajectory."""

    from env.envmujoco_t12a_14 import T12A14MuJoCoEnv

    agent, ckpt = load_sac_policy(str(model_path))

    env = T12A14MuJoCoEnv(
        xml_path=str(xml_path),
        goal_site=str(goal_site),
        eef_site=str(eef_site),
        max_steps=int(max_steps),
        action_repeat=int(action_repeat),
    )

    try:
        import mujoco
        import mujoco.viewer
    except ModuleNotFoundError:
        raise SystemExit("Python package 'mujoco' is not installed. Install with: pip install mujoco")

    def _apply_xml_view_defaults(v) -> None:
        # Force-apply MJCF camera defaults (same as MPPI script).
        model = env.model
        vis = getattr(model, "vis", None)
        glob = None
        if vis is not None:
            glob = getattr(vis, "global_", None)
            if glob is None:
                glob = getattr(vis, "global", None)

        stat = getattr(model, "stat", None)
        if glob is None or stat is None:
            return

        az = float(getattr(glob, "azimuth", 0.0))
        el = float(getattr(glob, "elevation", 0.0))
        center = np.asarray(getattr(stat, "center"), dtype=np.float64).reshape(3)
        extent = float(getattr(stat, "extent", 1.0))

        with v.lock():
            cam = v.cam
            # Force free camera to avoid any tracking/fixed-camera overrides.
            try:
                cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            except Exception:
                cam.type = 0
            cam.fixedcamid = -1
            cam.trackbodyid = -1
            cam.azimuth = az
            cam.elevation = el
            cam.distance = extent
            cam.lookat[0] = float(center[0])
            cam.lookat[1] = float(center[1])
            cam.lookat[2] = float(center[2])

    def update_traj_overlay(viewer, points: list[np.ndarray]) -> None:
        if viewer.user_scn is None:
            return
        if len(points) < 2:
            return

        scn = viewer.user_scn
        max_geoms = int(getattr(scn, "maxgeom", len(scn.geoms)))
        nseg = min(len(points) - 1, max_geoms)

        rgba = np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32)  # yellow
        width = float(traj_width)

        size = np.zeros((3,), dtype=np.float64)
        pos = np.zeros((3,), dtype=np.float64)
        mat = np.eye(3, dtype=np.float64).reshape(-1)

        with viewer.lock():
            scn.ngeom = 0
            for i in range(nseg):
                geom = scn.geoms[i]
                mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE, size, pos, mat, rgba)
                mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, width, points[i], points[i + 1])
            scn.ngeom = nseg

    obs = env.reset(init_qpos=init_qpos)
    traj_points: list[np.ndarray] = []
    eef_traj: list[np.ndarray] = []
    dist_traj: list[float] = []

    # Ensure relative mesh paths resolve.
    xml_path_abs = os.path.abspath(os.path.expandvars(os.path.expanduser(str(xml_path))))
    xml_dir = os.path.dirname(xml_path_abs)

    with _pushd(xml_dir):
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            _apply_xml_view_defaults(viewer)
            finished_reason: str | None = None
            for k in range(int(max_steps)):
                if not viewer.is_running():
                    break

                a = agent.select_action(obs, evaluate=True)
                obs, _r, done, info = env.step(a)

                eef = np.asarray(info["eef"], dtype=np.float64)
                dist = float(info["dist"])
                eef_traj.append(eef.astype(np.float32))
                dist_traj.append(dist)

                if not bool(no_draw_traj):
                    stride = max(1, int(traj_stride))
                    if k % stride == 0:
                        traj_points.append(eef)
                        maxp = max(2, int(traj_max_points))
                        if len(traj_points) > maxp:
                            traj_points = traj_points[-maxp:]
                        update_traj_overlay(viewer, traj_points)

                try:
                    viewer.set_texts((None, None, f"dist={dist:.4f}", None))
                except Exception:
                    pass

                viewer.sync()
                time.sleep(1)
                if done:
                    finished_reason = f"Done (dist={dist:.4f}, success={bool(info.get('success', False))})"
                    break

            if viewer.is_running() and (not bool(exit_on_done)):
                if finished_reason is None:
                    finished_reason = "Finished"
                try:
                    viewer.set_texts((None, None, finished_reason, "(close window to exit)"))
                except Exception:
                    pass
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(0.01)

    success = bool(info.get("success", False)) if dist_traj else False
    final_dist = float(dist_traj[-1]) if dist_traj else float("inf")
    return {
        "eef": np.asarray(eef_traj, dtype=np.float32),
        "dist": np.asarray(dist_traj, dtype=np.float32),
        "success": bool(success),
        "final_dist": float(final_dist),
        "ckpt": ckpt,
    }
