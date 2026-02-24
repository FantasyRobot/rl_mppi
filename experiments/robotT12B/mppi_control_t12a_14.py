#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
import time
from contextlib import contextmanager

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from algorithms.mppi.mppi_mujoco_arm import MuJoCoArmMPPI


@contextmanager
def _pushd(path: str):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _get_site_id(mujoco, model, site_name: str) -> int:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if sid < 0:
        raise SystemExit(f"Site not found in model: {site_name}")
    return int(sid)


def main() -> None:
    this_dir = os.path.dirname(os.path.abspath(__file__))
    default_xml = os.path.join(this_dir, "urdf", "t12a_14_normal.xml")

    p = argparse.ArgumentParser(description="MPPI control for t12a_14 (MuJoCo) from start to goal site")
    p.add_argument("--xml", type=str, default=default_xml, help="Path to MJCF xml")
    p.add_argument("--goal_site", type=str, default="goal", help="Name of goal site in XML")
    p.add_argument("--eef_site", type=str, default="end_effector", help="Name of end-effector site in XML")
    p.add_argument("--steps", type=int, default=2500, help="Max simulation steps")
    p.add_argument("--tol", type=float, default=0.03, help="Stop when ||eef-goal|| < tol")
    p.add_argument("--print_every", type=int, default=50, help="Print progress every N steps (headless only)")
    p.add_argument(
        "--action_repeat",
        type=int,
        default=5,
        help="Replan every N physics steps, hold the same ctrl in-between (speed vs tracking).",
    )
    p.add_argument("--no_viewer", action="store_true", help="Run headless (no GUI viewer)")
    p.add_argument("--exit_on_done", action="store_true", help="Exit immediately when done (default: keep window open)")

    # Trajectory drawing (viewer only)
    p.add_argument("--no_draw_traj", action="store_true", help="Disable drawing end-effector trajectory in the viewer")
    p.add_argument("--traj_max_points", type=int, default=400, help="Max end-effector points to keep for trajectory")
    p.add_argument("--traj_stride", type=int, default=1, help="Add one point every N outer-loop iterations")
    p.add_argument("--traj_width", type=float, default=4.0, help="Trajectory line width in pixels")

    # MPPI knobs (kept close to algorithms/mppi style defaults)
    p.add_argument("--horizon", type=int, default=25)
    p.add_argument("--num_samples", type=int, default=96)
    p.add_argument("--lambda_coeff", type=float, default=1.0)
    p.add_argument("--noise_std", type=float, default=0.06)
    p.add_argument("--pos_cost", type=float, default=200.0)
    p.add_argument("--action_cost", type=float, default=0.02)
    p.add_argument("--smooth_cost", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)


    p.add_argument(
        "--init_qpos",
        type=float,
        nargs='+',
        default=None,
        help="初始关节位置列表，如 --init_qpos 0.0 0.0 0.0 0.0 0.0 0.0"
    )
    args = p.parse_args()

    xml_path = os.path.expanduser(os.path.expandvars(str(args.xml)))
    if not os.path.isabs(xml_path):
        xml_path = os.path.abspath(xml_path)
    if not os.path.exists(xml_path):
        raise SystemExit(f"XML not found: {xml_path}")

    try:
        import mujoco
        import mujoco.viewer
    except ModuleNotFoundError:
        raise SystemExit(
            "Python package 'mujoco' is not installed. Install with:\n"
            "  pip install mujoco\n"
            "Then re-run this script."
        )

    xml_dir = os.path.dirname(xml_path)

    with _pushd(xml_dir):
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        # 指定初始关节位置（可通过命令行参数传入）
        if args.init_qpos is not None:
            if len(args.init_qpos) != data.qpos.shape[0]:
                raise ValueError(f"init_qpos长度({len(args.init_qpos)})与机械臂自由度({data.qpos.shape[0]})不符")
            data.qpos[:] = np.array(args.init_qpos)
        data.qvel[:] = np.zeros_like(data.qvel)  # 可选，初始化速度为零
        mujoco.mj_forward(model, data)  # 更新仿真状态

        # Resolve sites
        goal_sid = _get_site_id(mujoco, model, str(args.goal_site))
        eef_sid = _get_site_id(mujoco, model, str(args.eef_site))

        mujoco.mj_forward(model, data)
        goal_pos = np.asarray(data.site_xpos[goal_sid], dtype=np.float64).copy()

        ctrl = MuJoCoArmMPPI(
            model,
            eef_site=str(args.eef_site),
            horizon=int(args.horizon),
            num_samples=int(args.num_samples),
            lambda_coeff=float(args.lambda_coeff),
            noise_std=float(args.noise_std),
            pos_cost_coeff=float(args.pos_cost),
            action_cost_coeff=float(args.action_cost),
            smooth_cost_coeff=float(args.smooth_cost),
            seed=int(args.seed) if args.seed is not None else None,
        )

        def step_once() -> tuple[float, np.ndarray]:
            # Replan every action_repeat physics steps (major speed knob).
            rep = max(1, int(args.action_repeat))
            u = None
            for _ in range(rep):
                if u is None:
                    u = ctrl.get_action(data.qpos, data.qvel, goal_pos)
                data.ctrl[:] = u
                mujoco.mj_step(model, data)

            eef = np.asarray(data.site_xpos[eef_sid], dtype=np.float64).copy()
            dist = float(np.linalg.norm(eef - goal_pos))
            return dist, eef

        def update_traj_overlay(viewer, points: list[np.ndarray]) -> None:
            if viewer.user_scn is None:
                return
            if len(points) < 2:
                return

            scn = viewer.user_scn
            max_geoms = int(getattr(scn, "maxgeom", len(scn.geoms)))
            nseg = min(len(points) - 1, max_geoms)

            rgba = np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32)
            width = float(args.traj_width)

            # MuJoCo Python bindings require explicit arrays for mjv_initGeom.
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

        if args.no_viewer:
            last_dist = float("inf")
            for k in range(int(args.steps)):
                dist, _ = step_once()
                last_dist = dist
                pe = int(args.print_every)
                if pe > 0 and (k % pe == 0):
                    print(f"step={k:5d} dist={dist:.4f}")
                if dist < float(args.tol):
                    print(f"Reached goal: step={k} dist={dist:.4f}")
                    break
            else:
                print(f"Stopped (max steps): step={int(args.steps)} last_dist={last_dist:.4f}")
            return

        import glfw

        viewer_ref: dict[str, object | None] = {"viewer": None}

        def _apply_xml_view_defaults(v) -> None:
            # Force-apply MJCF camera defaults. Some mujoco.viewer builds don't
            # always initialize the passive viewer camera from MJCF.
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

        def _print_view() -> None:
            v = viewer_ref.get("viewer")
            if v is None:
                return
            cam = v.cam
            lookat = np.asarray(cam.lookat, dtype=np.float64)
            print("\n[VIEW] MuJoCo camera")
            print(f"  azimuth   : {float(cam.azimuth):.3f}")
            print(f"  elevation : {float(cam.elevation):.3f}")
            print(f"  distance  : {float(cam.distance):.4f}")
            print(f"  lookat    : {lookat[0]:.4f} {lookat[1]:.4f} {lookat[2]:.4f}")
            print(f"  type      : {int(cam.type)}")
            print(f"  fixedcamid: {int(cam.fixedcamid)}")
            print(f"  trackbodyid: {int(cam.trackbodyid)}")
            print("  XML <visual><global .../> suggestion:")
            print(f"    <global azimuth=\"{float(cam.azimuth):.3f}\" elevation=\"{float(cam.elevation):.3f}\"/>")
            print("  XML <statistic .../> suggestion (persists lookat/distance for free camera):")
            print(f"    <statistic center=\"{lookat[0]:.4f} {lookat[1]:.4f} {lookat[2]:.4f}\" extent=\"{float(cam.distance):.4f}\"/>")

        def _key_callback(key: int) -> None:
            # Press 'P' to print current view.
            if key == int(glfw.KEY_P):
                _print_view()

        with mujoco.viewer.launch_passive(model, data, key_callback=_key_callback) as viewer:
            viewer_ref["viewer"] = viewer
            _apply_xml_view_defaults(viewer)
            traj_points: list[np.ndarray] = []
            finished_reason: str | None = None
            for k in range(int(args.steps)):
                if not viewer.is_running():
                    break
                dist, eef = step_once()

                if not args.no_draw_traj:
                    stride = max(1, int(args.traj_stride))
                    if k % stride == 0:
                        traj_points.append(eef)
                        maxp = max(2, int(args.traj_max_points))
                        if len(traj_points) > maxp:
                            traj_points = traj_points[-maxp:]
                        update_traj_overlay(viewer, traj_points)

                try:
                    viewer.set_texts((None, None, f"dist={dist:.4f}", None))
                except Exception:
                    pass

                if dist < float(args.tol):
                    finished_reason = f"Reached goal (dist={dist:.4f})"
                    break
                viewer.sync()

            if viewer.is_running() and (not args.exit_on_done):
                # Keep the window open after finishing, and continue to render.
                if finished_reason is None:
                    finished_reason = "Finished steps"
                try:
                    viewer.set_texts((None, None, finished_reason, "(close window to exit)"))
                except Exception:
                    pass
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(0.01)


if __name__ == "__main__":
    main()
