#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from contextlib import contextmanager

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from experiments.multi_robots.hrsga.mujoco_interface import (
    _build_robot_handles,
    _build_robot_initial_joint_targets,
    apply_xml_view_defaults,
)


@contextmanager
def _pushd(path: str):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def main() -> None:
    default_xml = os.path.join(THIS_DIR, "multi_robots.xml")

    p = argparse.ArgumentParser(description="MuJoCo visualize multi_robots from existing MJCF (multi_robots.xml)")

    p.add_argument("--xml", type=str, default=default_xml, help="Path to MJCF xml (default: multi_robots.xml)")
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
            "Python package 'mujoco' is not installed. Install one of:\n"
            "  pip install mujoco\n"
            "(Optionally in your venv)"
        )

    xml_dir = os.path.dirname(xml_path)

    # The XML references meshes like file="link1.STL" without meshdir, so we load
    # with CWD set to the XML directory to make relative asset paths resolve.
    with _pushd(xml_dir):      
        model = mujoco.MjModel.from_xml_path(xml_path)
        # 设置积分器参数
        #model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST  # 隐式快速积分
        data = mujoco.MjData(model)


        import glfw

        robot_handles = _build_robot_handles(mujoco, model)

        # 默认使用 XML <custom> 中的 robotN_init_joints 初始化；如果显式传入 --init_qpos，则整条 qpos 优先。
        if args.init_qpos is not None:
            if len(args.init_qpos) != data.qpos.shape[0]:
                raise ValueError(f"init_qpos长度({len(args.init_qpos)})与机械臂自由度({data.qpos.shape[0]})不符")
            data.qpos[:] = np.asarray(args.init_qpos, dtype=np.float64)
            if getattr(data, "ctrl", None) is not None and data.ctrl.shape[0] == data.qpos.shape[0]:
                data.ctrl[:] = np.asarray(args.init_qpos, dtype=np.float64)
        else:
            initial_joint_targets = _build_robot_initial_joint_targets(mujoco, model, robot_handles)
            for handle, q_init in zip(robot_handles, initial_joint_targets):
                qpos_indices = handle["qpos_indices"]
                actuator_indices = handle["actuator_indices"]
                data.qpos[qpos_indices] = np.asarray(q_init, dtype=np.float64)
                data.ctrl[actuator_indices] = np.asarray(q_init, dtype=np.float64)
        mujoco.mj_forward(model, data)

        viewer_ref: dict[str, object | None] = {"viewer": None}

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
            print("  XML <visual><global .../> suggestion:")
            print(f"    <global azimuth=\"{float(cam.azimuth):.3f}\" elevation=\"{float(cam.elevation):.3f}\"/>")
            print("  XML <statistic .../> suggestion (persists lookat/distance for free camera):")
            print(f"    <statistic center=\"{lookat[0]:.4f} {lookat[1]:.4f} {lookat[2]:.4f}\" extent=\"{float(cam.distance):.4f}\"/>")

        def _key_callback(key: int) -> None:
            if key == int(glfw.KEY_P):
                _print_view()

        # Passive viewer: step simulation and render.
        with mujoco.viewer.launch_passive(model, data, key_callback=_key_callback) as viewer:
            viewer_ref["viewer"] = viewer
            apply_xml_view_defaults(mujoco, model, viewer)
            while viewer.is_running():
                mujoco.mj_step(model, data)
                viewer.sync()


if __name__ == "__main__":
    main()
