#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from contextlib import contextmanager


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
    default_xml = os.path.join(this_dir, "urdf", "t12a_14_simple.xml")

    p = argparse.ArgumentParser(description="MuJoCo visualize t12a_14 from existing MJCF (t12a_14.xml)")
    p.add_argument("--xml", type=str, default=default_xml, help="Path to MJCF xml (default: t12a_14.xml)")
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

        import numpy as np
        import glfw

        viewer_ref: dict[str, object | None] = {"viewer": None}

        def _apply_xml_view_defaults(v) -> None:
            # In some mujoco.viewer builds, the passive viewer may not apply
            # MJCF <visual><global .../> and <statistic .../> automatically.
            # We force-apply them so restarting reproduces the saved view.
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
            _apply_xml_view_defaults(viewer)
            while viewer.is_running():
                mujoco.mj_step(model, data)
                viewer.sync()


if __name__ == "__main__":
    main()
