from __future__ import annotations

import os
import re
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np


@contextmanager
def _pushd(path: str):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _require_mujoco():
    try:
        import mujoco
        import mujoco.viewer  # noqa: F401
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError("MuJoCo support requires 'mujoco' (pip install mujoco).") from error
    return mujoco


def _numeric_suffix(name: str, prefix: str) -> int:
    suffix = name[len(prefix):]
    return int(suffix) if suffix.isdigit() else 10**9


def _robot_index_from_name(name: str, prefix: str):
    match = re.match(rf"^{re.escape(prefix)}(\d+)", str(name))
    if not match:
        return None
    return int(match.group(1))


def _get_named_numeric_values(mujoco, model, numeric_name: str):
    for numeric_id in range(int(getattr(model, "nnumeric", 0))):
        current_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_NUMERIC, numeric_id) or ""
        if current_name != str(numeric_name):
            continue
        numeric_adr = int(model.numeric_adr[numeric_id])
        numeric_size = int(model.numeric_size[numeric_id])
        return np.asarray(model.numeric_data[numeric_adr:numeric_adr + numeric_size], dtype=np.float64).copy()
    return None


def load_model_and_data(xml_path: str):
    mujoco = _require_mujoco()
    resolved_xml_path = os.path.abspath(os.path.expanduser(os.path.expandvars(str(xml_path))))
    if not os.path.exists(resolved_xml_path):
        raise FileNotFoundError(f"XML not found: {resolved_xml_path}")

    xml_dir = os.path.dirname(resolved_xml_path)
    with _pushd(xml_dir):
        model = mujoco.MjModel.from_xml_path(resolved_xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return mujoco, model, data, resolved_xml_path


def load_layout_templates_from_mujoco(xml_path: str):
    mujoco, model, data, resolved_xml_path = load_model_and_data(xml_path)

    robot_positions = []
    robot_initial_joints = []
    obstacles = []
    goal_positions = []
    goal_positions_3d = []
    goal_radius = 0.0
    goal_height = 0.0
    max_joint_speed = None
    enforce_visit_order = None

    for body_id in range(int(model.nbody)):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        robot_index = _robot_index_from_name(body_name, "robot")
        if robot_index is not None and body_name == f"robot{robot_index}":
            body_pos = np.asarray(data.xpos[body_id], dtype=np.float64)
            robot_positions.append((int(robot_index), (float(body_pos[0]), float(body_pos[1]))))
            continue
        if body_name.startswith("obstacle"):
            body_pos = np.asarray(data.xpos[body_id], dtype=np.float64)
            geom_adr = int(model.body_geomadr[body_id])
            geom_num = int(model.body_geomnum[body_id])
            radius = float(model.geom_size[geom_adr][0]) if geom_num > 0 else 0.0
            obstacles.append(
                (
                    _numeric_suffix(body_name, "obstacle"),
                    {"center": (float(body_pos[0]), float(body_pos[1])), "radius": radius},
                )
            )

    for site_id in range(int(model.nsite)):
        site_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site_id) or ""
        if not site_name.startswith("goal"):
            continue
        site_pos = np.asarray(data.site_xpos[site_id], dtype=np.float64)
        goal_positions.append((_numeric_suffix(site_name, "goal"), (float(site_pos[0]), float(site_pos[1]))))
        goal_positions_3d.append((_numeric_suffix(site_name, "goal"), (float(site_pos[0]), float(site_pos[1]), float(site_pos[2]))))
        goal_height = max(goal_height, float(site_pos[2]))
        goal_radius = max(goal_radius, float(model.site_size[site_id][0]))

    for numeric_id in range(int(getattr(model, "nnumeric", 0))):
        numeric_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_NUMERIC, numeric_id) or ""
        robot_index = _robot_index_from_name(numeric_name, "robot")
        if robot_index is None or numeric_name != f"robot{robot_index}_init_joints":
            continue
        numeric_adr = int(model.numeric_adr[numeric_id])
        numeric_size = int(model.numeric_size[numeric_id])
        values = np.asarray(model.numeric_data[numeric_adr:numeric_adr + numeric_size], dtype=np.float64).copy()
        robot_initial_joints.append((int(robot_index), tuple(float(value) for value in values.tolist())))

    max_joint_speed_values = _get_named_numeric_values(mujoco, model, "max_joint_speed")
    if max_joint_speed_values is not None and max_joint_speed_values.size > 0:
        max_joint_speed = float(max_joint_speed_values.reshape(-1)[0])

    enforce_visit_order_values = _get_named_numeric_values(mujoco, model, "enforce_visit_order")
    if enforce_visit_order_values is not None and enforce_visit_order_values.size > 0:
        enforce_visit_order = bool(int(round(float(enforce_visit_order_values.reshape(-1)[0]))))

    return {
        "xml_path": resolved_xml_path,
        "robot_positions": [pos for _, pos in sorted(robot_positions, key=lambda item: item[0])],
        "robot_initial_joints": [values for _, values in sorted(robot_initial_joints, key=lambda item: item[0])],
        "goal_positions": [pos for _, pos in sorted(goal_positions, key=lambda item: item[0])],
        "goal_positions_3d": [pos for _, pos in sorted(goal_positions_3d, key=lambda item: item[0])],
        "obstacles": [item for _, item in sorted(obstacles, key=lambda pair: pair[0])],
        "goal_radius": goal_radius,
        "goal_height": goal_height,
        "max_joint_speed": max_joint_speed,
        "enforce_visit_order": enforce_visit_order,
    }


def apply_xml_view_defaults(mujoco, model, viewer) -> None:
    vis = getattr(model, "vis", None)
    glob = None
    if vis is not None:
        glob = getattr(vis, "global_", None)
        if glob is None:
            glob = getattr(vis, "global", None)

    stat = getattr(model, "stat", None)
    if glob is None or stat is None:
        return

    azimuth = float(getattr(glob, "azimuth", 0.0))
    elevation = float(getattr(glob, "elevation", 0.0))
    center = np.asarray(getattr(stat, "center"), dtype=np.float64).reshape(3)
    extent = float(getattr(stat, "extent", 1.0))

    with viewer.lock():
        _configure_free_camera_from_xml_defaults(mujoco, model, viewer.cam)


def _configure_free_camera_from_xml_defaults(mujoco, model, cam) -> None:
    vis = getattr(model, "vis", None)
    glob = None
    if vis is not None:
        glob = getattr(vis, "global_", None)
        if glob is None:
            glob = getattr(vis, "global", None)

    stat = getattr(model, "stat", None)
    if glob is None or stat is None:
        return

    azimuth = float(getattr(glob, "azimuth", 0.0))
    elevation = float(getattr(glob, "elevation", 0.0))
    center = np.asarray(getattr(stat, "center"), dtype=np.float64).reshape(3)
    extent = float(getattr(stat, "extent", 1.0))

    try:
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    except Exception:
        cam.type = 0
    cam.fixedcamid = -1
    cam.trackbodyid = -1
    cam.azimuth = azimuth
    cam.elevation = elevation
    cam.distance = extent
    cam.lookat[0] = float(center[0])
    cam.lookat[1] = float(center[1])
    cam.lookat[2] = float(center[2])


def _resolve_offscreen_render_size(model, requested_width: int, requested_height: int) -> tuple[int, int]:
    width = max(64, int(requested_width))
    height = max(64, int(requested_height))

    vis = getattr(model, "vis", None)
    glob = None
    if vis is not None:
        glob = getattr(vis, "global_", None)
        if glob is None:
            glob = getattr(vis, "global", None)

    max_width = int(getattr(glob, "offwidth", width)) if glob is not None else width
    max_height = int(getattr(glob, "offheight", height)) if glob is not None else height
    max_width = max(64, max_width)
    max_height = max(64, max_height)
    return min(width, max_width), min(height, max_height)


def _build_robot_handles(mujoco, model):
    robot_handles = {}

    for site_id in range(int(model.nsite)):
        site_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site_id) or ""
        robot_index = _robot_index_from_name(site_name, "end_effector")
        if robot_index is None:
            continue
        robot_handles.setdefault(robot_index, {})["site_id"] = int(site_id)
        robot_handles[robot_index]["site_name"] = site_name

    joint_specs = {}
    for joint_id in range(int(model.njnt)):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""
        robot_index = _robot_index_from_name(joint_name, "robot")
        if robot_index is None:
            continue
        joint_specs.setdefault(robot_index, []).append(
            {
                "joint_id": int(joint_id),
                "joint_name": joint_name,
                "joint_order": _numeric_suffix(joint_name, f"robot{robot_index}_joint"),
                "qpos_adr": int(model.jnt_qposadr[joint_id]),
                "dof_adr": int(model.jnt_dofadr[joint_id]),
            }
        )

    actuator_specs = {}
    for actuator_id in range(int(model.nu)):
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id) or ""
        robot_index = _robot_index_from_name(actuator_name, "robot")
        if robot_index is None:
            continue
        actuator_specs.setdefault(robot_index, []).append(
            {
                "actuator_id": int(actuator_id),
                "actuator_name": actuator_name,
                "actuator_order": _numeric_suffix(actuator_name, f"robot{robot_index}_motor_joint"),
                "ctrl_min": float(model.actuator_ctrlrange[actuator_id][0]),
                "ctrl_max": float(model.actuator_ctrlrange[actuator_id][1]),
            }
        )

    handles = []
    for robot_index in sorted(robot_handles.keys()):
        site_id = robot_handles[robot_index].get("site_id")
        joints = sorted(joint_specs.get(robot_index, []), key=lambda item: item["joint_order"])
        actuators = sorted(actuator_specs.get(robot_index, []), key=lambda item: item["actuator_order"])
        if site_id is None or not joints or not actuators or len(joints) != len(actuators):
            continue
        handles.append(
            {
                "robot_index": int(robot_index),
                "site_id": int(site_id),
                "site_name": robot_handles[robot_index]["site_name"],
                "qpos_indices": np.asarray([item["qpos_adr"] for item in joints], dtype=np.int32),
                "dof_indices": np.asarray([item["dof_adr"] for item in joints], dtype=np.int32),
                "actuator_indices": np.asarray([item["actuator_id"] for item in actuators], dtype=np.int32),
                "ctrl_min": np.asarray([item["ctrl_min"] for item in actuators], dtype=np.float64),
                "ctrl_max": np.asarray([item["ctrl_max"] for item in actuators], dtype=np.float64),
            }
        )
    return handles


def _build_goal_handles(mujoco, model, data):
    handles = []
    for site_id in range(int(model.nsite)):
        site_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site_id) or ""
        if not site_name.startswith("goal"):
            continue
        handles.append(
            {
                "goal_index": int(_numeric_suffix(site_name, "goal")),
                "site_id": int(site_id),
                "site_name": site_name,
                "position": np.asarray(data.site_xpos[site_id], dtype=np.float64).copy(),
                "radius": float(model.site_size[site_id][0]),
            }
        )
    return sorted(handles, key=lambda item: item["goal_index"])


def _build_robot_initial_joint_targets(mujoco, model, robot_handles):
    configured_targets = {}
    for numeric_id in range(int(getattr(model, "nnumeric", 0))):
        numeric_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_NUMERIC, numeric_id) or ""
        robot_index = _robot_index_from_name(numeric_name, "robot")
        if robot_index is None or numeric_name != f"robot{robot_index}_init_joints":
            continue
        numeric_adr = int(model.numeric_adr[numeric_id])
        numeric_size = int(model.numeric_size[numeric_id])
        configured_targets[int(robot_index)] = np.asarray(
            model.numeric_data[numeric_adr:numeric_adr + numeric_size],
            dtype=np.float64,
        ).copy()

    joint_targets = []
    for handle in robot_handles:
        qpos_indices = handle["qpos_indices"]
        qpos0 = np.asarray(model.qpos0[qpos_indices], dtype=np.float64).copy()
        configured = configured_targets.get(int(handle["robot_index"]), qpos0)
        joint_targets.append(_normalize_joint_target(qpos0, configured, handle["ctrl_min"], handle["ctrl_max"]))
    return joint_targets


def _copy_data_state(src_data, dst_data):
    for name in ("qpos", "qvel", "act", "ctrl", "qacc_warmstart"):
        if hasattr(src_data, name) and hasattr(dst_data, name):
            dst_value = getattr(dst_data, name)
            src_value = getattr(src_data, name)
            if dst_value.shape == src_value.shape:
                dst_value[:] = src_value


def _normalize_joint_target(current_qpos: np.ndarray, target, ctrl_min: np.ndarray, ctrl_max: np.ndarray):
    target_array = np.asarray(target, dtype=np.float64).reshape(-1)
    if target_array.size == 0:
        q_des = np.asarray(current_qpos, dtype=np.float64)
    else:
        q_des = np.asarray(current_qpos, dtype=np.float64).copy()
        limit = min(q_des.shape[0], target_array.shape[0])
        q_des[:limit] = target_array[:limit]
    return np.clip(q_des, ctrl_min, ctrl_max)


def _resolve_joint_delta_limit(model, frame_substeps: int, max_joint_speed):
    if max_joint_speed is None:
        return None
    max_joint_speed = float(max_joint_speed)
    if not np.isfinite(max_joint_speed) or max_joint_speed <= 0.0:
        return None
    step_dt = float(getattr(getattr(model, "opt", None), "timestep", 0.0)) * max(1, int(frame_substeps))
    if step_dt <= 0.0:
        return None
    return max_joint_speed * step_dt


def limit_joint_target_step(current_qpos: np.ndarray, target, ctrl_min: np.ndarray, ctrl_max: np.ndarray, *, max_joint_delta=None):
    q_des = _normalize_joint_target(current_qpos, target, ctrl_min, ctrl_max)
    if max_joint_delta is None:
        return q_des
    max_joint_delta = float(max_joint_delta)
    if not np.isfinite(max_joint_delta) or max_joint_delta <= 0.0:
        return q_des
    current = np.asarray(current_qpos, dtype=np.float64).reshape(-1)
    limited_delta = np.clip(q_des - current, -max_joint_delta, max_joint_delta)
    return np.clip(current + limited_delta, ctrl_min, ctrl_max)


def apply_joint_targets(mujoco, model, data, robot_handles, joint_targets, *, frame_substeps: int = 1, max_joint_speed=None):
    max_joint_delta = _resolve_joint_delta_limit(model, frame_substeps, max_joint_speed)
    for handle, target in zip(robot_handles, joint_targets):
        qpos_indices = handle["qpos_indices"]
        actuator_indices = handle["actuator_indices"]
        current_qpos = np.asarray(data.qpos[qpos_indices], dtype=np.float64)
        q_des = limit_joint_target_step(
            current_qpos,
            target,
            handle["ctrl_min"],
            handle["ctrl_max"],
            max_joint_delta=max_joint_delta,
        )
        data.ctrl[actuator_indices] = q_des

    for _ in range(max(1, int(frame_substeps))):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)


def get_robot_states(data, robot_handles):
    states = []
    for handle in robot_handles:
        qpos = np.asarray(data.qpos[handle["qpos_indices"]], dtype=np.float64).copy()
        ee_pos = np.asarray(data.site_xpos[handle["site_id"]], dtype=np.float64).copy()
        states.append(
            {
                "robot_index": int(handle["robot_index"]),
                "joint_values": qpos,
                "ee_pos": ee_pos,
            }
        )
    return states


def solve_joint_targets_for_ee_points(
    mujoco,
    model,
    data,
    robot_handles,
    target_points,
    *,
    ik_iterations: int = 60,
    damping: float = 1e-5,
    gain: float = 1.0,
    frame_substeps: int = 1,
    max_joint_speed=None,
):
    temp_data = mujoco.MjData(model)
    _copy_data_state(data, temp_data)
    mujoco.mj_forward(model, temp_data)
    jacp = np.zeros((3, int(model.nv)), dtype=np.float64)
    jacr = np.zeros((3, int(model.nv)), dtype=np.float64)
    joint_targets = []
    max_joint_delta = _resolve_joint_delta_limit(model, frame_substeps, max_joint_speed)

    for handle, target in zip(robot_handles, target_points):
        target = np.asarray(target, dtype=np.float64).reshape(3)
        qpos_indices = handle["qpos_indices"]
        current_qpos = np.asarray(temp_data.qpos[qpos_indices], dtype=np.float64).copy()
        for _ in range(max(1, int(ik_iterations))):
            current = np.asarray(temp_data.site_xpos[handle["site_id"]], dtype=np.float64)
            error = target - current
            if float(np.linalg.norm(error)) < 1e-4:
                break
            mujoco.mj_jacSite(model, temp_data, jacp, jacr, handle["site_id"])
            local_jacobian = jacp[:, handle["dof_indices"]]
            dq = _damped_least_squares_step(local_jacobian, error, damping=damping, gain=gain)
            updated_q = np.asarray(temp_data.qpos[qpos_indices], dtype=np.float64) + dq[: len(qpos_indices)]
            temp_data.qpos[qpos_indices] = np.clip(updated_q, handle["ctrl_min"], handle["ctrl_max"])
            mujoco.mj_forward(model, temp_data)
        solved_q = np.asarray(temp_data.qpos[qpos_indices], dtype=np.float64).copy()
        joint_targets.append(
            limit_joint_target_step(
                current_qpos,
                solved_q,
                handle["ctrl_min"],
                handle["ctrl_max"],
                max_joint_delta=max_joint_delta,
            )
        )

    return joint_targets


def _damped_least_squares_step(jacobian: np.ndarray, error: np.ndarray, damping: float, gain: float):
    jj_t = jacobian @ jacobian.T
    system = jj_t + float(damping) * np.eye(jj_t.shape[0], dtype=np.float64)
    delta = jacobian.T @ np.linalg.solve(system, error)
    return float(gain) * delta


def _append_trajectory_geoms_to_scene(mujoco, scn, ee_histories, traj_width, robot_palette):
    size_line = np.zeros((3,), dtype=np.float64)
    pos_line = np.zeros((3,), dtype=np.float64)
    mat_line = np.eye(3, dtype=np.float64).reshape(-1)
    geom_index = int(getattr(scn, "ngeom", 0))
    max_geoms = int(getattr(scn, "maxgeom", len(scn.geoms)))

    for robot_idx, traj in enumerate(ee_histories):
        color = robot_palette[robot_idx % len(robot_palette)]
        upto = len(traj)
        for point_idx in range(1, upto):
            if geom_index >= max_geoms:
                return
            geom = scn.geoms[geom_index]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE, size_line, pos_line, mat_line, color)
            p0 = np.asarray(traj[point_idx - 1], dtype=np.float64)
            p1 = np.asarray(traj[point_idx], dtype=np.float64)
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, float(traj_width), p0, p1)
            geom_index += 1
    scn.ngeom = geom_index


def _save_mujoco_recording(frames, output_path: str, fps: int):
    if not frames:
        return
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    fps = max(1, int(fps))

    if suffix == ".gif":
        try:
            from PIL import Image
        except ImportError as error:
            raise RuntimeError("Saving MuJoCo GIF requires Pillow to be installed.") from error
        quantize_method = getattr(getattr(Image, "Quantize", None), "FASTOCTREE", None)

        def _quantize_frame(frame_image, *, palette_image=None):
            if palette_image is not None:
                return frame_image.quantize(palette=palette_image, dither=getattr(Image, "Dither", Image).NONE)
            quantize_kwargs = {"colors": 255, "dither": getattr(Image, "Dither", Image).NONE}
            if quantize_method is not None:
                quantize_kwargs["method"] = quantize_method
            return frame_image.quantize(**quantize_kwargs)

        pil_frames = []
        palette_frame = None
        for frame in frames:
            rgb_frame = Image.fromarray(frame)
            palettized_frame = _quantize_frame(rgb_frame, palette_image=palette_frame)
            if palette_frame is None:
                palette_frame = palettized_frame
            pil_frames.append(palettized_frame)
        duration_ms = max(1, int(round(1000 / fps)))
        pil_frames[0].save(
            str(output),
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
            disposal=2,
        )
        return

    if suffix in {".mp4", ".m4v", ".mov", ".avi"}:
        try:
            import imageio.v2 as imageio
        except ImportError as error:
            raise RuntimeError("Saving MuJoCo video requires imageio and an ffmpeg backend.") from error
        imageio.mimsave(str(output), frames, fps=fps)
        return

    raise ValueError(f"Unsupported MuJoCo recording format: {output.suffix}. Use .gif or .mp4.")


def _drive_end_effectors_toward_targets(
    mujoco,
    model,
    data,
    robot_handles,
    target_points,
    *,
    ik_iterations: int,
    frame_substeps: int,
    damping: float,
    gain: float,
    max_joint_speed=None,
):
    jacp = np.zeros((3, int(model.nv)), dtype=np.float64)
    jacr = np.zeros((3, int(model.nv)), dtype=np.float64)
    max_joint_delta = _resolve_joint_delta_limit(model, 1, max_joint_speed)

    for _ in range(max(1, int(frame_substeps))):
        for handle, target in zip(robot_handles, target_points):
            target = np.asarray(target, dtype=np.float64).reshape(3)
            for _ in range(max(1, int(ik_iterations))):
                mujoco.mj_forward(model, data)
                current = np.asarray(data.site_xpos[handle["site_id"]], dtype=np.float64)
                error = target - current
                if float(np.linalg.norm(error)) < 1e-3:
                    break
                mujoco.mj_jacSite(model, data, jacp, jacr, handle["site_id"])
                local_jacobian = jacp[:, handle["dof_indices"]]
                dq = _damped_least_squares_step(local_jacobian, error, damping=damping, gain=gain)
                qpos_indices = handle["qpos_indices"]
                actuator_indices = handle["actuator_indices"]
                current_qpos = np.asarray(data.qpos[qpos_indices], dtype=np.float64)
                q_des = current_qpos + dq[: len(qpos_indices)]
                q_des = limit_joint_target_step(
                    current_qpos,
                    q_des,
                    handle["ctrl_min"],
                    handle["ctrl_max"],
                    max_joint_delta=max_joint_delta,
                )
                data.ctrl[actuator_indices] = q_des
            mujoco.mj_step(model, data)


def visualize_rollout_in_mujoco(
    xml_path: str,
    run: dict,
    *,
    title: str,
    playback_dt: float = 0.08,
    traj_width: float = 4.0,
    marker_radius: float = 0.035,
    exit_on_done: bool = False,
    ik_iterations: int = 40,
    frame_substeps: int = 250,
    ik_damping: float = 1e-5,
    ik_gain: float = 1.0,
    record_path: str | None = None,
    record_fps: int = 12,
    record_width: int = 1280,
    record_height: int = 720,
):
    mujoco, model, data, resolved_xml_path = load_model_and_data(xml_path)
    snapshots = list(run.get("snapshots", []))
    rollout_joint_targets = list(run.get("joint_targets", []))
    rollout_targets = list(run.get("target_points", []))
    playback_frame_substeps = int(max(1, run.get("control_substeps", frame_substeps)))
    if not snapshots:
        raise ValueError("run must contain 'snapshots' for MuJoCo playback.")
    robot_handles = _build_robot_handles(mujoco, model)
    if not robot_handles:
        raise ValueError("No robot kinematic chains with end_effector sites were found in the MuJoCo model.")
    ee_histories = [[] for _ in robot_handles]
    playback_max_joint_speed_values = _get_named_numeric_values(mujoco, model, "max_joint_speed")
    playback_max_joint_speed = None
    if playback_max_joint_speed_values is not None and playback_max_joint_speed_values.size > 0:
        playback_max_joint_speed = float(playback_max_joint_speed_values.reshape(-1)[0])

    # Align the playback model with the recorded reset state so the viewer does not
    # start from XML qpos0 and then lurch toward the first commanded target.
    first_snapshot_agents = list(snapshots[0].get("agents", [])) if snapshots else []
    for robot_idx, handle in enumerate(robot_handles):
        qpos_indices = handle["qpos_indices"]
        actuator_indices = handle["actuator_indices"]
        initial_qpos = None
        if robot_idx < len(first_snapshot_agents):
            snapshot_joint_values = np.asarray(first_snapshot_agents[robot_idx].get("joint_values", []), dtype=np.float64).reshape(-1)
            if snapshot_joint_values.size > 0:
                initial_qpos = snapshot_joint_values[: len(qpos_indices)]
        if initial_qpos is None or initial_qpos.size == 0:
            initial_qpos = np.asarray(data.qpos[qpos_indices], dtype=np.float64)
        initial_qpos = np.clip(initial_qpos, handle["ctrl_min"], handle["ctrl_max"])
        data.qpos[qpos_indices] = initial_qpos
        data.ctrl[actuator_indices] = initial_qpos[: len(actuator_indices)]
    mujoco.mj_forward(model, data)

    goal_heights = []
    for site_id in range(int(model.nsite)):
        site_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site_id) or ""
        if site_name.startswith("goal"):
            goal_heights.append(float(data.site_xpos[site_id][2]))
    target_z = float(np.mean(goal_heights)) if goal_heights else 0.30

    def _rgba(color_name: str, alpha: float = 1.0):
        mapping = {
            "blue": np.array([0.122, 0.467, 0.706, alpha], dtype=np.float32),
            "orange": np.array([1.0, 0.498, 0.055, alpha], dtype=np.float32),
            "green": np.array([0.173, 0.627, 0.173, alpha], dtype=np.float32),
            "red": np.array([0.839, 0.153, 0.157, alpha], dtype=np.float32),
            "brown": np.array([0.549, 0.337, 0.294, alpha], dtype=np.float32),
            "pink": np.array([0.89, 0.467, 0.761, alpha], dtype=np.float32),
            "gray": np.array([0.6, 0.6, 0.6, alpha], dtype=np.float32),
            "yellow": np.array([1.0, 0.85, 0.1, alpha], dtype=np.float32),
        }
        return mapping[color_name]

    robot_palette = [
        _rgba("blue"),
        _rgba("orange"),
        _rgba("green"),
        _rgba("red"),
        _rgba("brown"),
        _rgba("pink"),
    ]
    size_line = np.zeros((3,), dtype=np.float64)
    pos_line = np.zeros((3,), dtype=np.float64)
    mat_line = np.eye(3, dtype=np.float64).reshape(-1)
    recorded_frames = []
    renderer = None
    render_camera = None
    if record_path:
        render_width, render_height = _resolve_offscreen_render_size(model, record_width, record_height)
        if render_width != int(record_width) or render_height != int(record_height):
            print(
                f"[WARN] MuJoCo recording size clamped from {int(record_width)}x{int(record_height)} "
                f"to {render_width}x{render_height} by XML offscreen framebuffer limits."
            )
        renderer = mujoco.Renderer(model, height=render_height, width=render_width)
        render_camera = mujoco.MjvCamera()
        _configure_free_camera_from_xml_defaults(mujoco, model, render_camera)

    xml_dir = os.path.dirname(resolved_xml_path)
    with _pushd(xml_dir):
        with mujoco.viewer.launch_passive(model, data) as viewer:
            apply_xml_view_defaults(mujoco, model, viewer)
            finished_text = None
            for frame_idx, snapshot in enumerate(snapshots):
                if not viewer.is_running():
                    break

                if frame_idx > 0:
                    frame_joint_targets = rollout_joint_targets[frame_idx - 1] if frame_idx <= len(rollout_joint_targets) else None
                    if frame_joint_targets is not None:
                        apply_joint_targets(
                            mujoco,
                            model,
                            data,
                            robot_handles[: len(frame_joint_targets)],
                            frame_joint_targets,
                            frame_substeps=playback_frame_substeps,
                            max_joint_speed=playback_max_joint_speed,
                        )
                    else:
                        target_points = []
                        frame_targets = rollout_targets[frame_idx - 1] if frame_idx <= len(rollout_targets) else None
                        for handle in robot_handles:
                            robot_idx = int(handle["robot_index"]) - 1
                            if frame_targets is not None and robot_idx < len(frame_targets):
                                target = np.asarray(frame_targets[robot_idx], dtype=np.float64).reshape(-1)
                                if target.shape[0] >= 3:
                                    target_points.append(np.array([target[0], target[1], target[2]], dtype=np.float64))
                                elif target.shape[0] >= 2:
                                    target_points.append(np.array([target[0], target[1], target_z], dtype=np.float64))
                                else:
                                    current = np.asarray(data.site_xpos[handle["site_id"]], dtype=np.float64)
                                    target_points.append(current)
                            elif robot_idx < len(snapshot.get("agents", [])):
                                agent_pos = snapshot["agents"][robot_idx]["pos"]
                                snapshot_goal_height = float(snapshot.get("goal_height", target_z))
                                target_points.append(np.array([agent_pos[0], agent_pos[1], snapshot_goal_height], dtype=np.float64))
                            else:
                                current = np.asarray(data.site_xpos[handle["site_id"]], dtype=np.float64)
                                target_points.append(current)

                        _drive_end_effectors_toward_targets(
                            mujoco,
                            model,
                            data,
                            robot_handles,
                            target_points,
                            ik_iterations=ik_iterations,
                            frame_substeps=playback_frame_substeps,
                            damping=ik_damping,
                            gain=ik_gain,
                        )

                for history, handle in zip(ee_histories, robot_handles):
                    history.append(np.asarray(data.site_xpos[handle["site_id"]], dtype=np.float64).copy())

                with viewer.lock():
                    scn = viewer.user_scn
                    if scn is not None:
                        scn.ngeom = 0
                        _append_trajectory_geoms_to_scene(mujoco, scn, ee_histories, traj_width, robot_palette)

                if renderer is not None and render_camera is not None:
                    renderer.update_scene(data, camera=render_camera)
                    _append_trajectory_geoms_to_scene(mujoco, renderer.scene, ee_histories, traj_width, robot_palette)
                    recorded_frames.append(np.asarray(renderer.render(), dtype=np.uint8).copy())

                completed_tasks = sum(int(task.get("completed", False)) for task in snapshot.get("tasks", []))
                total_tasks = len(snapshot.get("tasks", []))
                info_text = f"step={snapshot.get('step', frame_idx)} completed={completed_tasks}/{total_tasks}"
                try:
                    viewer.set_texts((None, None, info_text, title))
                except Exception:
                    pass
                viewer.sync()
                time.sleep(max(0.0, float(playback_dt)))
                finished_text = f"{title} | {info_text}"

            if viewer.is_running() and not bool(exit_on_done):
                try:
                    viewer.set_texts((None, None, finished_text or title, "(close window to exit)"))
                except Exception:
                    pass
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(0.01)

    if renderer is not None:
        renderer.close()
    if record_path and recorded_frames:
        _save_mujoco_recording(recorded_frames, record_path, record_fps)
        print(f"[SAVE] mujoco_recording={record_path}")


__all__ = [
    "_build_robot_initial_joint_targets",
    "apply_xml_view_defaults",
    "load_layout_templates_from_mujoco",
    "load_model_and_data",
    "visualize_rollout_in_mujoco",
]