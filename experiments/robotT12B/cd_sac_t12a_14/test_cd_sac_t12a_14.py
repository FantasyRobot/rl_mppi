#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover
    plt = None

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))
for _p in (_THIS_DIR, _ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_RESULTS_DIR = os.path.join(_ROOT_DIR, "experiments", "results")


@contextmanager
def _pushd(path: str):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _resolve_plot_path(plot_path: str) -> str:
    plot_path = os.path.expanduser(os.path.expandvars(str(plot_path)))
    if not os.path.isabs(plot_path) and os.path.dirname(plot_path) == "":
        plot_path = os.path.join(_RESULTS_DIR, plot_path)
    out_dir = os.path.dirname(plot_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    return plot_path


def _derive_signals_plot_path(plot_path: str) -> str:
    root, ext = os.path.splitext(str(plot_path))
    if ext.strip() == "":
        ext = ".png"
    return f"{root}_signals{ext}"


def _save_joint_signals_csv(*, base_path: str, qvel: np.ndarray, qacc: np.ndarray, meta: dict) -> tuple[str, str, str]:
    base_path = os.path.expanduser(os.path.expandvars(str(base_path)))
    if base_path.lower().endswith(".csv"):
        base_path = base_path[:-4]
    out_dir = os.path.dirname(os.path.abspath(base_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    qvel_path = f"{base_path}_qvel.csv"
    qacc_path = f"{base_path}_qacc.csv"
    meta_path = f"{base_path}_meta.txt"

    np.savetxt(qvel_path, np.asarray(qvel, dtype=np.float64), delimiter=",")
    np.savetxt(qacc_path, np.asarray(qacc, dtype=np.float64), delimiter=",")
    with open(meta_path, "w", encoding="utf-8") as f:
        for k, v in meta.items():
            f.write(f"{k}: {v}\n")
    return qvel_path, qacc_path, meta_path


def load_model(save_path: str, obs_dim: int, action_dim: int):
    try:
        import torch
    except ModuleNotFoundError as e:
        raise SystemExit(
            "Torch is required for loading. Install it in your env, e.g. (CPU):\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
        ) from e

    from algorithms.sac.sac_utils import SACAgent

    if not os.path.exists(save_path):
        raise FileNotFoundError(f"Model file not found: {save_path}")

    checkpoint = torch.load(save_path, map_location="cpu")
    auto_entropy_tuning = bool(checkpoint.get("auto_entropy_tuning", False))

    agent = SACAgent(
        state_dim=int(obs_dim),
        action_dim=int(action_dim),
        hidden_dim=256,
        learning_rate=3e-4,
        alpha=float(checkpoint.get("alpha", 0.2)),
        gamma=0.99,
        tau=0.005,
        auto_entropy_tuning=auto_entropy_tuning,
        use_lr_scheduler=False,
    )

    agent.policy_net.load_state_dict(checkpoint["policy_state_dict"])
    agent.q_net1.load_state_dict(checkpoint["q1_state_dict"])
    agent.q_net2.load_state_dict(checkpoint["q2_state_dict"])
    agent.target_q_net1.load_state_dict(checkpoint["target_q1_state_dict"])
    agent.target_q_net2.load_state_dict(checkpoint["target_q2_state_dict"])

    agent.alpha = float(checkpoint.get("alpha", agent.alpha))
    if agent.auto_entropy_tuning:
        log_alpha = checkpoint.get("log_alpha", None)
        if log_alpha is not None:
            if isinstance(log_alpha, torch.Tensor):
                agent.log_alpha = log_alpha.to(torch.float32)
                agent.alpha = float(agent.log_alpha.exp().item())
            else:
                agent.log_alpha = torch.tensor(float(log_alpha), requires_grad=False)
                agent.alpha = float(agent.log_alpha.exp().item())

    return agent, checkpoint


def test_cd_sac_t12a_14(
    *,
    model_path: str,
    xml_path: str,
    goal_site: str = "goal",
    eef_site: str = "end_effector",
    num_tests: int = 5,
    max_steps: int = 2500,
    action_repeat: int = 5,
    plot_path: str = os.path.join(_RESULTS_DIR, "cd_sac_t12a_14_signals.png"),
    show_plot: bool = False,
    vel_bound: float | None = None,
    acc_bound: float | None = None,
):
    from env.envmujoco_t12a_14_constraints import T12A14MuJoCoEnvConstraints

    plot_path = _resolve_plot_path(plot_path)

    # Create a temp env to get dims, and read constraint config from checkpoint.
    tmp_env = T12A14MuJoCoEnvConstraints(
        xml_path=str(xml_path),
        goal_site=str(goal_site),
        eef_site=str(eef_site),
        max_steps=int(max_steps),
        action_repeat=int(action_repeat),
    )

    agent, checkpoint = load_model(str(model_path), tmp_env.obs_dim, tmp_env.action_dim)

    ckpt_vel = float(checkpoint.get("vel_bound", tmp_env.vel_bound))
    ckpt_acc = float(checkpoint.get("acc_bound", tmp_env.acc_bound))

    if vel_bound is None:
        vel_bound = ckpt_vel
    if acc_bound is None:
        acc_bound = ckpt_acc

    if float(vel_bound) != float(ckpt_vel) or float(acc_bound) != float(ckpt_acc):
        print(
            "[WARN] Test bounds override differs from checkpoint: "
            f"ckpt(vel={ckpt_vel}, acc={ckpt_acc}) vs test(vel={vel_bound}, acc={acc_bound})."
        )

    env = T12A14MuJoCoEnvConstraints(
        xml_path=str(xml_path),
        goal_site=str(goal_site),
        eef_site=str(eef_site),
        max_steps=int(max_steps),
        action_repeat=int(action_repeat),
        vel_bound=float(vel_bound),
        acc_bound=float(acc_bound),
    )

    rollouts: list[dict[str, np.ndarray]] = []
    successes = 0
    violations = 0

    for k in range(int(num_tests)):
        obs = env.reset()

        ts_dist: list[float] = []
        ts_reward: list[float] = []
        ts_vmax: list[float] = []
        ts_amax: list[float] = []

        # full joint signals (may be large); keep for optional detailed plots
        ts_qpos: list[np.ndarray] = []
        ts_qvel: list[np.ndarray] = []

        prev_qvel = np.asarray(env.data.qvel[: env.nu], dtype=np.float64).copy()
        dt_eff = float(env.model.opt.timestep) * float(env.action_repeat)
        if dt_eff <= 0:
            dt_eff = 1.0

        while True:
            a = agent.select_action(obs, evaluate=True)
            obs2, r, done, info = env.step(a)

            qpos = np.asarray(info.get("qpos"), dtype=np.float32)
            qvel = np.asarray(info.get("qvel"), dtype=np.float32)
            qacc = (np.asarray(qvel, dtype=np.float64) - prev_qvel) / dt_eff
            prev_qvel = np.asarray(qvel, dtype=np.float64)

            ts_dist.append(float(info.get("dist", 0.0)))
            ts_reward.append(float(r))
            ts_vmax.append(float(np.max(np.abs(qvel))))
            ts_amax.append(float(np.max(np.abs(qacc))))
            ts_qpos.append(qpos.copy())
            ts_qvel.append(qvel.copy())

            obs = obs2
            if done:
                if bool(info.get("constraint_violation", False)):
                    violations += 1
                if bool(info.get("success", False)):
                    successes += 1
                break

        rollouts.append(
            {
                "dist": np.asarray(ts_dist, dtype=np.float32),
                "reward": np.asarray(ts_reward, dtype=np.float32),
                "vmax": np.asarray(ts_vmax, dtype=np.float32),
                "amax": np.asarray(ts_amax, dtype=np.float32),
                "qpos": np.asarray(ts_qpos, dtype=np.float32),
                "qvel": np.asarray(ts_qvel, dtype=np.float32),
            }
        )

    print(f"Success Rate: {successes}/{num_tests} ({(successes/max(1,int(num_tests))*100.0):.1f}%)")
    print(f"Violation Rate (episode-level): {violations}/{num_tests} ({(violations/max(1,int(num_tests))*100.0):.1f}%)")

    if plt is None:
        print("[WARN] matplotlib is not installed; skipping plots. Install with: pip install matplotlib")
        return rollouts

    # Plot distance and reward.
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    for i, ro in enumerate(rollouts):
        ax1.plot(ro["dist"], linewidth=1.6, alpha=0.8, label="" if i else "dist")
        ax2.plot(ro["reward"], linewidth=1.6, alpha=0.8, label="" if i else "reward")

    ax1.set_title("CD-SAC T12A14: distance to goal")
    ax1.set_xlabel("step")
    ax1.set_ylabel("dist (m)")
    ax1.grid(True, alpha=0.3)

    ax2.set_title("CD-SAC T12A14: reward")
    ax2.set_xlabel("step")
    ax2.set_ylabel("reward")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")

    # Plot signals vmax/amax with bounds.
    signals_plot_path = _derive_signals_plot_path(plot_path)
    fig_s = plt.figure(figsize=(12, 6))
    ax_v = fig_s.add_subplot(2, 1, 1)
    ax_a = fig_s.add_subplot(2, 1, 2)

    for i, ro in enumerate(rollouts):
        ax_v.plot(ro["vmax"], linewidth=1.6, alpha=0.8)
        ax_a.plot(ro["amax"], linewidth=1.6, alpha=0.8)

    ax_v.axhline(float(vel_bound), color="red", linestyle="--", linewidth=1.2, label="vel_bound")
    ax_a.axhline(float(acc_bound), color="red", linestyle="--", linewidth=1.2, label="acc_bound")

    ax_v.set_title("Max |qvel| across joints")
    ax_v.set_xlabel("step")
    ax_v.set_ylabel("max |qvel|")
    ax_v.grid(True, alpha=0.3)

    ax_a.set_title("Max |qacc| across joints (finite-diff)")
    ax_a.set_xlabel("step")
    ax_a.set_ylabel("max |qacc|")
    ax_a.grid(True, alpha=0.3)

    fig_s.tight_layout()
    fig_s.savefig(signals_plot_path, dpi=150)
    print(f"Signals plot saved to {signals_plot_path}")

    if show_plot:
        plt.show()
    plt.close(fig)
    plt.close(fig_s)

    return rollouts


def test_cd_sac_t12a_14_viewer(
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
    sleep_sec: float = 0.0,
    export_csv_base: str | None = None,
    settle_steps: int = 0,
    vel_tol: float = 0.02,
) -> dict:
    """Run CD-SAC policy in a MuJoCo viewer and draw the end-effector trajectory."""

    from env.envmujoco_t12a_14_constraints import T12A14MuJoCoEnvConstraints

    try:
        import mujoco
        import mujoco.viewer
    except ModuleNotFoundError:
        raise SystemExit("Python package 'mujoco' is not installed. Install with: pip install mujoco")

    import glfw

    # Create temp env to get dims and checkpoint config.
    tmp_env = T12A14MuJoCoEnvConstraints(
        xml_path=str(xml_path),
        goal_site=str(goal_site),
        eef_site=str(eef_site),
        max_steps=int(max_steps),
        action_repeat=int(action_repeat),
    )
    agent, ckpt = load_model(str(model_path), tmp_env.obs_dim, tmp_env.action_dim)

    vel_bound = float(ckpt.get("vel_bound", tmp_env.vel_bound))
    acc_bound = float(ckpt.get("acc_bound", tmp_env.acc_bound))

    env = T12A14MuJoCoEnvConstraints(
        xml_path=str(xml_path),
        goal_site=str(goal_site),
        eef_site=str(eef_site),
        max_steps=int(max_steps),
        action_repeat=int(action_repeat),
        vel_bound=float(vel_bound),
        acc_bound=float(acc_bound),
    )

    def _apply_xml_view_defaults(v) -> None:
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
        print("  XML <statistic .../> suggestion:")
        print(f"    <statistic center=\"{lookat[0]:.4f} {lookat[1]:.4f} {lookat[2]:.4f}\" extent=\"{float(cam.distance):.4f}\"/>")

    def _key_callback(key: int) -> None:
        if key == int(glfw.KEY_P):
            _print_view()

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

    obs = env.reset()
    traj_points: list[np.ndarray] = []
    eef_traj: list[np.ndarray] = []
    dist_traj: list[float] = []
    reward_traj: list[float] = []

    qvel_traj: list[np.ndarray] = []
    qacc_traj: list[np.ndarray] = []
    prev_qvel = np.asarray(env.data.qvel[: env.nu], dtype=np.float64).copy()
    dt_eff = float(env.model.opt.timestep) * float(env.action_repeat)
    if dt_eff <= 0:
        dt_eff = 1.0

    xml_path_abs = os.path.abspath(os.path.expandvars(os.path.expanduser(str(xml_path))))
    xml_dir = os.path.dirname(xml_path_abs)

    with _pushd(xml_dir):
        with mujoco.viewer.launch_passive(env.model, env.data, key_callback=_key_callback) as viewer:
            viewer_ref["viewer"] = viewer
            _apply_xml_view_defaults(viewer)

            finished_reason: str | None = None
            for k in range(int(max_steps)):
                if not viewer.is_running():
                    break

                a = agent.select_action(obs, evaluate=True)
                obs, r, done, info = env.step(a)

                eef = np.asarray(info["eef"], dtype=np.float64)
                dist = float(info["dist"])
                eef_traj.append(eef.astype(np.float32))
                dist_traj.append(dist)
                reward_traj.append(float(r))

                qvel = np.asarray(info.get("qvel"), dtype=np.float64)
                qacc = (qvel - prev_qvel) / dt_eff
                prev_qvel = qvel
                qvel_traj.append(qvel.astype(np.float32))
                qacc_traj.append(qacc.astype(np.float32))

                if not bool(no_draw_traj):
                    stride = max(1, int(traj_stride))
                    if k % stride == 0:
                        traj_points.append(eef)
                        maxp = max(2, int(traj_max_points))
                        if len(traj_points) > maxp:
                            traj_points = traj_points[-maxp:]
                        update_traj_overlay(viewer, traj_points)

                try:
                    vio = int(bool(info.get("constraint_violation", False)))
                    viewer.set_texts((None, None, f"dist={dist:.4f} vio={vio}", None))
                except Exception:
                    pass

                viewer.sync()
                if float(sleep_sec) > 0:
                    time.sleep(float(sleep_sec))

                if done:
                    finished_reason = (
                        f"Done (dist={dist:.4f}, success={bool(info.get('success', False))}, "
                        f"violation={bool(info.get('constraint_violation', False))})"
                    )
                    break

            # Optional settling: keep simulating while holding current joint targets so qvel decays.
            if viewer.is_running() and bool(info.get("success", False)) and int(settle_steps) > 0:
                hold_ctrl = np.asarray(env.data.qpos[: env.nu], dtype=np.float64).copy()
                hold_ctrl = np.clip(hold_ctrl, env.ctrl_min, env.ctrl_max)
                hold_action = env.ctrl_to_action(hold_ctrl)

                for j in range(int(settle_steps)):
                    if not viewer.is_running():
                        break

                    obs, r, done2, info2 = env.step(hold_action)

                    eef = np.asarray(info2["eef"], dtype=np.float64)
                    dist = float(info2["dist"])
                    eef_traj.append(eef.astype(np.float32))
                    dist_traj.append(dist)
                    reward_traj.append(float(r))

                    qvel = np.asarray(info2.get("qvel"), dtype=np.float64)
                    qacc = (qvel - prev_qvel) / dt_eff
                    prev_qvel = qvel
                    qvel_traj.append(qvel.astype(np.float32))
                    qacc_traj.append(qacc.astype(np.float32))

                    if not bool(no_draw_traj):
                        stride = max(1, int(traj_stride))
                        if j % stride == 0:
                            traj_points.append(eef)
                            maxp = max(2, int(traj_max_points))
                            if len(traj_points) > maxp:
                                traj_points = traj_points[-maxp:]
                            update_traj_overlay(viewer, traj_points)

                    vmax = float(np.max(np.abs(qvel))) if qvel.size else 0.0
                    try:
                        viewer.set_texts((None, None, f"settling {j+1}/{int(settle_steps)} dist={dist:.4f} vmax={vmax:.3f}", None))
                    except Exception:
                        pass

                    viewer.sync()
                    if float(sleep_sec) > 0:
                        time.sleep(float(sleep_sec))

                    if vmax < float(vel_tol):
                        finished_reason = f"Settled (vmax={vmax:.4f})"
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

    if export_csv_base is not None and str(export_csv_base).strip() != "" and qvel_traj:
        base = str(export_csv_base)
        qvel_arr = np.asarray(qvel_traj, dtype=np.float32)
        qacc_arr = np.asarray(qacc_traj, dtype=np.float32)
        meta = {
            "dt_eff": float(dt_eff),
            "action_repeat": int(env.action_repeat),
            "timestep": float(env.model.opt.timestep),
            "vel_bound": float(env.vel_bound),
            "acc_bound": float(env.acc_bound),
            "nu": int(env.nu),
            "steps": int(qvel_arr.shape[0]),
            "xml_path": str(xml_path),
            "model_path": str(model_path),
        }
        qvel_path, qacc_path, meta_path = _save_joint_signals_csv(base_path=base, qvel=qvel_arr, qacc=qacc_arr, meta=meta)
        print(f"[EXPORT] qvel -> {qvel_path}")
        print(f"[EXPORT] qacc -> {qacc_path}")
        print(f"[EXPORT] meta -> {meta_path}")

    success = bool(info.get("success", False)) if dist_traj else False
    final_dist = float(dist_traj[-1]) if dist_traj else float("inf")
    return {
        "eef": np.asarray(eef_traj, dtype=np.float32),
        "dist": np.asarray(dist_traj, dtype=np.float32),
        "reward": np.asarray(reward_traj, dtype=np.float32),
        "qvel": np.asarray(qvel_traj, dtype=np.float32),
        "qacc": np.asarray(qacc_traj, dtype=np.float32),
        "success": bool(success),
        "final_dist": float(final_dist),
        "ckpt": ckpt,
    }


if __name__ == "__main__":
    raise SystemExit("Use cd_sac_t12a_14_cli.py test")
