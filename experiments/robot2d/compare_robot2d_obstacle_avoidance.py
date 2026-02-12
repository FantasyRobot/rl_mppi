#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover
    plt = None

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

_RESULTS_DIR = os.path.join(_ROOT_DIR, "experiments", "results")

from env.envrobot2d_obstacles import CircleObstacle, Robot2DEnvironmentObstacles
from algorithms.mppi.mppi_robot2d import MPPI
from algorithms.rl_mppi.rl_mppi_robot2d import (
    CDFCollisionRecoveryPolicy,
    IKHeuristicPolicy,
    PolicyWrapper,
    RLMppiController,
    load_sac_policy,
)


def _resolve_plot_path(plot_path: str) -> str:
    plot_path = os.path.expanduser(os.path.expandvars(str(plot_path)))
    if not os.path.isabs(plot_path) and os.path.dirname(plot_path) == "":
        plot_path = os.path.join(_RESULTS_DIR, plot_path)
    out_dir = os.path.dirname(plot_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    return plot_path


def run_episode(env: Robot2DEnvironmentObstacles, controller, *, initial_state: np.ndarray | None = None) -> dict:
    s = env.reset(initial_state=initial_state)

    eef = env.forward_kinematics_eef(s[: env.n])
    traj = [eef.copy()]
    q_traj = [np.asarray(s[: env.n], dtype=np.float32).copy()]
    qd_traj = [np.asarray(s[env.n : 2 * env.n], dtype=np.float32).copy()]
    clearances: list[float] = []

    t0 = time.perf_counter()
    total_reward = 0.0
    steps = 0
    hit = False

    while True:
        eef = env.forward_kinematics_eef(s[: env.n])
        if float(np.linalg.norm(eef - env.target_pos)) < float(env.reach_threshold):
            break

        a = controller.get_action(s, env.target_pos)
        s, r, done, info = env.step(a)
        total_reward += float(r)
        steps += 1

        traj.append(np.asarray(info.get("eef", env.forward_kinematics_eef(s[: env.n])), dtype=np.float32))
        q_traj.append(np.asarray(s[: env.n], dtype=np.float32).copy())
        qd_traj.append(np.asarray(s[env.n : 2 * env.n], dtype=np.float32).copy())
        clearances.append(float(info.get("min_obstacle_clearance", float("inf"))))
        if bool(info.get("hit_obstacle", False)):
            hit = True

        if done:
            break

    dt = time.perf_counter() - t0
    final_eef = env.forward_kinematics_eef(s[: env.n])
    final_dist = float(np.linalg.norm(final_eef - env.target_pos))
    success = final_dist < float(env.reach_threshold)

    q_arr = np.asarray(q_traj, dtype=np.float32)
    qd_arr = np.asarray(qd_traj, dtype=np.float32)
    dt = float(getattr(env, "dt", 1.0))
    if dt <= 0.0:
        dt = 1.0
    qdd_arr = np.zeros_like(qd_arr)
    if qd_arr.shape[0] >= 2:
        qdd_arr[1:] = (qd_arr[1:] - qd_arr[:-1]) / dt

    traj_arr = np.asarray(traj, dtype=np.float32)
    dist_arr = np.linalg.norm(traj_arr - np.asarray(env.target_pos, dtype=np.float32).reshape(1, 2), axis=1).astype(np.float32)
    t_arr = (np.arange(q_arr.shape[0], dtype=np.float32) * dt).astype(np.float32)

    return {
        "success": bool(success),
        "hit_obstacle": bool(hit),
        "steps": int(steps),
        "total_reward": float(total_reward),
        "final_dist": float(final_dist),
        "traj": traj_arr,
        "q_traj": q_arr,
        "qd_traj": qd_arr,
        "qdd_traj": qdd_arr,
        "dist_traj": dist_arr,
        "t": t_arr,
        "min_clearance": float(np.min(clearances)) if clearances else float("inf"),
        "episode_time_s": float(dt),
    }


def plot_timeseries(*, results: dict[str, dict], env: Robot2DEnvironmentObstacles, plot_path: str, show: bool) -> None:
    if plt is None:
        print("[WARN] matplotlib not installed; skipping time-series plots")
        return

    plot_path = _resolve_plot_path(plot_path)
    base, ext = os.path.splitext(plot_path)
    ts_path = base + "_timeseries" + (ext if ext else ".png")

    names = list(results.keys())
    n = int(env.n)
    fig, axes = plt.subplots(4, n, figsize=(5.2 * max(1, n), 9.2), squeeze=False)

    colors = {"MPPI": "tab:blue", "SAC": "tab:green", "RL-MPPI": "tab:orange"}
    styles = {"MPPI": "-", "SAC": ":", "RL-MPPI": "--"}

    def _plot_joint_row(row: int, key: str, ylabel_prefix: str) -> None:
        for j in range(n):
            ax = axes[row][j]
            for name in names:
                ep = results[name]
                t = np.asarray(ep.get("t", np.arange(ep[key].shape[0], dtype=np.float32) * float(getattr(env, "dt", 1.0))), dtype=np.float32)
                y = np.asarray(ep[key], dtype=np.float32)
                ax.plot(
                    t,
                    y[:, j],
                    linestyle=styles.get(name, "-"),
                    color=colors.get(name, None),
                    linewidth=1.8,
                    label=name if j == 0 else None,
                )
            ax.set_xlabel("t (s)")
            ax.grid(True, alpha=0.35)
            ax.set_title(f"{ylabel_prefix}{j}")

    _plot_joint_row(0, "q_traj", "q[")
    _plot_joint_row(1, "qd_traj", "qd[")
    _plot_joint_row(2, "qdd_traj", "qdd[")

    # Distance-to-goal (plot in first column; hide the rest).
    axd = axes[3][0]
    for name in names:
        ep = results[name]
        t = np.asarray(ep.get("t", np.arange(ep["dist_traj"].shape[0], dtype=np.float32) * float(getattr(env, "dt", 1.0))), dtype=np.float32)
        d = np.asarray(ep["dist_traj"], dtype=np.float32)
        axd.plot(
            t,
            d,
            linestyle=styles.get(name, "-"),
            color=colors.get(name, None),
            linewidth=2.2,
            label=name,
        )
    axd.set_title("||eef - goal||")
    axd.set_xlabel("t (s)")
    axd.grid(True, alpha=0.35)
    axd.legend(loc="upper right")

    for j in range(1, n):
        axes[3][j].axis("off")

    fig.tight_layout()
    fig.savefig(ts_path)
    print(f"Time-series plot saved to {ts_path}")
    if show:
        plt.show()
    plt.close(fig)


def export_mat(*, results: dict[str, dict], env: Robot2DEnvironmentObstacles, plot_path: str, mat_path: str) -> str | None:
    """Export compare results for MATLAB plotting.

    Produces a .mat file containing obstacles/target and per-method time-series.
    """

    mat_path = os.path.expanduser(os.path.expandvars(str(mat_path)))
    if not mat_path:
        base, _ = os.path.splitext(_resolve_plot_path(plot_path))
        mat_path = base + ".mat"
    if not os.path.isabs(mat_path) and os.path.dirname(mat_path) == "":
        mat_path = os.path.join(_RESULTS_DIR, mat_path)
    out_dir = os.path.dirname(mat_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    try:
        from scipy.io import savemat
    except ModuleNotFoundError:
        print("[WARN] scipy not installed; cannot export .mat. Install with: pip install scipy")
        return None

    # MATLAB-friendly names.
    name_map = {
        "MPPI": "MPPI",
        "SAC": "SAC",
        "RL-MPPI": "RLMPPI",
        "RLMPPI": "RLMPPI",
    }

    payload: dict[str, object] = {
        "dt": float(getattr(env, "dt", 0.0)),
        "n": int(getattr(env, "n", 0)),
        "link_lengths": np.asarray(getattr(env, "link_lengths", []), dtype=np.float32),
        "target_pos": np.asarray(getattr(env, "target_pos", np.zeros((2,), dtype=np.float32)), dtype=np.float32),
        "obstacles": np.asarray([[float(o.x), float(o.y), float(o.r)] for o in getattr(env, "obstacles", [])], dtype=np.float32),
    }

    for k, ep in results.items():
        mk = name_map.get(str(k), str(k).replace("-", ""))
        payload[f"{mk}_t"] = np.asarray(ep.get("t", []), dtype=np.float32).reshape(-1, 1)
        payload[f"{mk}_eef"] = np.asarray(ep.get("traj", []), dtype=np.float32)
        payload[f"{mk}_q"] = np.asarray(ep.get("q_traj", []), dtype=np.float32)
        payload[f"{mk}_qd"] = np.asarray(ep.get("qd_traj", []), dtype=np.float32)
        payload[f"{mk}_qdd"] = np.asarray(ep.get("qdd_traj", []), dtype=np.float32)
        payload[f"{mk}_dist"] = np.asarray(ep.get("dist_traj", []), dtype=np.float32).reshape(-1, 1)

    savemat(mat_path, payload, do_compression=True)
    print(f"MATLAB .mat exported to {mat_path}")
    return mat_path


def _add_contour_avoidance_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--contour_avoidance",
        action="store_true",
        help="Enable velocity shaping near obstacles (project joint velocity along clearance contours).",
    )
    p.add_argument(
        "--no_contour_avoidance",
        action="store_true",
        help="Disable contour avoidance (overrides --contour_avoidance).",
    )
    p.add_argument(
        "--contour_clearance_start",
        type=float,
        default=0.25,
        help="Start shaping when min clearance < this value.",
    )
    p.add_argument(
        "--contour_clearance_full",
        type=float,
        default=0.08,
        help="Full shaping when min clearance <= this value.",
    )
    p.add_argument(
        "--contour_repulse_gain",
        type=float,
        default=0.0,
        help="Optional repulsive normal gain added near obstacles (0 disables).",
    )
    p.add_argument(
        "--contour_fd_eps",
        type=float,
        default=1e-3,
        help="Finite-difference epsilon for clearance gradient (joint space).",
    )
    p.add_argument(
        "--contour_mode",
        type=str,
        default="clearance",
        choices=["clearance", "cdf"],
        help="Contour source: 'clearance' (task-space minimum clearance) or 'cdf' (configuration-space distance field).",
    )
    p.add_argument(
        "--contour_cdf_method",
        type=str,
        default="offline_grid",
        choices=["offline_grid", "online_computation"],
        help="CDF method when --contour_mode=cdf.",
    )


def _draw_obstacles(ax, env: Robot2DEnvironmentObstacles) -> None:
    if plt is None:
        return
    for obs in env.obstacles:
        circ = plt.Circle((obs.x, obs.y), obs.r, color="gray", alpha=0.35)
        ax.add_patch(circ)
        circ2 = plt.Circle(
            (obs.x, obs.y),
            obs.r + env.obstacle_margin,
            color="gray",
            alpha=0.15,
            fill=False,
            linestyle="--",
        )
        ax.add_patch(circ2)


def _draw_robot_links(
    ax,
    env: Robot2DEnvironmentObstacles,
    q: np.ndarray,
    *,
    color: str,
    alpha: float,
    lw: float = 2.5,
    marker_size: float = 2.0,
) -> None:
    pts = env.forward_kinematics_all_joints(q)
    ax.plot(pts[:, 0], pts[:, 1], "-", color=color, alpha=float(alpha), linewidth=float(lw))
    ax.plot(pts[:, 0], pts[:, 1], "o", color=color, alpha=float(alpha), markersize=float(marker_size))


def plot_results(
    *,
    results: dict[str, dict],
    env: Robot2DEnvironmentObstacles,
    plot_path: str,
    show: bool,
    draw_links: bool,
    num_link_frames: int,
) -> None:
    if plt is None:
        print("[WARN] matplotlib not installed; skipping plots")
        return

    plot_path = _resolve_plot_path(plot_path)

    names = list(results.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(6.5 * len(names), 6.5), squeeze=False)
    axes = axes[0]

    colors = {"MPPI": "tab:blue", "SAC": "tab:green", "RL-MPPI": "tab:orange"}
    styles = {"MPPI": "-", "SAC": ":", "RL-MPPI": "--"}

    link_total = float(np.sum(env.link_lengths))
    xlim = (-link_total - 0.6, link_total + 0.6)
    ylim = (-link_total - 0.6, link_total + 0.6)

    for ax, name in zip(axes, names, strict=False):
        ep = results[name]
        traj = np.asarray(ep["traj"], dtype=np.float32)
        q_traj = np.asarray(ep.get("q_traj", np.zeros((0, env.n), dtype=np.float32)), dtype=np.float32)

        _draw_obstacles(ax, env)

        # End-effector path.
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            linestyle=styles.get(name, "-"),
            color=colors.get(name, None),
            linewidth=2.2,
            label=f"{name} (EEF)",
        )
        ax.plot(traj[0, 0], traj[0, 1], "s", markersize=7, color="black", label="Start")
        ax.plot(float(env.target_pos[0]), float(env.target_pos[1]), "rx", markersize=12, label="Goal")

        # Robot link shapes along the rollout (like cdf_shooting).
        if draw_links and q_traj.size > 0:
            k = int(max(2, num_link_frames))
            k = int(min(k, int(q_traj.shape[0])))
            idx = np.linspace(0, q_traj.shape[0] - 1, num=k, dtype=int)
            idx = np.unique(idx)
            for j, t in enumerate(idx):
                alpha = 0.10 + 0.85 * (j / max(1, (len(idx) - 1)))
                _draw_robot_links(
                    ax,
                    env,
                    q_traj[int(t)],
                    color=colors.get(name, "tab:blue"),
                    alpha=float(alpha),
                    lw=2.0,
                    marker_size=2.0,
                )

        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.35)
        ax.set_title(f"Robot2D: {name}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Robot2D obstacle avoidance: MPPI vs RL-MPPI")
    p.add_argument("--target_x", type=float, default=-2.8)
    p.add_argument("--target_y", type=float, default=1.8)
    p.add_argument("--max_steps", type=int, default=450)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plot_path", type=str, default=os.path.join(_RESULTS_DIR, "robot2d_obstacle_compare.png"))
    p.add_argument("--show_plot", action="store_true")

    p.add_argument(
        "--export_mat",
        type=int,
        default=1,
        help="Export results to a MATLAB .mat file (1/0). Default: 1",
    )
    p.add_argument(
        "--mat_path",
        type=str,
        default="",
        help="Optional .mat output path. If empty, uses plot_path base name under experiments/results.",
    )

    p.add_argument("--draw_links", type=int, default=1, help="Overlay robot link shapes along the rollout (1/0)")
    p.add_argument("--num_link_frames", type=int, default=12, help="How many robot poses to draw along the rollout")

    # Obstacles: 'x,y,r;x,y,r'
    p.add_argument("--obstacles", type=str, default="0,1.8,0.20")

    # Controller params
    p.add_argument("--horizon", type=int, default=25)
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--lambda_coeff", type=float, default=0.6)
    p.add_argument("--noise_std", type=float, default=0.6)

    p.add_argument("--cdf_recovery", type=int, default=0, help="Enable CDF-based collision recovery (1/0)")
    p.add_argument("--cdf_method", type=str, default="offline_grid", help="CDF method: offline_grid (default) or online_computation")
    p.add_argument("--cdf_normal_gain", type=float, default=1.0)
    p.add_argument("--cdf_tangent_gain", type=float, default=0.5)

    default_sac_model_path = os.path.join(_ROOT_DIR, "experiments", "robot2d", "models", "sac_robot2d_model_online.pth")
    p.add_argument(
        "--sac_model_path",
        type=str,
        default=default_sac_model_path,
        help=(
            "SAC checkpoint used as RL-MPPI prior (Robot2D .pth). "
            "Default: experiments/robot2d/models/sac_robot2d_model_online.pth"
        ),
    )
    p.add_argument(
        "--use_model_obstacles",
        type=int,
        default=0,
        help=(
            "When using --sac_model_path, override --obstacles with the obstacle map stored in the checkpoint (1/0). "
            "Default is 0 so --obstacles takes effect."
        ),
    )

    _add_contour_avoidance_args(p)

    args = p.parse_args()

    print(
        "[CONFIG][MPPI] "
        + str(
            {
                "horizon": int(args.horizon),
                "num_samples": int(args.num_samples),
                "lambda_coeff": float(args.lambda_coeff),
                "noise_std": float(args.noise_std),
                "use_pd_nominal": True,
                "contour_avoidance": bool(
                    (not bool(getattr(args, "no_contour_avoidance", False)))
                    or bool(getattr(args, "contour_avoidance", False))
                ),
                "contour_mode": str(getattr(args, "contour_mode", "")),
                "contour_clearance_start": float(getattr(args, "contour_clearance_start", 0.0)),
                "contour_clearance_full": float(getattr(args, "contour_clearance_full", 0.0)),
                "contour_repulse_gain": float(getattr(args, "contour_repulse_gain", 0.0)),
                "contour_fd_eps": float(getattr(args, "contour_fd_eps", 0.0)),
                "cdf_recovery": bool(int(args.cdf_recovery)),
            }
        )
    )
    print(
        "[CONFIG][SAC] "
        + str(
            {
                "sac_model_path": str(args.sac_model_path),
                "use_model_obstacles": bool(int(args.use_model_obstacles)),
                "cdf_recovery": bool(int(args.cdf_recovery)),
                "cdf_method": str(args.cdf_method),
                "cdf_normal_gain": float(args.cdf_normal_gain),
                "cdf_tangent_gain": float(args.cdf_tangent_gain),
            }
        )
    )

    # If default SAC checkpoint doesn't exist, fall back to the heuristic prior.
    # (Avoid surprising hard-failure on fresh checkouts.)
    if str(args.sac_model_path).strip() and (not os.path.exists(str(args.sac_model_path))):
        print(f"[WARN] SAC model not found: {args.sac_model_path} -> falling back to IK heuristic prior")
        args.sac_model_path = ""

    np.random.seed(int(args.seed))

    contour_enabled = (not bool(getattr(args, "no_contour_avoidance", False)))
    if bool(getattr(args, "contour_avoidance", False)):
        contour_enabled = True

    class _ControllerWithRecovery:
        def __init__(self, base_controller, *, env: Robot2DEnvironmentObstacles, recovery_policy: PolicyWrapper):
            self.base = base_controller
            self.env = env
            self.recovery = recovery_policy

        def reset(self) -> None:
            if hasattr(self.base, "reset"):
                self.base.reset()

        def get_action(self, state: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
            q = np.asarray(state[: self.env.n], dtype=np.float32)
            clearance, collided = self.env.min_obstacle_clearance(q)
            if bool(collided) or float(clearance) < 0.0:
                return self.recovery.act(state, env=self.env)
            return self.base.get_action(state, target_pos)

    def parse_obs(s: str) -> list[CircleObstacle]:
        s = (s or "").strip()
        if not s:
            return []
        out: list[CircleObstacle] = []
        for part in s.split(";"):
            xs, ys, rs = [t.strip() for t in part.split(",")]
            out.append(CircleObstacle(float(xs), float(ys), float(rs)))
        return out

    obstacles = parse_obs(str(args.obstacles))
    obstacles_source = "args"

    if str(args.sac_model_path).strip() and bool(int(args.use_model_obstacles)) and os.path.exists(str(args.sac_model_path)):
        try:
            import torch
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError("PyTorch is required to load obstacles from SAC checkpoint") from e

        ckpt = torch.load(str(args.sac_model_path), map_location="cpu")
        ckpt_obs = ckpt.get("obstacles", None)
        if ckpt_obs is None:
            print("[WARN] SAC checkpoint has no 'obstacles' field; using --obstacles")
        else:
            obstacles = [CircleObstacle(float(x), float(y), float(r)) for (x, y, r) in ckpt_obs]
            obstacles_source = "sac_checkpoint"
            print(f"[COMPARE] Using obstacles from SAC checkpoint: n={len(obstacles)}")

    print(
        f"[COMPARE] Final obstacles source={obstacles_source}, "
        f"obstacles={[(float(o.x), float(o.y), float(o.r)) for o in obstacles]}"
    )

    env = Robot2DEnvironmentObstacles(
        link_lengths=[2.0, 2.0],
        target_pos=[float(args.target_x), float(args.target_y)],
        max_steps=int(args.max_steps),
        obstacles=obstacles,
        obstacle_margin=0.25,
        obstacle_penalty=250.0,
        terminate_on_collision=(not bool(int(args.cdf_recovery))),
        collision_check="arm",
        contour_avoidance=bool(contour_enabled),
        contour_mode=str(args.contour_mode),
        contour_cdf_method=str(args.contour_cdf_method),
        contour_clearance_start=float(args.contour_clearance_start),
        contour_clearance_full=float(args.contour_clearance_full),
        contour_fd_eps=float(args.contour_fd_eps),
        contour_repulse_gain=float(args.contour_repulse_gain),
    )

    def _wrap_pi(a: np.ndarray) -> np.ndarray:
        return (a + np.pi) % (2.0 * np.pi) - np.pi

    def _ik_2link(xy: np.ndarray, l1: float, l2: float, *, elbow: str) -> np.ndarray:
        x = float(xy[0])
        y = float(xy[1])
        r2 = x * x + y * y
        # cos(q2) from the law of cosines; clip to keep numeric stability.
        c2 = (r2 - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
        c2 = float(np.clip(c2, -1.0, 1.0))
        q2_mag = float(np.arccos(c2))
        q2 = q2_mag if elbow == "down" else -q2_mag
        s2 = float(np.sin(q2))
        k1 = l1 + l2 * c2
        k2 = l2 * s2
        q1 = float(np.arctan2(y, x) - np.arctan2(k2, k1))
        return _wrap_pi(np.asarray([q1, q2], dtype=np.float32))

    mppi = MPPI(
        env,
        horizon=int(args.horizon),
        num_samples=int(args.num_samples),
        lambda_coeff=float(args.lambda_coeff),
        noise_std=float(args.noise_std),
        use_pd_nominal=True,
    )

    prior: object
    if str(args.sac_model_path).strip():
        prior = load_sac_policy(str(args.sac_model_path), env=env)
        print(f"[RL-MPPI] Using SAC prior: {args.sac_model_path}")
    else:
        prior = IKHeuristicPolicy(kp=6.0, kd=2.0, repulse_gain=1.0)
        print("[RL-MPPI] Using IK heuristic prior")

    sac_policy = prior if isinstance(prior, PolicyWrapper) else PolicyWrapper(act_fn=prior)

    class _PolicyController:
        def __init__(self, policy: PolicyWrapper, *, env: Robot2DEnvironmentObstacles):
            self.policy = policy
            self.env = env

        def reset(self) -> None:
            if hasattr(self.policy, "reset"):
                self.policy.reset()

        def get_action(self, state: np.ndarray, target_pos: np.ndarray) -> np.ndarray:  # noqa: ARG002
            return np.asarray(self.policy.act(state, env=self.env), dtype=np.float32)

    rl_mppi = RLMppiController(
        env,
        policy=(prior if isinstance(prior, PolicyWrapper) else PolicyWrapper(act_fn=prior)),
        horizon=int(args.horizon),
        num_samples=int(args.num_samples),
        lambda_coeff=float(args.lambda_coeff),
        noise_std=float(args.noise_std),
    )

    if bool(int(args.cdf_recovery)):
        recovery = PolicyWrapper(
            act_fn=CDFCollisionRecoveryPolicy(
                method=str(args.cdf_method),
                proj_gain=float(args.cdf_normal_gain),
                tangent_step=float(args.cdf_tangent_gain),
            )
        )
        mppi = _ControllerWithRecovery(mppi, env=env, recovery_policy=recovery)
        sac = _ControllerWithRecovery(_PolicyController(sac_policy, env=env), env=env, recovery_policy=recovery)
        rl_mppi = _ControllerWithRecovery(rl_mppi, env=env, recovery_policy=recovery)
        print("[CDF] Collision recovery enabled (terminate_on_collision=0)")
    else:
        sac = _PolicyController(sac_policy, env=env)

    # A fixed initial pose (near-straight arm) for reproducibility.
    init_q = np.asarray([0.25, 0.10], dtype=np.float32)
    init_qd = np.zeros((env.n,), dtype=np.float32)
    init_state = np.concatenate([init_q, init_qd], axis=0)

    print(f"[COMPARE] init_q={init_q.tolist()}")
    if int(env.n) == 2 and env.link_lengths.shape[0] == 2:
        l1, l2 = float(env.link_lengths[0]), float(env.link_lengths[1])
        q_goal_down = _ik_2link(env.target_pos, l1, l2, elbow="down")
        q_goal_up = _ik_2link(env.target_pos, l1, l2, elbow="up")
        d_down = float(np.linalg.norm(_wrap_pi(q_goal_down - init_q)))
        d_up = float(np.linalg.norm(_wrap_pi(q_goal_up - init_q)))
        q_goal = q_goal_down if d_down <= d_up else q_goal_up
        print(f"[COMPARE] target_pos={np.asarray(env.target_pos, dtype=np.float32).tolist()}")
        print(f"[COMPARE] goal_q(elbow_down)={q_goal_down.tolist()}")
        print(f"[COMPARE] goal_q(elbow_up)={q_goal_up.tolist()}")
        print(f"[COMPARE] chosen_goal_q={q_goal.tolist()}")

    mppi.reset()
    sac.reset()
    rl_mppi.reset()

    ep_mppi = run_episode(env, mppi, initial_state=init_state)
    ep_sac = run_episode(env, sac, initial_state=init_state)
    ep_rl = run_episode(env, rl_mppi, initial_state=init_state)

    print("[MPPI]", {k: ep_mppi[k] for k in ("success", "hit_obstacle", "steps", "final_dist", "min_clearance", "episode_time_s")})
    print("[SAC]", {k: ep_sac[k] for k in ("success", "hit_obstacle", "steps", "final_dist", "min_clearance", "episode_time_s")})
    print("[RL-MPPI]", {k: ep_rl[k] for k in ("success", "hit_obstacle", "steps", "final_dist", "min_clearance", "episode_time_s")})

    plot_results(
        results={"MPPI": ep_mppi, "SAC": ep_sac, "RL-MPPI": ep_rl},
        env=env,
        plot_path=str(args.plot_path),
        show=bool(args.show_plot),
        draw_links=bool(int(args.draw_links)),
        num_link_frames=int(args.num_link_frames),
    )

    plot_timeseries(
        results={"MPPI": ep_mppi, "SAC": ep_sac, "RL-MPPI": ep_rl},
        env=env,
        plot_path=str(args.plot_path),
        show=bool(args.show_plot),
    )

    if bool(int(getattr(args, "export_mat", 0))):
        export_mat(
            results={"MPPI": ep_mppi, "SAC": ep_sac, "RL-MPPI": ep_rl},
            env=env,
            plot_path=str(args.plot_path),
            mat_path=str(getattr(args, "mat_path", "")),
        )


if __name__ == "__main__":
    main()
