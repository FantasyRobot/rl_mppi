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
from algorithms.rl_mppi.rl_mppi_robot2d import IKHeuristicPolicy, PolicyWrapper, RLMppiController, load_sac_policy


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
        clearances.append(float(info.get("min_obstacle_clearance", float("inf"))))
        if bool(info.get("hit_obstacle", False)):
            hit = True

        if done:
            break

    dt = time.perf_counter() - t0
    final_eef = env.forward_kinematics_eef(s[: env.n])
    final_dist = float(np.linalg.norm(final_eef - env.target_pos))
    success = final_dist < float(env.reach_threshold)

    return {
        "success": bool(success),
        "hit_obstacle": bool(hit),
        "steps": int(steps),
        "total_reward": float(total_reward),
        "final_dist": float(final_dist),
        "traj": np.asarray(traj, dtype=np.float32),
        "q_traj": np.asarray(q_traj, dtype=np.float32),
        "min_clearance": float(np.min(clearances)) if clearances else float("inf"),
        "episode_time_s": float(dt),
    }


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

    colors = {"MPPI": "tab:blue", "RL-MPPI": "tab:orange"}
    styles = {"MPPI": "-", "RL-MPPI": "--"}

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
            idx = np.linspace(0, q_traj.shape[0] - 1, num=k, dtype=int)
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

    p.add_argument("--draw_links", type=int, default=1, help="Overlay robot link shapes along the rollout (1/0)")
    p.add_argument("--num_link_frames", type=int, default=12, help="How many robot poses to draw along the rollout")

    # Obstacles: 'x,y,r;x,y,r'
    p.add_argument("--obstacles", type=str, default="0,1.8,0.20")

    # Controller params
    p.add_argument("--horizon", type=int, default=25)
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--lambda_coeff", type=float, default=0.6)
    p.add_argument("--noise_std", type=float, default=0.6)

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
        default=1,
        help="When using --sac_model_path, override --obstacles with the obstacle map stored in the checkpoint (1/0)",
    )

    args = p.parse_args()

    # If default SAC checkpoint doesn't exist, fall back to the heuristic prior.
    # (Avoid surprising hard-failure on fresh checkouts.)
    if str(args.sac_model_path).strip() and (not os.path.exists(str(args.sac_model_path))):
        print(f"[WARN] SAC model not found: {args.sac_model_path} -> falling back to IK heuristic prior")
        args.sac_model_path = ""

    np.random.seed(int(args.seed))

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
            print(f"[COMPARE] Using obstacles from SAC checkpoint: n={len(obstacles)}")

    env = Robot2DEnvironmentObstacles(
        link_lengths=[2.0, 2.0],
        target_pos=[float(args.target_x), float(args.target_y)],
        max_steps=int(args.max_steps),
        obstacles=obstacles,
        obstacle_margin=0.25,
        obstacle_penalty=250.0,
        terminate_on_collision=True,
        collision_check="arm",
    )

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

    rl_mppi = RLMppiController(
        env,
        policy=(prior if isinstance(prior, PolicyWrapper) else PolicyWrapper(act_fn=prior)),
        horizon=int(args.horizon),
        num_samples=int(args.num_samples),
        lambda_coeff=float(args.lambda_coeff),
        noise_std=float(args.noise_std),
    )

    # A fixed initial pose (near-straight arm) for reproducibility.
    init_q = np.asarray([0.25, 0.10], dtype=np.float32)
    init_qd = np.zeros((env.n,), dtype=np.float32)
    init_state = np.concatenate([init_q, init_qd], axis=0)

    mppi.reset()
    rl_mppi.reset()

    ep_mppi = run_episode(env, mppi, initial_state=init_state)
    ep_rl = run_episode(env, rl_mppi, initial_state=init_state)

    print("[MPPI]", {k: ep_mppi[k] for k in ("success", "hit_obstacle", "steps", "final_dist", "min_clearance", "episode_time_s")})
    print("[RL-MPPI]", {k: ep_rl[k] for k in ("success", "hit_obstacle", "steps", "final_dist", "min_clearance", "episode_time_s")})

    plot_results(
        results={"MPPI": ep_mppi, "RL-MPPI": ep_rl},
        env=env,
        plot_path=str(args.plot_path),
        show=bool(args.show_plot),
        draw_links=bool(int(args.draw_links)),
        num_link_frames=int(args.num_link_frames),
    )


if __name__ == "__main__":
    main()
