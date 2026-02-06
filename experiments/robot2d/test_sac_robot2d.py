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
from algorithms.rl_mppi.rl_mppi_robot2d import load_sac_policy


def _parse_obstacles(s: str) -> list[CircleObstacle]:
    s = (s or "").strip()
    if not s:
        return []
    out: list[CircleObstacle] = []
    for part in s.split(";"):
        xs, ys, rs = [t.strip() for t in part.split(",")]
        out.append(CircleObstacle(float(xs), float(ys), float(rs)))
    return out


def _resolve_plot_path(plot_path: str) -> str:
    plot_path = os.path.expanduser(os.path.expandvars(str(plot_path)))
    if not os.path.isabs(plot_path) and os.path.dirname(plot_path) == "":
        plot_path = os.path.join(_RESULTS_DIR, plot_path)
    out_dir = os.path.dirname(plot_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    return plot_path


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
    color: str = "tab:blue",
    alpha: float = 0.6,
    lw: float = 2.5,
    marker_size: float = 2.0,
) -> None:
    pts = env.forward_kinematics_all_joints(q)
    ax.plot(pts[:, 0], pts[:, 1], "-", color=color, alpha=float(alpha), linewidth=float(lw))
    ax.plot(pts[:, 0], pts[:, 1], "o", color=color, alpha=float(alpha), markersize=float(marker_size))


def test_sac_robot2d(
    *,
    model_path: str,
    link_lengths: list[float],
    target_pos: list[float],
    obstacles: list[CircleObstacle],
    num_tests: int = 10,
    max_steps: int = 450,
    plot_path: str | None = None,
    show_plot: bool = False,
    draw_links: bool = True,
    num_link_frames: int = 12,
) -> dict:
    env = Robot2DEnvironmentObstacles(
        link_lengths=link_lengths,
        target_pos=target_pos,
        max_steps=int(max_steps),
        obstacles=obstacles,
        obstacle_margin=0.25,
        obstacle_penalty=250.0,
        terminate_on_collision=True,
        collision_check="arm",
    )

    policy = load_sac_policy(str(model_path), env=env)

    successes = 0
    collisions = 0
    final_dists: list[float] = []
    times: list[float] = []

    first_traj: list[np.ndarray] = []
    first_q_traj: list[np.ndarray] = []

    for i in range(int(num_tests)):
        s = env.reset()
        traj = []
        q_traj = []
        t0 = time.perf_counter()
        hit = False
        for _ in range(int(max_steps)):
            traj.append(env.forward_kinematics_eef(s[: env.n]).copy())
            q_traj.append(np.asarray(s[: env.n], dtype=np.float32).copy())
            a = policy.act(s, env=env)
            s, _, done, info = env.step(a)
            hit = hit or bool(info.get("hit_obstacle", False))
            if done:
                break
        times.append(time.perf_counter() - t0)

        eef = env.forward_kinematics_eef(s[: env.n])
        dist = float(np.linalg.norm(eef - env.target_pos))
        final_dists.append(dist)
        if dist < float(env.reach_threshold):
            successes += 1
        if hit:
            collisions += 1

        if i == 0:
            first_traj = traj
            first_q_traj = q_traj

    out = {
        "success_rate": float(successes / max(1, int(num_tests))),
        "collision_rate": float(collisions / max(1, int(num_tests))),
        "avg_final_dist": float(np.mean(final_dists) if final_dists else float("nan")),
        "avg_time_s": float(np.mean(times) if times else float("nan")),
    }

    if plot_path is not None and plt is not None and first_traj:
        plot_path = _resolve_plot_path(plot_path)
        fig, ax = plt.subplots(figsize=(6, 6))
        traj = np.asarray(first_traj, dtype=np.float32)
        ax.plot(traj[:, 0], traj[:, 1], "b-", lw=2.2, label="EEF")
        ax.scatter([traj[0, 0]], [traj[0, 1]], c="g", s=60, label="start")
        ax.scatter([env.target_pos[0]], [env.target_pos[1]], c="r", s=60, label="goal")

        _draw_obstacles(ax, env)

        if bool(draw_links) and first_q_traj:
            q_arr = np.asarray(first_q_traj, dtype=np.float32)
            k = int(max(2, int(num_link_frames)))
            idx = np.linspace(0, q_arr.shape[0] - 1, num=k, dtype=int)
            for j, t in enumerate(idx):
                alpha = 0.10 + 0.85 * (j / max(1, (len(idx) - 1)))
                _draw_robot_links(ax, env, q_arr[int(t)], color="tab:blue", alpha=float(alpha), lw=2.0, marker_size=2.0)

        reach = float(np.sum(env.link_lengths))
        ax.set_xlim(-reach - 0.5, reach + 0.5)
        ax.set_ylim(-reach - 0.5, reach + 0.5)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title("Robot2D SAC test (one episode)")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        print(f"Saved plot to {plot_path}")
        if bool(show_plot):
            plt.show()
        plt.close(fig)

    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Test Robot2D SAC and optionally plot a trajectory")

    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--link_lengths", type=str, default="2.0,2.0")
    p.add_argument("--target_x", type=float, default=-2.8)
    p.add_argument("--target_y", type=float, default=1.8)
    p.add_argument("--obstacles", type=str, default="0,1.8,0.20")

    p.add_argument("--num_tests", type=int, default=10)
    p.add_argument("--max_steps", type=int, default=450)
    p.add_argument("--plot_path", type=str, default=os.path.join(_RESULTS_DIR, "robot2d_sac_test.png"))
    p.add_argument("--show_plot", action="store_true")
    p.add_argument("--draw_links", type=int, default=1, help="Overlay robot link shapes along the rollout (1/0)")
    p.add_argument("--num_link_frames", type=int, default=12, help="How many robot poses to draw along the rollout")

    args = p.parse_args()

    link_lengths = [float(x.strip()) for x in str(args.link_lengths).split(",") if x.strip()]
    out = test_sac_robot2d(
        model_path=str(args.model_path),
        link_lengths=link_lengths,
        target_pos=[float(args.target_x), float(args.target_y)],
        obstacles=_parse_obstacles(str(args.obstacles)),
        num_tests=int(args.num_tests),
        max_steps=int(args.max_steps),
        plot_path=str(args.plot_path) if str(args.plot_path).strip() else None,
        show_plot=bool(args.show_plot),
        draw_links=bool(int(args.draw_links)),
        num_link_frames=int(args.num_link_frames),
    )

    print(out)


if __name__ == "__main__":
    main()
