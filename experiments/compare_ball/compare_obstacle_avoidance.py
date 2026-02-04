#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

# Ensure repo root is importable regardless of CWD.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

_RESULTS_DIR = os.path.join(_ROOT_DIR, "experiments", "results")


def _resolve_plot_path(plot_path: str) -> str:
    plot_path = os.path.expanduser(os.path.expandvars(str(plot_path)))
    if not os.path.isabs(plot_path) and os.path.dirname(plot_path) == "":
        plot_path = os.path.join(_RESULTS_DIR, plot_path)
    out_dir = os.path.dirname(plot_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    return plot_path

from env.envball_obstacles import BallEnvironmentObstacles, CircleObstacle
from algorithms.mppi.mppi_ball import MPPI
from algorithms.rl_mppi.rl_mppi_ball import RLMppiController, load_sac_policy


class SACController:
    def __init__(self, policy, env):
        self.policy = policy
        self.env = env

    def reset(self) -> None:
        return None

    def get_action(self, state: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
        return self.policy.act(state, env=self.env)


def parse_obstacles(s: str) -> list[CircleObstacle]:
    """Parse obstacles from 'x,y,r;x,y,r'"""
    s = (s or "").strip()
    if not s:
        return []
    obstacles: list[CircleObstacle] = []
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        xs, ys, rs = [p.strip() for p in part.split(",")]
        obstacles.append(CircleObstacle(float(xs), float(ys), float(rs)))
    return obstacles


def run_episode(env: BallEnvironmentObstacles, controller, target_pos: np.ndarray, *, initial_state: np.ndarray) -> dict:
    state = env.reset(initial_state=np.asarray(initial_state, dtype=np.float32))

    traj_xy = [np.asarray(state[:2], dtype=np.float32).copy()]
    hit_obstacle_any = False
    min_clearance = float("inf")

    t0 = time.perf_counter()
    total_reward = 0.0
    steps_taken = 0

    while True:
        # If we already start within the reach threshold, avoid extra controller compute.
        if float(np.linalg.norm(state[:2] - env.target_pos)) < float(env.reach_threshold):
            break
        action = controller.get_action(state, target_pos)
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        next_state, reward, done, info = env.step(action)
        total_reward += float(reward)
        steps_taken += 1
        traj_xy.append(np.asarray(next_state[:2], dtype=np.float32).copy())

        clr = float(info.get("min_obstacle_clearance", float("inf")))
        min_clearance = min(min_clearance, clr)
        if bool(info.get("hit_obstacle", False)):
            hit_obstacle_any = True

        if float(info.get("distance", 1e9)) < float(env.reach_threshold):
            state = next_state
            break
        if done:
            state = next_state
            break

        state = next_state

    dt = time.perf_counter() - t0
    final_distance = float(np.linalg.norm(state[:2] - env.target_pos))
    success = final_distance < float(env.reach_threshold)

    traj_xy = np.asarray(traj_xy, dtype=np.float32)
    path_len = float(np.sum(np.sqrt(np.sum(np.diff(traj_xy, axis=0) ** 2, axis=1))))

    return {
        "success": bool(success),
        "steps_taken": int(steps_taken),
        "total_reward": float(total_reward),
        "final_distance": float(final_distance),
        "hit_obstacle": bool(hit_obstacle_any),
        "min_clearance": float(min_clearance),
        "path_length": float(path_len),
        "episode_time_s": float(dt),
        "traj_xy": traj_xy,
    }


def plot_runs(*, runs: dict[str, list[dict]], obstacles: list[CircleObstacle], target_pos: np.ndarray, plot_path: str, show_plot: bool) -> None:
    plot_path = _resolve_plot_path(plot_path)
    plt.figure(figsize=(10, 10))

    ax = plt.gca()
    influence = float(getattr(plot_runs, "obstacle_margin", 0.2))

    for obs in obstacles:
        circ = plt.Circle((obs.x, obs.y), obs.r, color="gray", alpha=0.35)
        ax.add_patch(circ)
        circ2 = plt.Circle((obs.x, obs.y), obs.r + influence, color="gray", alpha=0.15, fill=False, linestyle="--")
        ax.add_patch(circ2)

    colors = {"MPPI": "tab:blue", "SAC": "tab:green", "RL-MPPI": "tab:orange"}
    styles = {"MPPI": "-", "SAC": ":", "RL-MPPI": "--"}

    def _pick_representative_episode(eps: list[dict]) -> dict | None:
        if not eps:
            return None

        # Prefer successful episodes, then pick the one with smallest final distance.
        successes = [e for e in eps if bool(e.get("success", False))]
        pool = successes if successes else eps

        def key(ep: dict) -> tuple[float, float]:
            fd = float(ep.get("final_distance", float("inf")))
            t = float(ep.get("episode_time_s", 0.0))
            return (fd, t)

        return min(pool, key=key)

    plot_all = bool(getattr(plot_runs, "plot_all_trajectories", False))

    for name, eps in runs.items():
        if plot_all:
            for i, ep in enumerate(eps):
                traj = np.asarray(ep["traj_xy"], dtype=np.float32)
                plt.plot(
                    traj[:, 0],
                    traj[:, 1],
                    color=colors.get(name, None),
                    linestyle=styles.get(name, "-"),
                    alpha=0.45,
                    linewidth=1.4,
                    label=name if i == 0 else "",
                )
                if i == 0:
                    plt.plot(traj[0, 0], traj[0, 1], "s", markersize=7, color="black", label="Start")
            continue

        ep = _pick_representative_episode(eps)
        if ep is None:
            continue
        traj = np.asarray(ep["traj_xy"], dtype=np.float32)
        plt.plot(
            traj[:, 0],
            traj[:, 1],
            color=colors.get(name, None),
            linestyle=styles.get(name, "-"),
            alpha=0.8,
            linewidth=2.2,
            label=name,
        )
        plt.plot(traj[0, 0], traj[0, 1], "s", markersize=7, color="black", label="Start" if name == list(runs.keys())[0] else "")

    plt.plot(float(target_pos[0]), float(target_pos[1]), "rx", markersize=12, label="Target")
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.grid(True)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("Obstacle avoidance: MPPI vs SAC vs RL-MPPI")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(loc="upper right")

    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    if show_plot:
        plt.show()
    plt.close()


def summarize(name: str, results: list[dict]) -> None:
    succ = float(np.mean([1.0 if r["success"] else 0.0 for r in results]))
    hit = float(np.mean([1.0 if r["hit_obstacle"] else 0.0 for r in results]))
    avg_clear = float(np.mean([r["min_clearance"] for r in results]))
    avg_len = float(np.mean([r["path_length"] for r in results]))
    avg_steps = float(np.mean([r["steps_taken"] for r in results]))
    avg_time = float(np.mean([r["episode_time_s"] for r in results]))

    print(f"\n[{name}]")
    print(f"  success_rate: {succ:.2%}")
    print(f"  obstacle_hit_rate: {hit:.2%}")
    print(f"  avg_min_clearance: {avg_clear:.3f}")
    print(f"  avg_path_length: {avg_len:.3f}")
    print(f"  avg_steps: {avg_steps:.1f}")
    print(f"  avg_episode_time_s: {avg_time:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Obstacle avoidance comparison: MPPI vs SAC vs RL-MPPI")
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.path.join(_ROOT_DIR, "experiments", "sac_ball", "models", "sac_ball_model_online.pth"),
    )
    parser.add_argument("--init_x", type=float, default=-6.0)
    parser.add_argument("--init_y", type=float, default=-6.0)
    parser.add_argument("--init_vx", type=float, default=0.0)
    parser.add_argument("--init_vy", type=float, default=0.0)
    parser.add_argument("--target_x", type=float, default=3.0)
    parser.add_argument("--target_y", type=float, default=3.0)
    parser.add_argument("--num_tests", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=2000)

    parser.add_argument(
        "--obstacles",
        type=str,
        default="-2.5,-2.5,1.2;2.5,0,1.2;-2.5,2.5,1.2",
        help="Circular obstacles as 'x,y,r;x,y,r'",
    )
    parser.add_argument("--obstacle_margin", type=float, default=0.6)
    parser.add_argument("--obstacle_penalty", type=float, default=200.0, help="Extra reward penalty strength inside margin")
    parser.add_argument("--terminate_on_collision", action="store_true")

    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--lambda_coeff", type=float, default=0.2)
    parser.add_argument("--noise_std", type=float, default=0.6)
    parser.add_argument("--obstacle_cost_coeff", type=float, default=40000.0, help="Planner cost penalty weight")
    parser.add_argument("--obstacle_safety_distance", type=float, default=0.1, help="Inflate obstacle radius in planner cost")
    parser.add_argument("--collision_cost", type=float, default=1e7, help="Planner cost added on predicted collision")
    parser.add_argument("--disable_los_cost", action="store_true", help="Disable line-of-sight obstacle cost")
    parser.add_argument("--los_influence", type=float, default=0.8, help="LOS influence distance")
    parser.add_argument("--los_cost_coeff", type=float, default=20000.0, help="LOS penalty weight")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--plot_path",
        type=str,
        default=os.path.join(_ROOT_DIR, "experiments", "results", "obstacle_compare.png"),
    )
    parser.add_argument("--show_plot", action="store_true")
    parser.add_argument(
        "--plot_all_trajectories",
        action="store_true",
        help="Plot every rollout trajectory for each algorithm (default: plot one representative trajectory per algorithm)",
    )

    args = parser.parse_args()

    np.random.seed(int(args.seed))

    obstacles = parse_obstacles(str(args.obstacles))
    obstacles_tuples = [(o.x, o.y, o.r) for o in obstacles]

    target_pos = np.asarray([float(args.target_x), float(args.target_y)], dtype=np.float32)
    init_state = np.asarray([float(args.init_x), float(args.init_y), float(args.init_vx), float(args.init_vy)], dtype=np.float32)

    def make_env() -> BallEnvironmentObstacles:
        return BallEnvironmentObstacles(
            target_pos=target_pos.tolist(),
            max_steps=int(args.max_steps),
            reward_scale=100.0,
            obstacles=obstacles,
            obstacle_margin=float(args.obstacle_margin),
            obstacle_penalty=float(args.obstacle_penalty),
            terminate_on_collision=bool(args.terminate_on_collision),
        )

    env0 = make_env()
    policy = load_sac_policy(str(args.model_path), state_dim=env0.state_dim, action_dim=env0.action_dim)

    def make_mppi(env):
        return MPPI(
            env=env,
            horizon=int(args.horizon),
            num_samples=int(args.num_samples),
            lambda_coeff=float(args.lambda_coeff),
            noise_std=float(args.noise_std),
            dt=env.dt,
            vectorized_rollouts=True,
            obstacles=obstacles_tuples,
            obstacle_margin=float(args.obstacle_margin),
            obstacle_cost_coeff=float(args.obstacle_cost_coeff),
            obstacle_safety_distance=float(args.obstacle_safety_distance),
            collision_cost=float(args.collision_cost),
            use_los_obstacle_cost=not bool(args.disable_los_cost),
            los_influence=float(args.los_influence),
            los_cost_coeff=float(args.los_cost_coeff),
        )

    def make_sac(env):
        return SACController(policy, env)

    def make_rl_mppi(env):
        return RLMppiController(
            env=env,
            policy=policy,
            horizon=int(args.horizon),
            num_samples=int(args.num_samples),
            lambda_coeff=float(args.lambda_coeff),
            noise_std=float(args.noise_std),
            dt=env.dt,
            vectorized_rollouts=True,
            obstacles=obstacles_tuples,
            obstacle_margin=float(args.obstacle_margin),
            obstacle_cost_coeff=float(args.obstacle_cost_coeff),
            obstacle_safety_distance=float(args.obstacle_safety_distance),
            collision_cost=float(args.collision_cost),
            use_los_obstacle_cost=not bool(args.disable_los_cost),
            los_influence=float(args.los_influence),
            los_cost_coeff=float(args.los_cost_coeff),
        )

    initial_states = [init_state.copy() for _ in range(int(args.num_tests))]

    def run_many(make_controller):
        results = []
        t0 = time.perf_counter()
        for k in range(int(args.num_tests)):
            env = make_env()
            ctrl = make_controller(env)
            if hasattr(ctrl, "reset"):
                ctrl.reset()
            results.append(run_episode(env, ctrl, target_pos, initial_state=initial_states[k]))
        total = time.perf_counter() - t0
        return results, float(total)

    print("Scenario:")
    print(f"  init_state={init_state.tolist()} target={target_pos.tolist()} num_tests={args.num_tests} max_steps={args.max_steps}")
    print(f"  obstacles={[ (o.x,o.y,o.r) for o in obstacles ]}")
    print("Planner params:")
    print(f"  horizon={args.horizon} num_samples={args.num_samples} lambda={args.lambda_coeff} noise_std={args.noise_std}")
    print(f"  obstacle_margin={args.obstacle_margin} obstacle_cost_coeff={args.obstacle_cost_coeff}")

    mppi_res, mppi_t = run_many(make_mppi)
    sac_res, sac_t = run_many(make_sac)
    rl_res, rl_t = run_many(make_rl_mppi)

    summarize("MPPI", mppi_res)
    print(f"  total_time_s: {mppi_t:.3f}")
    summarize("SAC", sac_res)
    print(f"  total_time_s: {sac_t:.3f}")
    summarize("RL-MPPI", rl_res)
    print(f"  total_time_s: {rl_t:.3f}")

    runs = {"MPPI": mppi_res, "SAC": sac_res, "RL-MPPI": rl_res}
    plot_runs.obstacle_margin = float(args.obstacle_margin)
    plot_runs.plot_all_trajectories = bool(args.plot_all_trajectories)
    plot_runs(runs=runs, obstacles=obstacles, target_pos=target_pos, plot_path=str(args.plot_path), show_plot=bool(args.show_plot))


if __name__ == "__main__":
    main()
