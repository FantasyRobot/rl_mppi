#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

# Ensure imports work no matter where this is launched from.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, _ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from env.envball_utils import BallEnvironment
from mppi_ball import MPPI
from rl_mppi_ball import RLMppiController, load_sac_policy

def run_episode(
    env: BallEnvironment,
    controller,
    target_pos: np.ndarray,
    *,
    initial_state: np.ndarray,
    collect_trajectory: bool,
) -> dict:
    state = env.reset(initial_state=np.asarray(initial_state, dtype=np.float32))
    total_reward = 0.0
    steps_taken = 0
    hit_boundary_any = False

    traj_xy: list[np.ndarray] | None = None
    if collect_trajectory:
        traj_xy = [np.asarray(state[:2], dtype=np.float32).copy()]

    t0 = time.perf_counter()
    while True:
        action = controller.get_action(state, target_pos)
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        next_state, reward, done, info = env.step(action)
        total_reward += float(reward)
        steps_taken += 1
        if bool(info.get("hit_boundary", False)):
            hit_boundary_any = True

        if traj_xy is not None:
            traj_xy.append(np.asarray(next_state[:2], dtype=np.float32).copy())

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

    return {
        "success": success,
        "steps_taken": steps_taken,
        "total_reward": total_reward,
        "final_distance": final_distance,
        "hit_boundary": hit_boundary_any,
        "episode_time_s": dt,
        "traj_xy": (np.asarray(traj_xy, dtype=np.float32) if traj_xy is not None else None),
    }


def run_benchmark(
    *,
    name: str,
    make_env,
    make_controller,
    target_pos: np.ndarray,
    num_tests: int,
    initial_states: list[np.ndarray],
    collect_trajectories: bool,
) -> dict:
    results = []
    t0 = time.perf_counter()

    if len(initial_states) != num_tests:
        raise ValueError(f"initial_states length {len(initial_states)} != num_tests {num_tests}")

    for k in range(num_tests):
        env = make_env()
        controller = make_controller(env)
        if hasattr(controller, "reset"):
            controller.reset()
        results.append(
            run_episode(
                env,
                controller,
                target_pos,
                initial_state=initial_states[k],
                collect_trajectory=collect_trajectories,
            )
        )

    total_time = time.perf_counter() - t0

    success_rate = float(np.mean([1.0 if r["success"] else 0.0 for r in results]))
    avg_steps = float(np.mean([r["steps_taken"] for r in results]))
    avg_episode_time = float(np.mean([r["episode_time_s"] for r in results]))

    return {
        "name": name,
        "num_tests": num_tests,
        "success_rate": success_rate,
        "avg_steps": avg_steps,
        "avg_episode_time_s": avg_episode_time,
        "total_time_s": float(total_time),
        "results": results,
    }


def plot_comparison(
    *,
    mppi_stats: dict,
    rl_mppi_stats: dict,
    target_pos: np.ndarray,
    plot_path: str,
    show_plot: bool,
) -> None:
    mppi_trajs = [r.get("traj_xy", None) for r in mppi_stats.get("results", [])]
    rl_trajs = [r.get("traj_xy", None) for r in rl_mppi_stats.get("results", [])]
    if not mppi_trajs or not rl_trajs or any(t is None for t in mppi_trajs) or any(t is None for t in rl_trajs):
        raise RuntimeError("Trajectories were not collected; re-run with plotting enabled.")

    plt.figure(figsize=(10, 10))

    for i, (tm, tr) in enumerate(zip(mppi_trajs, rl_trajs)):
        tm = np.asarray(tm, dtype=np.float32)
        tr = np.asarray(tr, dtype=np.float32)
        plt.plot(tm[:, 0], tm[:, 1], color="tab:blue", alpha=0.6, linewidth=1.5, label="MPPI" if i == 0 else "")
        plt.plot(tr[:, 0], tr[:, 1], color="tab:orange", alpha=0.6, linewidth=1.5, linestyle="--", label="RL-MPPI" if i == 0 else "")
        plt.plot(tm[0, 0], tm[0, 1], "s", markersize=6, color="green", label="Start" if i == 0 else "")

    plt.plot(float(target_pos[0]), float(target_pos[1]), "rx", markersize=12, label="Target")
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.grid(True)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("MPPI vs RL-MPPI trajectories")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(loc="upper right")

    out_dir = os.path.dirname(plot_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(plot_path)
    print(f"Trajectory comparison plot saved to {plot_path}")
    if show_plot:
        plt.show()
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare pure MPPI vs RL-guided MPPI on BallEnvironment")
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.path.join("sac", "sac_ball", "models", "sac_ball_model_online.pth"),
        help="SAC checkpoint path for RL-MPPI",
    )
    parser.add_argument("--target_x", type=float, default=3.0)
    parser.add_argument("--target_y", type=float, default=3.0)
    parser.add_argument("--num_tests", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=2000)

    # Shared MPPI hyperparameters
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--lambda_coeff", type=float, default=0.2)
    parser.add_argument("--noise_std", type=float, default=0.6)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_vectorized", action="store_true", help="Disable vectorized rollouts (debug/legacy)")
    parser.add_argument("--plot_path", type=str, default="outputs/compare_mppi_vs_rl_mppi.png")
    parser.add_argument("--show_plot", action="store_true")
    parser.add_argument("--no_plot", action="store_true", help="Disable trajectory collection/plotting")

    args = parser.parse_args()

    target_pos = np.asarray([float(args.target_x), float(args.target_y)], dtype=np.float32)

    def make_env() -> BallEnvironment:
        return BallEnvironment(target_pos=target_pos.tolist(), max_steps=int(args.max_steps), reward_scale=100.0)

    # Preload policy once so we don't count model load time.
    env0 = make_env()
    policy = load_sac_policy(str(args.model_path), state_dim=env0.state_dim, action_dim=env0.action_dim)

    # Pre-sample the same initial states for a fair comparison.
    # (Otherwise whichever controller runs first consumes the RNG stream.)
    np.random.seed(int(args.seed))
    tmp_env = make_env()
    initial_states: list[np.ndarray] = []
    for _ in range(int(args.num_tests)):
        s = tmp_env.reset()
        initial_states.append(np.asarray(s, dtype=np.float32).copy())

    def make_rl_mppi_controller(env: BallEnvironment) -> RLMppiController:
        return RLMppiController(
            env=env,
            policy=policy,
            horizon=int(args.horizon),
            num_samples=int(args.num_samples),
            lambda_coeff=float(args.lambda_coeff),
            noise_std=float(args.noise_std),
            dt=env.dt,
            vectorized_rollouts=not bool(args.no_vectorized),
        )

    def make_mppi_controller(env: BallEnvironment) -> MPPI:
        return MPPI(
            env=env,
            horizon=int(args.horizon),
            num_samples=int(args.num_samples),
            lambda_coeff=float(args.lambda_coeff),
            noise_std=float(args.noise_std),
            dt=env.dt,
            vectorized_rollouts=not bool(args.no_vectorized),
        )

    print("Benchmark settings:")
    print(f"  target_pos={target_pos.tolist()}, num_tests={args.num_tests}, max_steps={args.max_steps}")
    print(f"  horizon={args.horizon}, num_samples={args.num_samples}, lambda={args.lambda_coeff}, noise_std={args.noise_std}")
    print(f"  vectorized_rollouts={not args.no_vectorized}")

    collect_trajectories = not bool(args.no_plot)

    np.random.seed(int(args.seed))
    mppi_stats = run_benchmark(
        name="MPPI",
        make_env=make_env,
        make_controller=make_mppi_controller,
        target_pos=target_pos,
        num_tests=int(args.num_tests),
        initial_states=initial_states,
        collect_trajectories=collect_trajectories,
    )

    np.random.seed(int(args.seed))
    rl_mppi_stats = run_benchmark(
        name="RL-MPPI",
        make_env=make_env,
        make_controller=make_rl_mppi_controller,
        target_pos=target_pos,
        num_tests=int(args.num_tests),
        initial_states=initial_states,
        collect_trajectories=collect_trajectories,
    )

    def _print(stats: dict) -> None:
        print(f"\n[{stats['name']}]")
        print(f"  success_rate: {stats['success_rate']:.2%}")
        print(f"  avg_steps: {stats['avg_steps']:.1f}")
        print(f"  avg_episode_time_s: {stats['avg_episode_time_s']:.3f}")
        print(f"  total_time_s: {stats['total_time_s']:.3f}")

    _print(mppi_stats)
    _print(rl_mppi_stats)

    speedup = mppi_stats["total_time_s"] / max(1e-9, rl_mppi_stats["total_time_s"])
    print(f"\nSpeed (MPPI total / RL-MPPI total): {speedup:.2f}x")

    if collect_trajectories:
        plot_comparison(
            mppi_stats=mppi_stats,
            rl_mppi_stats=rl_mppi_stats,
            target_pos=target_pos,
            plot_path=str(args.plot_path),
            show_plot=bool(args.show_plot),
        )


if __name__ == "__main__":
    main()
