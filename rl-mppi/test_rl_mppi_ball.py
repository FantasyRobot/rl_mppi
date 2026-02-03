#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Ensure imports work no matter where this is launched from.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, _ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from env.envball_utils import BallEnvironment
from rl_mppi_ball import RLMppiController, load_sac_policy


def plot_trajectories(trajectories: list[np.ndarray], target_pos: np.ndarray, *, plot_path: str, show_plot: bool) -> None:
    plt.figure(figsize=(10, 10))
    for i, traj in enumerate(trajectories):
        plt.plot(traj[:, 0], traj[:, 1], marker="o", markersize=3, label=f"Test {i+1}")
        plt.plot(traj[0, 0], traj[0, 1], "s", markersize=8, color="green", label=("Start" if i == 0 else ""))

    plt.plot(target_pos[0], target_pos[1], "rx", markersize=15, label="Target")
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.grid(True)
    plt.title("Ball Trajectories - RL-MPPI")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend(loc="upper right")

    out_dir = os.path.dirname(plot_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(plot_path)
    print(f"Trajectory plot saved to {plot_path}")
    if show_plot:
        plt.show()
    plt.close()


def test_rl_mppi_ball(
    *,
    model_path: str,
    target_pos: list[float],
    num_tests: int,
    max_steps: int,
    plot_path: str,
    show_plot: bool,
    init_near_target_radius: float | None,
    horizon: int,
    num_samples: int,
    lambda_coeff: float,
    noise_std: float,
) -> tuple[list[dict], list[np.ndarray]]:
    env = BallEnvironment(target_pos=target_pos, max_steps=max_steps, reward_scale=100.0)

    print(f"Loading SAC model from {model_path}...")
    policy = load_sac_policy(model_path, state_dim=env.state_dim, action_dim=env.action_dim)

    controller = RLMppiController(
        env=env,
        policy=policy,
        horizon=horizon,
        num_samples=num_samples,
        lambda_coeff=lambda_coeff,
        noise_std=noise_std,
        dt=env.dt,
    )

    results: list[dict] = []
    trajectories: list[np.ndarray] = []

    boundary_hit_episodes = 0

    for test_idx in range(num_tests):
        controller.reset()
        if init_near_target_radius is not None:
            r = float(init_near_target_radius)
            init_state = np.zeros(4, dtype=float)
            init_state[0] = env.target_pos[0] + np.random.uniform(-r, r)
            init_state[1] = env.target_pos[1] + np.random.uniform(-r, r)
            init_state[2] = 0.0
            init_state[3] = 0.0
            state = env.reset(initial_state=init_state)
        else:
            state = env.reset()

        init_distance = float(np.linalg.norm(state[:2] - env.target_pos))
        total_reward = 0.0
        steps_taken = 0
        success = False
        hit_boundary_any = False

        traj_xy = [state[:2].copy()]

        while True:
            action = controller.get_action(state, env.target_pos)
            action = np.clip(action, -1.0, 1.0)
            next_state, reward, done, info = env.step(action)

            total_reward += float(reward)
            steps_taken += 1
            traj_xy.append(next_state[:2].copy())

            if bool(info.get("hit_boundary", False)):
                hit_boundary_any = True

            if float(info.get("distance", 1e9)) < float(env.reach_threshold):
                success = True
                state = next_state
                break

            if done:
                state = next_state
                break

            state = next_state

        if hit_boundary_any:
            boundary_hit_episodes += 1

        final_distance = float(np.linalg.norm(state[:2] - env.target_pos))
        results.append(
            {
                "test_idx": test_idx + 1,
                "init_distance": init_distance,
                "total_reward": total_reward,
                "steps_taken": steps_taken,
                "success": success,
                "final_distance": final_distance,
                "hit_boundary": hit_boundary_any,
            }
        )
        trajectories.append(np.asarray(traj_xy, dtype=np.float32))

        print(
            f"Test {test_idx + 1}: init_dist={init_distance:.4f}, success={success}, steps={steps_taken}, "
            f"reward={total_reward:.2f}, final_dist={final_distance:.4f}, hit_boundary={hit_boundary_any}"
        )

    success_rate = sum(1 for r in results if r["success"]) / max(1, num_tests)
    avg_steps = float(np.mean([r["steps_taken"] for r in results]))
    avg_reward = float(np.mean([r["total_reward"] for r in results]))
    avg_final_distance = float(np.mean([r["final_distance"] for r in results]))

    print("\nSummary Statistics:")
    print(f"Success Rate: {success_rate:.2%}")
    print(f"Average Steps: {avg_steps:.2f}")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Final Distance: {avg_final_distance:.4f}")
    print(f"Boundary Hit Rate: {boundary_hit_episodes}/{num_tests} ({(boundary_hit_episodes/num_tests*100.0):.1f}%)")

    plot_trajectories(trajectories, np.asarray(env.target_pos, dtype=np.float32), plot_path=plot_path, show_plot=show_plot)

    return results, trajectories


def main() -> None:
    parser = argparse.ArgumentParser(description="Test RL-MPPI controller for 2D ball using SAC prior")
    parser.add_argument("--model_path", type=str, default=os.path.join("..", "sac", "sac_ball", "models", "sac_ball_model_online.pth"))
    parser.add_argument("--target_x", type=float, default=3.0)
    parser.add_argument("--target_y", type=float, default=3.0)
    parser.add_argument("--num_tests", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--plot_path", type=str, default="outputs/ball_trajectories_rl_mppi.png")
    parser.add_argument("--show_plot", action="store_true")
    parser.add_argument("--init_near_target_radius", type=float, default=None)

    # RL-MPPI hyperparameters
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--lambda_coeff", type=float, default=0.2)
    parser.add_argument("--noise_std", type=float, default=0.6)

    args = parser.parse_args()

    test_rl_mppi_ball(
        model_path=str(args.model_path),
        target_pos=[float(args.target_x), float(args.target_y)],
        num_tests=int(args.num_tests),
        max_steps=int(args.max_steps),
        plot_path=str(args.plot_path),
        show_plot=bool(args.show_plot),
        init_near_target_radius=args.init_near_target_radius,
        horizon=int(args.horizon),
        num_samples=int(args.num_samples),
        lambda_coeff=float(args.lambda_coeff),
        noise_std=float(args.noise_std),
    )


if __name__ == "__main__":
    main()
