#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

# Ensure repo root is importable regardless of CWD.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

_RESULTS_DIR = os.path.join(_ROOT_DIR, "experiments", "results")


def _resolve_plot_path(plot_path: str) -> str:
    """Resolve plot output path to live under experiments/results by default.

    If the user passes a filename without any directory component (e.g. foo.png),
    place it under <root>/experiments/results.
    """

    plot_path = os.path.expanduser(os.path.expandvars(str(plot_path)))
    if not os.path.isabs(plot_path) and os.path.dirname(plot_path) == "":
        plot_path = os.path.join(_RESULTS_DIR, plot_path)
    out_dir = os.path.dirname(plot_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    return plot_path

from env.envball_utils import BallEnvironment
from algorithms.mppi.mppi_ball import MPPI
from algorithms.rl_mppi.rl_mppi_ball import RLMppiController, load_sac_policy


class SACController:
    """SAC-only baseline controller (deterministic policy act)."""

    def __init__(self, policy, env: BallEnvironment):
        self.policy = policy
        self.env = env

    def reset(self) -> None:
        return None

    def get_action(self, state: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
        return self.policy.act(state, env=self.env)


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
        # If we already start within the reach threshold, avoid extra controller compute.
        if float(np.linalg.norm(state[:2] - env.target_pos)) < float(env.reach_threshold):
            break
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
    sac_stats: dict,
    rl_mppi_stats: dict,
    target_pos: np.ndarray,
    plot_path: str,
    show_plot: bool,
) -> None:
    plot_path = _resolve_plot_path(plot_path)
    mppi_trajs = [r.get("traj_xy", None) for r in mppi_stats.get("results", [])]
    sac_trajs = [r.get("traj_xy", None) for r in sac_stats.get("results", [])]
    rl_trajs = [r.get("traj_xy", None) for r in rl_mppi_stats.get("results", [])]

    if not mppi_trajs or not sac_trajs or not rl_trajs or any(t is None for t in mppi_trajs) or any(t is None for t in sac_trajs) or any(t is None for t in rl_trajs):
        raise RuntimeError("Trajectories were not collected; re-run with plotting enabled.")

    plt.figure(figsize=(10, 10))

    for i, (tm, ts, tr) in enumerate(zip(mppi_trajs, sac_trajs, rl_trajs)):
        tm = np.asarray(tm, dtype=np.float32)
        ts = np.asarray(ts, dtype=np.float32)
        tr = np.asarray(tr, dtype=np.float32)
        plt.plot(tm[:, 0], tm[:, 1], color="tab:blue", alpha=0.6, linewidth=1.5, label="MPPI" if i == 0 else "")
        plt.plot(ts[:, 0], ts[:, 1], color="tab:green", alpha=0.6, linewidth=1.5, linestyle=":", label="SAC" if i == 0 else "")
        plt.plot(tr[:, 0], tr[:, 1], color="tab:orange", alpha=0.6, linewidth=1.5, linestyle="--", label="RL-MPPI" if i == 0 else "")
        plt.plot(tm[0, 0], tm[0, 1], "s", markersize=6, color="green", label="Start" if i == 0 else "")

    plt.plot(float(target_pos[0]), float(target_pos[1]), "rx", markersize=12, label="Target")
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.grid(True)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("MPPI vs SAC vs RL-MPPI trajectories")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(loc="upper right")

    plt.savefig(plot_path)
    print(f"Trajectory comparison plot saved to {plot_path}")
    if show_plot:
        plt.show()
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare MPPI vs SAC vs RL-guided MPPI on BallEnvironment")
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.path.join(_ROOT_DIR, "experiments", "sac_ball", "models", "sac_ball_model_online.pth"),
        help="SAC checkpoint path for RL-MPPI",
    )
    parser.add_argument("--target_x", type=float, default=3.0)
    parser.add_argument("--target_y", type=float, default=3.0)
    parser.add_argument(
        "--target_from_checkpoint",
        action="store_true",
        help="Use target_pos stored in the SAC checkpoint metadata (if present)",
    )
    parser.add_argument("--num_tests", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=2000)

    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--lambda_coeff", type=float, default=0.2)
    parser.add_argument("--noise_std", type=float, default=0.6)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_vectorized", action="store_true", help="Disable vectorized rollouts (debug/legacy)")
    parser.add_argument(
        "--plot_path",
        type=str,
        default=os.path.join(_ROOT_DIR, "experiments", "results", "compare_all.png"),
    )
    parser.add_argument("--show_plot", action="store_true")
    parser.add_argument("--no_plot", action="store_true", help="Disable trajectory collection/plotting")

    args = parser.parse_args()

    ckpt_target = None
    if bool(args.target_from_checkpoint):
        try:
            import torch

            ckpt = torch.load(str(args.model_path), map_location="cpu")
            ckpt_target = ckpt.get("target_pos", None)
        except Exception as e:
            raise SystemExit(f"Failed to read checkpoint metadata from {args.model_path}: {e}")
        if ckpt_target is None:
            raise SystemExit(
                "Checkpoint does not contain target_pos metadata. "
                "Re-train with updated training code or pass --target_x/--target_y explicitly."
            )
        target_pos = np.asarray([float(ckpt_target[0]), float(ckpt_target[1])], dtype=np.float32)
    else:
        target_pos = np.asarray([float(args.target_x), float(args.target_y)], dtype=np.float32)

    def make_env() -> BallEnvironment:
        return BallEnvironment(target_pos=target_pos.tolist(), max_steps=int(args.max_steps), reward_scale=100.0)

    env0 = make_env()
    policy = load_sac_policy(str(args.model_path), state_dim=env0.state_dim, action_dim=env0.action_dim)

    if ckpt_target is None:
        try:
            import torch

            ckpt = torch.load(str(args.model_path), map_location="cpu")
            ckpt_target = ckpt.get("target_pos", None)
        except Exception:
            ckpt_target = None
    if ckpt_target is not None:
        ckpt_target_arr = np.asarray(ckpt_target, dtype=np.float32).reshape(2)
        if np.linalg.norm(ckpt_target_arr - target_pos) > 1e-6:
            print(
                f"[WARN] Checkpoint target_pos={ckpt_target_arr.tolist()} differs from evaluation target_pos={target_pos.tolist()}\n"
                "       SAC-only and RL-MPPI priors may perform poorly unless the policy is target-conditioned.\n"
                "       Consider using --target_from_checkpoint."
            )

    def make_sac_controller(env: BallEnvironment) -> SACController:
        return SACController(policy, env)

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
    sac_stats = run_benchmark(
        name="SAC",
        make_env=make_env,
        make_controller=make_sac_controller,
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
    _print(sac_stats)
    _print(rl_mppi_stats)

    speed_mppi_vs_rl = mppi_stats["total_time_s"] / max(1e-9, rl_mppi_stats["total_time_s"])
    speed_sac_vs_rl = sac_stats["total_time_s"] / max(1e-9, rl_mppi_stats["total_time_s"])
    print(f"\nSpeed (MPPI total / RL-MPPI total): {speed_mppi_vs_rl:.2f}x")
    print(f"Speed (SAC total / RL-MPPI total): {speed_sac_vs_rl:.2f}x")

    if collect_trajectories:
        plot_comparison(
            mppi_stats=mppi_stats,
            sac_stats=sac_stats,
            rl_mppi_stats=rl_mppi_stats,
            target_pos=target_pos,
            plot_path=str(args.plot_path),
            show_plot=bool(args.show_plot),
        )


if __name__ == "__main__":
    main()
