#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Make sure we can import `env` and `sac_utils` when running this file directly.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SAC_DIR = os.path.dirname(_THIS_DIR)
_ROOT_DIR = os.path.dirname(_SAC_DIR)
for _p in (_THIS_DIR, _SAC_DIR, _ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_RESULTS_DIR = os.path.join(_ROOT_DIR, "experiments", "results")


def _resolve_plot_path(plot_path: str) -> str:
    plot_path = os.path.expanduser(os.path.expandvars(str(plot_path)))
    if not os.path.isabs(plot_path) and os.path.dirname(plot_path) == "":
        plot_path = os.path.join(_RESULTS_DIR, plot_path)
    out_dir = os.path.dirname(plot_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    return plot_path

from env.envball_utils import BallEnvironment
from algorithms.sac.sac_utils import SACAgent, scale_state

def load_model(save_path, state_dim, action_dim):
    """Load a trained SAC agent model."""
    if not os.path.exists(save_path):
        raise FileNotFoundError(
            f"Model file not found: {save_path}. "
            "Train first (sac_ball_cli.py train_online) or pass a valid --model_path."
        )

    import torch
    checkpoint = torch.load(save_path, map_location="cpu")

    auto_entropy_tuning = bool(checkpoint.get("auto_entropy_tuning", False))

    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        learning_rate=3e-4,
        alpha=float(checkpoint.get("alpha", 0.2)),
        gamma=0.99,
        tau=0.005,
        auto_entropy_tuning=auto_entropy_tuning,
        use_lr_scheduler=False,
    )
    agent.policy_net.load_state_dict(checkpoint['policy_state_dict'])
    agent.q_net1.load_state_dict(checkpoint['q1_state_dict'])
    agent.q_net2.load_state_dict(checkpoint['q2_state_dict'])
    agent.target_q_net1.load_state_dict(checkpoint['target_q1_state_dict'])
    agent.target_q_net2.load_state_dict(checkpoint['target_q2_state_dict'])

    agent.alpha = float(checkpoint.get('alpha', agent.alpha))
    if agent.auto_entropy_tuning:
        log_alpha = checkpoint.get('log_alpha', None)
        if log_alpha is not None:
            # log_alpha may be a Tensor saved from training
            if isinstance(log_alpha, torch.Tensor):
                agent.log_alpha = log_alpha.to(torch.float32)
                agent.alpha = float(agent.log_alpha.exp().item())
            else:
                # Fallback for odd formats
                agent.log_alpha = torch.tensor(float(log_alpha), requires_grad=False)
                agent.alpha = float(agent.log_alpha.exp().item())
    
    return agent, checkpoint

def test_sac_ball(
    model_path,
    target_pos=None,
    num_tests=10,
    max_steps=100,
    plot_path="ball_trajectories.png",
    show_plot=False,
    init_near_target_radius=None,
    normalize_state: bool | None = None,
):
    """
    Test the trained SAC agent on the ball environment.
    
    Args:
        model_path (str): Path to the trained model
        target_pos (list): Target position for the environment
        num_tests (int): Number of test runs
        max_steps (int): Maximum steps per test run
    """
    # Create environment
    env = BallEnvironment(target_pos=target_pos, max_steps=max_steps, reward_scale=100.0)
    state_dim = env.state_dim
    action_dim = env.action_dim
    
    # Load trained model
    print(f"Loading model from {model_path}...")
    agent, checkpoint = load_model(model_path, state_dim, action_dim)

    # Auto-enable fixed normalization if the checkpoint indicates it.
    ckpt_norm = checkpoint.get("state_norm", None)
    if normalize_state is None:
        normalize_state = (ckpt_norm in ("fixed_bounds", "fixed_bounds_relative"))

    def _norm_state(s: np.ndarray) -> np.ndarray:
        if not normalize_state:
            return s
        out = np.asarray(s, dtype=np.float32).copy()
        if ckpt_norm == "fixed_bounds_relative":
            out[0] = (out[0] - float(env.target_pos[0])) / float(env.pos_bound)
            out[1] = (out[1] - float(env.target_pos[1])) / float(env.pos_bound)
        else:
            out[0] = out[0] / float(env.pos_bound)
            out[1] = out[1] / float(env.pos_bound)
        out[2] = out[2] / float(env.vel_bound)
        out[3] = out[3] / float(env.vel_bound)
        return out
    # Try to load scalers saved during training
    scalers = None
    scaler_path = model_path + ".scalers.npz"
    if os.path.exists(scaler_path):
        try:
            data = np.load(scaler_path)
            scalers = {"x_min": data['x_min'], "x_max": data['x_max'], "u_min": data['u_min'], "u_max": data['u_max']}
            print(f"Loaded scalers from {scaler_path}")
        except Exception:
            scalers = None
    
    # Test results
    test_results = []
    all_trajectories = []
    
    print(f"Testing agent for {num_tests} runs...")

    boundary_hit_episodes = 0
    
    for test_idx in range(num_tests):
        # Reset environment
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
        trajectory = [state[:2]]  # Only store position for plotting
        total_reward = 0
        steps_taken = 0
        success = False
        distances = []
        action_debug = []
        hit_boundary_any = False
        while True:
            # If already within reach threshold, stop without extra policy compute.
            if float(np.linalg.norm(state[:2] - env.target_pos)) < float(env.reach_threshold):
                success = True
                distances.append(float(np.linalg.norm(state[:2] - env.target_pos)))
                break
            # 状态归一化与训练一致
            if scalers is not None:
                scaled_state = scale_state(state, scalers)
            else:
                scaled_state = state
            scaled_state = _norm_state(scaled_state)
            # 动作直接用SAC输出（[-1,1]），不做反归一化
            action = agent.select_action(scaled_state, evaluate=True)
            action_debug.append(action)
            # Clip动作，确保在[-1,1]
            action = np.clip(action, -1.0, 1.0)
            next_state, reward, done, info = env.step(action)
            if bool(info.get("hit_boundary", False)):
                hit_boundary_any = True
            total_reward += reward
            steps_taken += 1
            distances.append(info["distance"])
            trajectory.append(next_state[:2])
            if float(info["distance"]) < float(env.reach_threshold):
                success = True
                break
            if done:
                break
            state = next_state
        if hit_boundary_any:
            boundary_hit_episodes += 1
        # Debug输出动作范围
        action_debug_arr = np.asarray(action_debug, dtype=float)
        if action_debug_arr.size == 0:
            print(f"Test {test_idx + 1} action range: (no actions recorded)")
        else:
            print(
                f"Test {test_idx + 1} action range: min {action_debug_arr.min(axis=0)}, max {action_debug_arr.max(axis=0)}"
            )
        print(f"Test {test_idx + 1} hit_boundary={hit_boundary_any}")
        test_results.append({
            "test_idx": test_idx + 1,
            "init_distance": init_distance,
            "total_reward": total_reward,
            "steps_taken": steps_taken,
            "success": success,
            "final_distance": distances[-1]
        })
        all_trajectories.append(np.array(trajectory))
        print(
            f"Test {test_idx + 1}: init_dist={init_distance:.4f}, Success={success}, Steps={steps_taken}, "
            f"Reward={total_reward:.2f}, Final Distance={distances[-1]:.4f}"
        )
    
    # Calculate summary statistics
    success_rate = sum(1 for r in test_results if r["success"]) / num_tests
    avg_steps = np.mean([r["steps_taken"] for r in test_results])
    avg_reward = np.mean([r["total_reward"] for r in test_results])
    avg_final_distance = np.mean([r["final_distance"] for r in test_results])
    
    print(f"\nSummary Statistics:")
    print(f"Success Rate: {success_rate:.2%}")
    print(f"Average Steps: {avg_steps:.2f}")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Final Distance: {avg_final_distance:.4f}")
    print(f"Boundary Hit Rate: {boundary_hit_episodes}/{num_tests} ({(boundary_hit_episodes/num_tests*100.0):.1f}%)")
    
    # Plot trajectories (default: save only, do not block GUI)
    plot_trajectories(all_trajectories, env.target_pos, plot_path=plot_path, show_plot=show_plot)
    
    return test_results, all_trajectories

def plot_trajectories(trajectories, target_pos, plot_path="ball_trajectories.png", show_plot=False):
    """
    Plot all test trajectories.
    
    Args:
        trajectories (list): List of numpy arrays representing trajectories
        target_pos (list): Target position
    """
    plt.figure(figsize=(10, 10))
    
    # Plot each trajectory
    for i, traj in enumerate(trajectories):
        plt.plot(traj[:, 0], traj[:, 1], marker='o', markersize=3, label=f'Test {i+1}')
        # Plot start point for each trajectory
        plt.plot(traj[0, 0], traj[0, 1], 's', markersize=8, color='green', label=f'Test {i+1} Start' if i == 0 else "")
    
    # Plot target position
    plt.plot(target_pos[0], target_pos[1], 'rx', markersize=15, label='Target')
    
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.grid(True)
    plt.title('Ball Trajectories - SAC Agent')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend(loc='upper right')
    
    plot_path = _resolve_plot_path(plot_path)
    plt.savefig(plot_path)
    print(f"Trajectory plot saved to {plot_path}")
    if show_plot:
        plt.show()
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Test SAC agent for ball environment")
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.path.join(_ROOT_DIR, "experiments", "sac_ball", "models", "sac_ball_model_online.pth"),
        help="Path to the trained model",
    )
    parser.add_argument("--target_x", type=float, default=3.0, help="Target x position")
    parser.add_argument("--target_y", type=float, default=3.0, help="Target y position")
    parser.add_argument("--num_tests", type=int, default=10, help="Number of test runs")
    parser.add_argument("--max_steps", type=int, default=2000, help="Maximum steps per test run")
    parser.add_argument(
        "--plot_path",
        type=str,
        default=os.path.join(_ROOT_DIR, "experiments", "results", "ball_trajectories_sac.png"),
        help="Path to save trajectory plot",
    )
    parser.add_argument("--show_plot", action="store_true", help="Show GUI plot (may block)")
    parser.add_argument(
        "--init_near_target_radius",
        type=float,
        default=None,
        help="If set, initialize each test within this radius around the target (useful for debugging)",
    )
    parser.add_argument("--normalize_state", action="store_true", help="Force enable fixed-bounds state normalization")
    parser.add_argument("--no_normalize_state", action="store_true", help="Force disable fixed-bounds state normalization")

    args = parser.parse_args()
    if not os.path.exists(args.model_path):
        raise SystemExit(
            f"Model file not found: {args.model_path}\n"
            "Train first: python sac_ball_cli.py train_online\n"
            "Or pass:   --model_path path/to/model.pth"
        )
    target_pos = [args.target_x, args.target_y]

    test_sac_ball(
        model_path=args.model_path,
        target_pos=target_pos,
        num_tests=args.num_tests,
        max_steps=args.max_steps,
        plot_path=args.plot_path,
        show_plot=args.show_plot,
        init_near_target_radius=args.init_near_target_radius,
        normalize_state=(False if args.no_normalize_state else (True if args.normalize_state else None)),
    )

if __name__ == "__main__":
    main()