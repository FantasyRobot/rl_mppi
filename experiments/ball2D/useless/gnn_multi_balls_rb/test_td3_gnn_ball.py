import os

import matplotlib.pyplot as plt
import numpy as np

from envball_gnn_rb import BallRoboBalletEnvironment
from train_td3_gnn_ball import load_agent_from_checkpoint


BALL_COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']


def build_eval_agent(model_path):
    """Load a saved TD3-GNN checkpoint and return an evaluation-ready agent."""

    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model file not found: {model_path}')
    return load_agent_from_checkpoint(model_path)


def test_td3_gnn_ball(model_path, num_tests=5, max_steps=320, plot_path='gnn_multi_ball_rb_eval.png', show_plot=False):
    """Run multiple rollouts and visualize RoboBallet-style 2D trajectories."""

    agent = build_eval_agent(model_path)
    all_runs = []
    print(f'Testing RoboBallet-style multi-ball GNN-TD3 agent for {num_tests} runs...')
    for test_idx in range(num_tests):
        env = BallRoboBalletEnvironment(max_steps=max_steps)
        obs = env.reset()
        trajectories = [[obs['ball_pos'][ball_idx].copy()] for ball_idx in range(env.num_balls)]
        total_reward = 0.0
        info = {}
        while not env.terminal():
            action = agent.select_action(obs, noise_sigma=0.0)
            obs, reward, done, info = env.step(action)
            total_reward += float(reward)
            for ball_idx in range(env.num_balls):
                trajectories[ball_idx].append(obs['ball_pos'][ball_idx].copy())
            if done:
                break
        print(
            f"Test {test_idx + 1}: Success={bool(info.get('success', False))}, Steps={env.step_count}, "
            f"TargetsDone={int(info.get('targets_done', 0))}/{env.num_targets}, Reward={total_reward:.4f}, "
            f"CollisionPenalty={float(info.get('total_collision_penalty', 0.0)):.6f}"
        )
        all_runs.append(
            {
                'trajectories': [np.asarray(traj) for traj in trajectories],
                'target_pos': obs['target_pos'].copy(),
                'target_done': obs['target_done'].copy(),
                'target_cluster': obs['target_cluster'].copy(),
            }
        )
    plot_trajectories(all_runs, plot_path, show_plot)


def plot_trajectories(runs, plot_path, show_plot):
    """Plot one rollout per subplot with solved and unsolved targets distinguished."""

    fig, axes = plt.subplots(1, len(runs), figsize=(6 * len(runs), 6), squeeze=False)
    for run_idx, run in enumerate(runs):
        ax = axes[0, run_idx]
        for ball_idx, traj in enumerate(run['trajectories']):
            ax.plot(traj[:, 0], traj[:, 1], color=BALL_COLORS[ball_idx], linewidth=2.0, label=f'Ball {ball_idx}')
            ax.scatter(traj[0, 0], traj[0, 1], color=BALL_COLORS[ball_idx], marker='o', s=40)
            ax.scatter(traj[-1, 0], traj[-1, 1], color=BALL_COLORS[ball_idx], marker='s', s=40)
        for target_idx, target_pos in enumerate(run['target_pos']):
            cluster = int(run['target_cluster'][target_idx])
            done = bool(run['target_done'][target_idx] > 0.5)
            marker = '*' if done else 'x'
            ax.scatter(target_pos[0], target_pos[1], color=BALL_COLORS[cluster], marker=marker, s=160, alpha=0.9)
        ax.set_title(f'Run {run_idx + 1}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True)
        ax.set_xlim(-5.2, 5.2)
        ax.set_ylim(-5.2, 5.2)
        ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(plot_path)
    if show_plot:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'td3_gnn_best.npz')
    test_td3_gnn_ball(model_path)