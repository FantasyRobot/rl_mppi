import argparse
import os
import sys

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
ROOT_DIR = os.path.dirname(SCRIPTS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from envball_hrsga import HRSGABallEnvironment
from hrsga_ball_model import load_agent_from_checkpoint


BALL_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:brown", "tab:pink"]


def _ensure_parent_dir(path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def _agent_is_idle(agent_snapshot):
    return (
        int(agent_snapshot.get("task_id", -1)) < 0
        and int(agent_snapshot.get("service_task_id", -1)) < 0
        and not bool(agent_snapshot.get("is_servicing", False))
        and not bool(agent_snapshot.get("is_returning_home", False))
    )


def build_eval_agent(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return load_agent_from_checkpoint(model_path)


def test_hrsga_ball(
    model_path,
    num_tests=5,
    max_steps=None,
    plot_path="hrsga_ball_eval.png",
    gif_path=None,
    gif_fps=6,
    show_plot=False,
    seed=5300,
    layout_mode="representative",
    num_balls=None,
    num_tasks=None,
):
    agent = build_eval_agent(model_path)
    if max_steps is None:
        max_steps = agent.max_steps
    resolved_num_balls = agent.num_balls if num_balls is None else int(num_balls)
    resolved_num_tasks = agent.num_tasks if num_tasks is None else num_tasks
    resolved_num_tasks = int(max(resolved_num_balls, resolved_num_tasks if resolved_num_tasks is not None else resolved_num_balls * 2))

    all_runs = []
    layout_label = layout_mode
    print(f"Testing HRSGA ball agent for {num_tests} runs with {layout_label} start/task layouts...")
    for test_idx in range(num_tests):
        env = HRSGABallEnvironment(num_balls=resolved_num_balls, num_tasks=resolved_num_tasks, max_steps=max_steps)
        snapshot = env.reset(seed=seed + test_idx, layout_mode=layout_mode)
        trajectories = [[agent_state["pos"]] for agent_state in snapshot["agents"]]
        rollout_snapshots = [snapshot]
        info = {}
        while True:
            action = agent.select_action(snapshot)
            snapshot, done, info = env.step(action)
            rollout_snapshots.append(snapshot)
            for ball_idx, agent_state in enumerate(snapshot["agents"]):
                trajectories[ball_idx].append(agent_state["pos"])
            if done:
                break

        print(
            f"Test {test_idx + 1}: Success={bool(info.get('success', False))}, Steps={snapshot['step']}, "
            f"Completed={int(info.get('completed_tasks', 0))}/{len(snapshot['tasks'])}, "
            f"Deadline={float(info.get('deadline_satisfaction', 0.0)):.3f}, "
            f"Collisions={int(info.get('total_collisions', 0))}, "
            f"MeanTaskDistance={float(info.get('mean_task_distance', 0.0)):.3f}"
        )
        all_runs.append(
            {
                "trajectories": [np.asarray(traj, dtype=np.float32) for traj in trajectories],
                "snapshot": snapshot,
                "snapshots": rollout_snapshots,
                "idle_flags": [_agent_is_idle(agent_state) for agent_state in snapshot["agents"]],
            }
        )

    plot_trajectories(all_runs, plot_path, show_plot)
    if gif_path:
        save_rollout_gif(all_runs, gif_path=gif_path, fps=gif_fps)


def plot_trajectories(runs, plot_path, show_plot):
    _ensure_parent_dir(plot_path)
    fig, axes = plt.subplots(1, len(runs), figsize=(6 * len(runs), 6), squeeze=False)
    for run_idx, run in enumerate(runs):
        ax = axes[0, run_idx]
        snapshot = run["snapshot"]
        for obstacle in snapshot["obstacles"]:
            patch = plt.Circle(obstacle["center"], obstacle["radius"], color="gray", alpha=0.25)
            ax.add_patch(patch)
        for ball_idx, traj in enumerate(run["trajectories"]):
            color = BALL_COLORS[ball_idx % len(BALL_COLORS)]
            is_idle = bool(run["idle_flags"][ball_idx])
            alpha = 0.35 if is_idle else 1.0
            linestyle = "--" if is_idle else "-"
            label = f"Ball {ball_idx}" if not is_idle else f"Ball {ball_idx} idle"
            ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=2.0, linestyle=linestyle, alpha=alpha, label=label)
            ax.scatter(traj[0, 0], traj[0, 1], color=color, marker="o", s=40)
            ax.scatter(traj[-1, 0], traj[-1, 1], color=color, marker="s", s=40, alpha=alpha)
            if is_idle:
                ax.text(traj[-1, 0] + 0.08, traj[-1, 1] - 0.12, "idle", color=color, fontsize=8)
        for task_idx, task in enumerate(snapshot["tasks"]):
            color = BALL_COLORS[task_idx % len(BALL_COLORS)]
            marker = "*" if task["completed"] else "x"
            ax.scatter(task["pos"][0], task["pos"][1], color=color, marker=marker, s=160, alpha=0.95)
        ax.set_title(f"Run {run_idx + 1}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(-5.1, 5.1)
        ax.set_ylim(-5.1, 5.1)
        ax.grid(True)
        ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(plot_path)
    if show_plot:
        plt.show()
    plt.close(fig)


def _draw_run_frame(ax, run_idx, run, frame_idx):
    frame_idx = min(frame_idx, len(run["snapshots"]) - 1)
    snapshot = run["snapshots"][frame_idx]
    ax.clear()
    for obstacle in snapshot["obstacles"]:
        patch = plt.Circle(obstacle["center"], obstacle["radius"], color="gray", alpha=0.25)
        ax.add_patch(patch)
    for ball_idx, traj in enumerate(run["trajectories"]):
        color = BALL_COLORS[ball_idx % len(BALL_COLORS)]
        is_idle = _agent_is_idle(snapshot["agents"][ball_idx])
        path = traj[: frame_idx + 1]
        alpha = 0.35 if is_idle else 1.0
        linestyle = "--" if is_idle else "-"
        label = f"Ball {ball_idx}" if not is_idle else f"Ball {ball_idx} idle"
        ax.plot(path[:, 0], path[:, 1], color=color, linewidth=2.0, linestyle=linestyle, alpha=alpha, label=label)
        ax.scatter(traj[0, 0], traj[0, 1], color=color, marker="o", s=40)
        ax.scatter(path[-1, 0], path[-1, 1], color=color, marker="s", s=40, alpha=alpha)
        if is_idle:
            ax.text(path[-1, 0] + 0.08, path[-1, 1] - 0.12, "idle", color=color, fontsize=8)
    for task_idx, task in enumerate(snapshot["tasks"]):
        color = BALL_COLORS[task_idx % len(BALL_COLORS)]
        marker = "*" if task["completed"] else "x"
        label = f"T{int(task.get('visit_rank', task_idx + 1))}/d{int(task.get('dwell_steps', 1))}"
        ax.scatter(task["pos"][0], task["pos"][1], color=color, marker=marker, s=160, alpha=0.95)
        ax.text(task["pos"][0] + 0.08, task["pos"][1] + 0.08, label, color=color, fontsize=8, weight="bold")
    ax.set_title(f"Run {run_idx + 1} | Step {snapshot['step']}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-5.1, 5.1)
    ax.set_ylim(-5.1, 5.1)
    ax.grid(True)
    ax.legend(loc="upper left")


def save_rollout_gif(runs, gif_path, fps=6):
    _ensure_parent_dir(gif_path)
    fig, axes = plt.subplots(1, len(runs), figsize=(6 * len(runs), 6), squeeze=False)
    axes_row = axes[0]
    max_frames = max(len(run["snapshots"]) for run in runs)

    def update(frame_idx):
        for run_idx, (ax, run) in enumerate(zip(axes_row, runs)):
            _draw_run_frame(ax, run_idx, run, frame_idx)
        fig.suptitle("HRSGA Rollout Animation")
        return []

    anim = animation.FuncAnimation(fig, update, frames=max_frames, interval=max(1, int(1000 / max(1, fps))), blit=False)
    try:
        writer = animation.PillowWriter(fps=max(1, int(fps)))
        anim.save(gif_path, writer=writer)
        print(f"[SAVE] gif={gif_path}")
    except ImportError as error:
        raise RuntimeError("Saving GIF requires Pillow to be installed in the Python environment.") from error
    finally:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained HRSGA policy")
    parser.add_argument("--model_path", type=str, default=os.path.join(ROOT_DIR, "models", "hrsga_ball_best.pt"))
    parser.add_argument("--num_tests", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--plot_path", type=str, default="hrsga_ball_eval.png")
    parser.add_argument("--gif_path", type=str, default=None)
    parser.add_argument("--gif_fps", type=int, default=6)
    parser.add_argument("--show_plot", action="store_true")
    parser.add_argument("--seed", type=int, default=5300)
    parser.add_argument("--layout_mode", type=str, choices=["representative", "structured", "random"], default="representative")
    parser.add_argument("--num_balls", type=int, default=None)
    parser.add_argument("--num_tasks", type=int, default=None)
    parser.add_argument("--fixed_layout", action="store_true", help="Use the original structured start/task layout")
    parser.add_argument("--random_layout", action="store_true", help="Use fully random start/task generation")
    args = parser.parse_args()
    layout_mode = "structured" if args.fixed_layout else "random" if args.random_layout else args.layout_mode
    test_hrsga_ball(
        model_path=args.model_path,
        num_tests=args.num_tests,
        max_steps=args.max_steps,
        plot_path=args.plot_path,
        gif_path=args.gif_path,
        gif_fps=args.gif_fps,
        show_plot=args.show_plot,
        seed=args.seed,
        layout_mode=layout_mode,
        num_balls=args.num_balls,
        num_tasks=args.num_tasks,
    )


if __name__ == "__main__":
    main()