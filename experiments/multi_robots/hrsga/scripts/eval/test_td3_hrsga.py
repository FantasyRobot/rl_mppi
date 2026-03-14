import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
ROOT_DIR = os.path.dirname(SCRIPTS_DIR)
for candidate in (ROOT_DIR, os.path.join(SCRIPTS_DIR, "train")):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from env_hrsga import HRSGAEnvironment
from mujoco_interface import visualize_rollout_in_mujoco
from train_td3_hrsga import load_agent_from_checkpoint


ROBOT_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:brown", "tab:pink"]


def _ensure_parent_dir(path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def test_td3_hrsga(
    model_path,
    num_tests=5,
    max_steps=None,
    plot_path="td3_hrsga_eval.png",
    show_plot=False,
    seed=5300,
    layout_mode="representative",
    xml_path=None,
    viewer_mode="matplotlib",
    playback_dt=0.08,
    traj_width=4.0,
    exit_on_done=False,
):
    agent = load_agent_from_checkpoint(model_path)
    if max_steps is None:
        max_steps = agent.max_steps

    all_runs = []
    print(f"Testing TD3-HRSGA agent for {num_tests} runs with {layout_mode} start/task layouts...")
    for test_idx in range(num_tests):
        env = HRSGAEnvironment(max_steps=max_steps, xml_path=xml_path or getattr(agent, "xml_path", None))
        snapshot = env.reset(seed=seed + test_idx, layout_mode=layout_mode)
        trajectories = [[agent_state["pos"]] for agent_state in snapshot["agents"]]
        rollout_snapshots = [snapshot]
        rollout_joint_targets = []
        info = {}
        while True:
            action = agent.select_action(snapshot, noise_sigma=0.0)
            rollout_joint_targets.append(np.asarray(action, dtype=np.float32))
            snapshot, done, info = env.step(action)
            rollout_snapshots.append(snapshot)
            for robot_idx, agent_state in enumerate(snapshot["agents"]):
                trajectories[robot_idx].append(agent_state["pos"])
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
                "joint_targets": rollout_joint_targets,
                "control_substeps": int(env.control_substeps),
            }
        )

    resolved_xml_path = os.path.abspath(xml_path or getattr(agent, "xml_path", None) or os.path.join(ROOT_DIR, "..", "urdf", "multi_robots.xml"))
    if viewer_mode == "mujoco":
        for run_idx, run in enumerate(all_runs):
            visualize_rollout_in_mujoco(
                resolved_xml_path,
                run,
                title=f"TD3-HRSGA Run {run_idx + 1}/{len(all_runs)}",
                playback_dt=playback_dt,
                traj_width=traj_width,
                exit_on_done=exit_on_done,
            )
        return

    plot_trajectories(all_runs, plot_path, show_plot)


def plot_trajectories(runs, plot_path, show_plot):
    _ensure_parent_dir(plot_path)
    fig, axes = plt.subplots(1, len(runs), figsize=(6 * len(runs), 6), squeeze=False)
    for run_idx, run in enumerate(runs):
        ax = axes[0, run_idx]
        snapshot = run["snapshot"]
        for obstacle in snapshot["obstacles"]:
            patch = plt.Circle(obstacle["center"], obstacle["radius"], color="gray", alpha=0.25)
            ax.add_patch(patch)
        for robot_idx, traj in enumerate(run["trajectories"]):
            color = ROBOT_COLORS[robot_idx % len(ROBOT_COLORS)]
            ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=2.0, label=f"Robot {robot_idx}")
            ax.scatter(traj[0, 0], traj[0, 1], color=color, marker="o", s=40)
            ax.scatter(traj[-1, 0], traj[-1, 1], color=color, marker="s", s=40)
        for task_idx, task in enumerate(snapshot["tasks"]):
            color = ROBOT_COLORS[task_idx % len(ROBOT_COLORS)]
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained TD3-HRSGA policy")
    parser.add_argument("--model_path", type=str, default=os.path.join(ROOT_DIR, "models", "td3_hrsga_best.pt"))
    parser.add_argument("--num_tests", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--plot_path", type=str, default="td3_hrsga_eval.png")
    parser.add_argument("--show_plot", action="store_true")
    parser.add_argument("--seed", type=int, default=5300)
    parser.add_argument("--layout_mode", type=str, choices=["representative", "structured", "random"], default="representative")
    parser.add_argument("--xml_path", type=str, default=None)
    parser.add_argument("--viewer_mode", type=str, choices=["matplotlib", "mujoco"], default="matplotlib")
    parser.add_argument("--playback_dt", type=float, default=0.08)
    parser.add_argument("--traj_width", type=float, default=4.0)
    parser.add_argument("--exit_on_done", action="store_true")
    parser.add_argument("--fixed_layout", action="store_true", help="Use the original structured start/task layout")
    parser.add_argument("--random_layout", action="store_true", help="Use fully random start/task generation")
    args = parser.parse_args()
    layout_mode = "structured" if args.fixed_layout else "random" if args.random_layout else args.layout_mode
    test_td3_hrsga(
        model_path=args.model_path,
        num_tests=args.num_tests,
        max_steps=args.max_steps,
        plot_path=args.plot_path,
        show_plot=args.show_plot,
        seed=args.seed,
        layout_mode=layout_mode,
        xml_path=args.xml_path,
        viewer_mode=args.viewer_mode,
        playback_dt=args.playback_dt,
        traj_width=args.traj_width,
        exit_on_done=args.exit_on_done,
    )


if __name__ == "__main__":
    main()