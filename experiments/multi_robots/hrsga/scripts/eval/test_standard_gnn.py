import argparse
import glob
import json
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

from env_hrsga import HRSGAEnvironment
from mujoco_interface import visualize_rollout_in_mujoco
from standard_gnn_model import load_agent_from_checkpoint


ROBOT_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:brown", "tab:pink"]


def _ensure_parent_dir(path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def _indexed_output_path(path, run_idx, run_count):
    if not path or run_count <= 1:
        return path
    root, ext = os.path.splitext(path)
    return f"{root}_run{run_idx + 1}{ext}"


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


def _load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _find_training_config_path(model_path, config_path=None):
    if config_path:
        resolved = os.path.abspath(config_path)
        return resolved if os.path.exists(resolved) else None

    resolved_model_path = os.path.abspath(model_path)
    parent_dir = os.path.basename(os.path.dirname(resolved_model_path)).lower()
    if parent_dir == "checkpoints":
        candidate = os.path.join(os.path.dirname(os.path.dirname(resolved_model_path)), "config.json")
        return candidate if os.path.exists(candidate) else None

    if parent_dir == "models":
        checkpoint_matches = sorted(
            glob.glob(os.path.join(ROOT_DIR, "runs", "*", "checkpoints", os.path.basename(resolved_model_path)))
        )
        config_candidates = [
            os.path.join(os.path.dirname(os.path.dirname(match)), "config.json")
            for match in checkpoint_matches
            if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(match)), "config.json"))
        ]
        if len(config_candidates) == 1:
            return config_candidates[0]

    return None


def _load_training_eval_defaults(model_path, *, config_path=None):
    resolved_config_path = _find_training_config_path(model_path, config_path=config_path)
    if not resolved_config_path:
        return {}, None

    payload = _load_json(resolved_config_path)
    if isinstance(payload, dict) and isinstance(payload.get("config"), dict):
        payload = payload["config"]
    return payload if isinstance(payload, dict) else {}, resolved_config_path


def _resolve_eval_settings(
    agent,
    model_path,
    *,
    config_path=None,
    num_tests=None,
    max_steps=None,
    seed=None,
    layout_mode=None,
    xml_path=None,
    enforce_visit_order=None,
    strict_collision_stop=None,
):
    training_config, resolved_config_path = _load_training_eval_defaults(model_path, config_path=config_path)
    base_seed = training_config.get("seed")
    resolved = {
        "num_tests": int(training_config.get("rollout_episodes", 5)),
        "max_steps": int(training_config.get("max_steps", getattr(agent, "max_steps", 280))),
        "seed": int(base_seed) + 10000 if base_seed is not None else 5300,
        "layout_mode": str(training_config.get("eval_layout_mode", "representative")),
        "xml_path": training_config.get("xml_path") or getattr(agent, "xml_path", None),
        "enforce_visit_order": training_config.get("enforce_visit_order"),
        "strict_collision_stop": bool(training_config.get("eval_strict_collision_stop", False)),
        "config_path": resolved_config_path,
    }

    if num_tests is not None:
        resolved["num_tests"] = int(num_tests)
    if max_steps is not None:
        resolved["max_steps"] = int(max_steps)
    if seed is not None:
        resolved["seed"] = int(seed)
    if layout_mode is not None:
        resolved["layout_mode"] = str(layout_mode)
    if xml_path is not None:
        resolved["xml_path"] = xml_path
    if enforce_visit_order is not None:
        resolved["enforce_visit_order"] = bool(enforce_visit_order)
    if strict_collision_stop is not None:
        resolved["strict_collision_stop"] = bool(strict_collision_stop)

    return resolved


def test_standard_gnn(
    model_path,
    num_tests=None,
    max_steps=None,
    plot_path="standard_gnn_eval.png",
    gif_path=None,
    gif_fps=6,
    show_plot=False,
    seed=None,
    layout_mode=None,
    xml_path=None,
    viewer_mode="matplotlib",
    playback_dt=0.08,
    traj_width=4.0,
    exit_on_done=False,
    video_path=None,
    mujoco_record_fps=12,
    mujoco_record_width=1280,
    mujoco_record_height=720,
    enforce_visit_order=None,
    strict_collision_stop=None,
    config_path=None,
):
    agent = build_eval_agent(model_path)
    resolved = _resolve_eval_settings(
        agent,
        model_path,
        config_path=config_path,
        num_tests=num_tests,
        max_steps=max_steps,
        seed=seed,
        layout_mode=layout_mode,
        xml_path=xml_path,
        enforce_visit_order=enforce_visit_order,
        strict_collision_stop=strict_collision_stop,
    )
    num_tests = int(resolved["num_tests"])
    max_steps = int(resolved["max_steps"])
    seed = int(resolved["seed"])
    layout_mode = resolved["layout_mode"]
    xml_path = resolved["xml_path"]
    enforce_visit_order = resolved["enforce_visit_order"]
    strict_collision_stop = bool(resolved["strict_collision_stop"])

    all_runs = []
    summary_runs = []
    print(
        "[EVAL] "
        f"model={os.path.abspath(model_path)} "
        f"config={resolved['config_path'] or 'none'} "
        f"num_tests={num_tests} "
        f"seed={seed} "
        f"layout_mode={layout_mode} "
        f"max_steps={max_steps} "
        f"strict_collision_stop={strict_collision_stop} "
        f"enforce_visit_order={enforce_visit_order} "
        f"xml_path={xml_path or getattr(agent, 'xml_path', None)}"
    )
    print(f"Testing Standard GNN agent for {num_tests} runs with {layout_mode} start/task layouts...")
    for test_idx in range(num_tests):
        env = HRSGAEnvironment(
            max_steps=max_steps,
            xml_path=xml_path or getattr(agent, "xml_path", None),
            enforce_visit_order=enforce_visit_order,
        )
        snapshot = env.reset(seed=seed + test_idx, layout_mode=layout_mode)
        trajectories = [[agent_state["pos"]] for agent_state in snapshot["agents"]]
        rollout_snapshots = [snapshot]
        rollout_joint_targets = []
        info = {}
        collision_terminated = False
        while True:
            action = agent.select_action(snapshot)
            rollout_joint_targets.append(np.asarray(action, dtype=np.float32))
            snapshot, done, info = env.step(action)
            if strict_collision_stop and int(info.get("collisions", 0)) > 0:
                done = True
                collision_terminated = True
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
            f"CollisionStop={collision_terminated}, "
            f"MeanTaskDistance={float(info.get('mean_task_distance', 0.0)):.3f}"
        )
        summary_runs.append(
            {
                "success": float(info.get("success", False)),
                "completed_fraction": float(info.get("completed_tasks", 0)) / max(1, len(snapshot["tasks"])),
                "deadline_satisfaction": float(info.get("deadline_satisfaction", 0.0)),
                "collisions": float(info.get("total_collisions", 0)),
                "mean_task_distance": float(info.get("mean_task_distance", 0.0)),
                "collision_terminated": float(collision_terminated),
            }
        )
        all_runs.append(
            {
                "trajectories": [np.asarray(traj, dtype=np.float32) for traj in trajectories],
                "snapshot": snapshot,
                "snapshots": rollout_snapshots,
                "joint_targets": rollout_joint_targets,
                "control_substeps": int(env.control_substeps),
                "idle_flags": [_agent_is_idle(agent_state) for agent_state in snapshot["agents"]],
            }
        )

    resolved_xml_path = os.path.abspath(
        xml_path or getattr(agent, "xml_path", None) or os.path.join(ROOT_DIR, "..", "urdf", "multi_robots.xml")
    )
    if viewer_mode == "mujoco":
        for run_idx, run in enumerate(all_runs):
            record_path = None
            if video_path:
                record_path = _indexed_output_path(video_path, run_idx, len(all_runs))
            elif gif_path:
                record_path = _indexed_output_path(gif_path, run_idx, len(all_runs))
            visualize_rollout_in_mujoco(
                resolved_xml_path,
                run,
                title=f"Standard GNN Run {run_idx + 1}/{len(all_runs)}",
                playback_dt=playback_dt,
                traj_width=traj_width,
                exit_on_done=exit_on_done,
                record_path=record_path,
                record_fps=mujoco_record_fps,
                record_width=mujoco_record_width,
                record_height=mujoco_record_height,
            )
        return

    if summary_runs:
        print(
            "[SUMMARY] "
            f"success={float(np.mean([run['success'] for run in summary_runs])):.3f} "
            f"completed={float(np.mean([run['completed_fraction'] for run in summary_runs])):.3f} "
            f"deadline={float(np.mean([run['deadline_satisfaction'] for run in summary_runs])):.3f} "
            f"collisions={float(np.mean([run['collisions'] for run in summary_runs])):.3f} "
            f"collision_stop={float(np.mean([run['collision_terminated'] for run in summary_runs])):.3f} "
            f"task_dist={float(np.mean([run['mean_task_distance'] for run in summary_runs])):.3f}"
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
        for robot_idx, traj in enumerate(run["trajectories"]):
            color = ROBOT_COLORS[robot_idx % len(ROBOT_COLORS)]
            is_idle = bool(run["idle_flags"][robot_idx])
            alpha = 0.35 if is_idle else 1.0
            linestyle = "--" if is_idle else "-"
            label = f"Robot {robot_idx}" if not is_idle else f"Robot {robot_idx} idle"
            ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=2.0, linestyle=linestyle, alpha=alpha, label=label)
            ax.scatter(traj[0, 0], traj[0, 1], color=color, marker="o", s=40)
            ax.scatter(traj[-1, 0], traj[-1, 1], color=color, marker="s", s=40, alpha=alpha)
            if is_idle:
                ax.text(traj[-1, 0] + 0.08, traj[-1, 1] - 0.12, "idle", color=color, fontsize=8)
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


def _draw_run_frame(ax, run_idx, run, frame_idx):
    frame_idx = min(frame_idx, len(run["snapshots"]) - 1)
    snapshot = run["snapshots"][frame_idx]
    ax.clear()
    for obstacle in snapshot["obstacles"]:
        patch = plt.Circle(obstacle["center"], obstacle["radius"], color="gray", alpha=0.25)
        ax.add_patch(patch)
    for robot_idx, traj in enumerate(run["trajectories"]):
        color = ROBOT_COLORS[robot_idx % len(ROBOT_COLORS)]
        is_idle = _agent_is_idle(snapshot["agents"][robot_idx])
        path = traj[: frame_idx + 1]
        alpha = 0.35 if is_idle else 1.0
        linestyle = "--" if is_idle else "-"
        label = f"Robot {robot_idx}" if not is_idle else f"Robot {robot_idx} idle"
        ax.plot(path[:, 0], path[:, 1], color=color, linewidth=2.0, linestyle=linestyle, alpha=alpha, label=label)
        ax.scatter(traj[0, 0], traj[0, 1], color=color, marker="o", s=40)
        ax.scatter(path[-1, 0], path[-1, 1], color=color, marker="s", s=40, alpha=alpha)
        if is_idle:
            ax.text(path[-1, 0] + 0.08, path[-1, 1] - 0.12, "idle", color=color, fontsize=8)
    for task_idx, task in enumerate(snapshot["tasks"]):
        color = ROBOT_COLORS[task_idx % len(ROBOT_COLORS)]
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
        fig.suptitle("Standard GNN Rollout Animation")
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
    parser = argparse.ArgumentParser(description="Evaluate a trained Standard GNN policy")
    parser.add_argument("--model_path", type=str, default=os.path.join(ROOT_DIR, "models", "standard_gnn_best.pt"))
    parser.add_argument("--config_path", type=str, default=None, help="Optional training config.json path; when omitted the script tries to infer it from the checkpoint path")
    parser.add_argument("--num_tests", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--plot_path", type=str, default="standard_gnn_eval.png")
    parser.add_argument("--gif_path", type=str, default=None)
    parser.add_argument("--gif_fps", type=int, default=6)
    parser.add_argument("--show_plot", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--layout_mode", type=str, choices=["representative", "structured", "random"], default=None)
    parser.add_argument("--xml_path", type=str, default=None)
    parser.add_argument("--viewer_mode", type=str, choices=["matplotlib", "mujoco"], default="matplotlib")
    parser.add_argument("--playback_dt", type=float, default=0.08)
    parser.add_argument("--traj_width", type=float, default=4.0)
    parser.add_argument("--exit_on_done", action="store_true")
    parser.add_argument("--video_path", type=str, default=None, help="When viewer_mode=mujoco, save each run as a video file such as .mp4")
    parser.add_argument("--mujoco_record_fps", type=int, default=12)
    parser.add_argument("--mujoco_record_width", type=int, default=1280)
    parser.add_argument("--mujoco_record_height", type=int, default=720)
    parser.add_argument("--enforce_visit_order", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--strict_collision_stop", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--fixed_layout", action="store_true", help="Use the original structured start/task layout")
    parser.add_argument("--random_layout", action="store_true", help="Use fully random start/task generation")
    args = parser.parse_args()
    layout_mode = "structured" if args.fixed_layout else "random" if args.random_layout else args.layout_mode
    test_standard_gnn(
        model_path=args.model_path,
        config_path=args.config_path,
        num_tests=args.num_tests,
        max_steps=args.max_steps,
        plot_path=args.plot_path,
        gif_path=args.gif_path,
        gif_fps=args.gif_fps,
        show_plot=args.show_plot,
        seed=args.seed,
        layout_mode=layout_mode,
        xml_path=args.xml_path,
        viewer_mode=args.viewer_mode,
        playback_dt=args.playback_dt,
        traj_width=args.traj_width,
        exit_on_done=args.exit_on_done,
        video_path=args.video_path,
        mujoco_record_fps=args.mujoco_record_fps,
        mujoco_record_width=args.mujoco_record_width,
        mujoco_record_height=args.mujoco_record_height,
        enforce_visit_order=args.enforce_visit_order,
        strict_collision_stop=args.strict_collision_stop,
    )


if __name__ == "__main__":
    main()