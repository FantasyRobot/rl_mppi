import argparse
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

from envball_hrsga import HRSGABallEnvironment
from hrsga_ball_model import load_agent_from_checkpoint as load_hrsga_agent
from standard_gnn_ball_model import load_agent_from_checkpoint as load_gnn_agent


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


def _task_rule_config(snapshot, enforce_visit_order):
    dwell_values = [int(task.get("dwell_steps", 1)) for task in snapshot["tasks"]]
    return {
        "visit_order_mode": "ranked" if enforce_visit_order else "free",
        "enforce_visit_order": bool(enforce_visit_order),
        "min_dwell_steps": min(dwell_values) if dwell_values else 0,
        "max_dwell_steps": max(dwell_values) if dwell_values else 0,
    }


def _task_metadata(snapshot):
    return [
        {
            "index": int(task_index),
            "pos": [float(task["pos"][0]), float(task["pos"][1])],
            "visit_rank": int(task.get("visit_rank", task_index + 1)),
            "dwell_steps": int(task.get("dwell_steps", 1)),
        }
        for task_index, task in enumerate(snapshot["tasks"])
    ]


def _format_task_label(task_idx, task, task_label_mode):
    if task_label_mode == "none":
        return None
    if task_label_mode == "index":
        return f"T{task_idx + 1}/d{int(task.get('dwell_steps', 1))}"
    return f"T{int(task.get('visit_rank', task_idx + 1))}/d{int(task.get('dwell_steps', 1))}"


def _plot_task(ax, task_idx, task, task_label_mode):
    color = BALL_COLORS[task_idx % len(BALL_COLORS)]
    marker = "*" if task["completed"] else "x"
    ax.scatter(task["pos"][0], task["pos"][1], color=color, marker=marker, s=140, alpha=0.95)
    label = _format_task_label(task_idx, task, task_label_mode)
    if label is not None:
        ax.text(task["pos"][0] + 0.08, task["pos"][1] + 0.08, label, color=color, fontsize=8, weight="bold")


def save_json(path, payload):
    _ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def build_default_output_paths(layout_mode):
    root = os.path.join(ROOT_DIR, "runs", "benchmarks")
    tag = f"compare_trained_gnn_hrsga_{layout_mode}"
    return {
        "plot": os.path.join(root, f"{tag}.png"),
        "summary": os.path.join(root, f"{tag}.json"),
    }


def rollout(agent, seed, max_steps=None, layout_mode="representative", num_balls=None, num_tasks=None, enforce_visit_order=True):
    max_steps = agent.max_steps if max_steps is None else max_steps
    resolved_num_balls = agent.num_balls if num_balls is None else int(num_balls)
    resolved_num_tasks = getattr(agent, "num_tasks", None) if num_tasks is None else num_tasks
    resolved_num_tasks = int(max(resolved_num_balls, resolved_num_tasks if resolved_num_tasks is not None else resolved_num_balls * 2))
    env = HRSGABallEnvironment(
        num_balls=resolved_num_balls,
        num_tasks=resolved_num_tasks,
        max_steps=max_steps,
        enforce_visit_order=bool(enforce_visit_order),
    )
    snapshot = env.reset(seed=seed, layout_mode=layout_mode)
    initial_snapshot = json.loads(json.dumps(snapshot))
    trajectories = [[agent_state["pos"]] for agent_state in snapshot["agents"]]
    rollout_snapshots = [snapshot]
    info = {}
    collision_terminated = False
    while True:
        action = agent.select_action(snapshot)
        snapshot, done, info = env.step(action)
        rollout_snapshots.append(snapshot)
        for ball_idx, agent_state in enumerate(snapshot["agents"]):
            trajectories[ball_idx].append(agent_state["pos"])
        if int(info.get("collisions", 0)) > 0:
            done = True
            collision_terminated = True
        if done:
            break
    info = dict(info)
    info["collision_terminated"] = collision_terminated
    return {
        "trajectories": [np.asarray(traj, dtype=np.float32) for traj in trajectories],
        "snapshot": snapshot,
        "initial_snapshot": initial_snapshot,
        "snapshots": rollout_snapshots,
        "info": info,
    }


def compare_models(
    hrsga_model_path,
    gnn_model_path,
    num_tests=5,
    max_steps=None,
    plot_path=None,
    gif_path=None,
    gif_fps=6,
    summary_path=None,
    show_plot=False,
    seed=5300,
    layout_mode="representative",
    num_balls=None,
    num_tasks=None,
    task_label_mode="ranked",
    enforce_visit_order=True,
):
    output_paths = build_default_output_paths(layout_mode)
    plot_path = plot_path or output_paths["plot"]
    summary_path = summary_path or output_paths["summary"]
    hrsga_agent = load_hrsga_agent(hrsga_model_path)
    gnn_agent = load_gnn_agent(gnn_model_path)
    fig, axes = plt.subplots(2, num_tests, figsize=(6 * num_tests, 12), squeeze=False)
    comparison_summaries = []
    comparison_runs = []
    reference_snapshot = None
    print(f"Comparing trained Standard-GNN and HRSGA policies with {layout_mode} start/task layouts...")
    for test_idx in range(num_tests):
        run_seed = seed + test_idx
        gnn_run = rollout(
            gnn_agent,
            seed=run_seed,
            max_steps=max_steps,
            layout_mode=layout_mode,
            num_balls=num_balls,
            num_tasks=num_tasks,
            enforce_visit_order=enforce_visit_order,
        )
        hrsga_run = rollout(
            hrsga_agent,
            seed=run_seed,
            max_steps=max_steps,
            layout_mode=layout_mode,
            num_balls=num_balls,
            num_tasks=num_tasks,
            enforce_visit_order=enforce_visit_order,
        )
        if reference_snapshot is None:
            reference_snapshot = hrsga_run["initial_snapshot"]
        comparison_summaries.append(
            {
                "seed": int(run_seed),
                "standard_gnn": {
                    "success": bool(gnn_run["info"].get("success", False)),
                    "completed_tasks": int(gnn_run["info"].get("completed_tasks", 0)),
                    "collisions": int(gnn_run["info"].get("total_collisions", 0)),
                    "collision_stop": bool(gnn_run["info"].get("collision_terminated", False)),
                    "deadline_satisfaction": float(gnn_run["info"].get("deadline_satisfaction", 0.0)),
                },
                "hrsga": {
                    "success": bool(hrsga_run["info"].get("success", False)),
                    "completed_tasks": int(hrsga_run["info"].get("completed_tasks", 0)),
                    "collisions": int(hrsga_run["info"].get("total_collisions", 0)),
                    "collision_stop": bool(hrsga_run["info"].get("collision_terminated", False)),
                    "deadline_satisfaction": float(hrsga_run["info"].get("deadline_satisfaction", 0.0)),
                },
            }
        )
        comparison_runs.append(
            {
                "seed": int(run_seed),
                "standard_gnn_run": gnn_run,
                "hrsga_run": hrsga_run,
            }
        )
        print(
            f"Seed {run_seed}: "
            f"GNN(success={bool(gnn_run['info'].get('success', False))}, completed={int(gnn_run['info'].get('completed_tasks', 0))}, collisions={int(gnn_run['info'].get('total_collisions', 0))}, collision_stop={bool(gnn_run['info'].get('collision_terminated', False))}) | "
            f"HRSGA(success={bool(hrsga_run['info'].get('success', False))}, completed={int(hrsga_run['info'].get('completed_tasks', 0))}, collisions={int(hrsga_run['info'].get('total_collisions', 0))}, collision_stop={bool(hrsga_run['info'].get('collision_terminated', False))})"
        )
        for row_idx, (title, run) in enumerate((("Standard GNN", gnn_run), ("HRSGA", hrsga_run))):
            ax = axes[row_idx, test_idx]
            snapshot = run["snapshot"]
            for obstacle in snapshot["obstacles"]:
                patch = plt.Circle(obstacle["center"], obstacle["radius"], color="gray", alpha=0.25)
                ax.add_patch(patch)
            for ball_idx, traj in enumerate(run["trajectories"]):
                color = BALL_COLORS[ball_idx % len(BALL_COLORS)]
                is_idle = _agent_is_idle(snapshot["agents"][ball_idx])
                alpha = 0.35 if is_idle else 1.0
                linestyle = "--" if is_idle else "-"
                ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=2.0, linestyle=linestyle, alpha=alpha)
                ax.scatter(traj[0, 0], traj[0, 1], color=color, marker="o", s=35)
                ax.scatter(traj[-1, 0], traj[-1, 1], color=color, marker="s", s=35, alpha=alpha)
                if is_idle:
                    ax.text(traj[-1, 0] + 0.08, traj[-1, 1] - 0.12, "idle", color=color, fontsize=8)
            for task_idx, task in enumerate(snapshot["tasks"]):
                _plot_task(ax, task_idx, task, task_label_mode)
            ax.set_title(f"{title} Run {test_idx + 1}")
            ax.set_xlim(-5.1, 5.1)
            ax.set_ylim(-5.1, 5.1)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.grid(True)
    fig.tight_layout()
    _ensure_parent_dir(plot_path)
    fig.savefig(plot_path)
    if show_plot:
        plt.show()
    plt.close(fig)
    if gif_path:
        save_comparison_gif(comparisons=comparison_runs, gif_path=gif_path, fps=gif_fps, task_label_mode=task_label_mode)
    save_json(
        summary_path,
        {
            "layout_mode": layout_mode,
            "num_tests": int(num_tests),
            "seed_start": int(seed),
            "task_rules": _task_rule_config(reference_snapshot or {"tasks": []}, enforce_visit_order=enforce_visit_order),
            "hrsga_model_path": hrsga_model_path,
            "gnn_model_path": gnn_model_path,
            "plot_path": plot_path,
            "task_label_mode": task_label_mode,
            "task_metadata": _task_metadata(reference_snapshot or {"tasks": []}),
            "runs": comparison_summaries,
        },
    )
    print(f"[SAVE] plot={plot_path}")
    print(f"[SAVE] summary={summary_path}")


def _draw_run_panel(ax, title, run, frame_idx, task_label_mode):
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
        ax.plot(path[:, 0], path[:, 1], color=color, linewidth=2.0, linestyle=linestyle, alpha=alpha)
        ax.scatter(traj[0, 0], traj[0, 1], color=color, marker="o", s=35)
        ax.scatter(path[-1, 0], path[-1, 1], color=color, marker="s", s=35, alpha=alpha)
        if is_idle:
            ax.text(path[-1, 0] + 0.08, path[-1, 1] - 0.12, "idle", color=color, fontsize=8)
    for task_idx, task in enumerate(snapshot["tasks"]):
        _plot_task(ax, task_idx, task, task_label_mode)
    ax.set_title(f"{title} | Step {snapshot['step']}")
    ax.set_xlim(-5.1, 5.1)
    ax.set_ylim(-5.1, 5.1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)


def save_comparison_gif(comparisons, gif_path, fps=6, task_label_mode="ranked"):
    if not comparisons:
        raise ValueError("No comparison runs available for GIF export.")
    _ensure_parent_dir(gif_path)
    num_tests = len(comparisons)
    fig, axes = plt.subplots(2, num_tests, figsize=(6 * num_tests, 12), squeeze=False)
    max_frames = max(
        max(len(item["standard_gnn_run"]["snapshots"]), len(item["hrsga_run"]["snapshots"]))
        for item in comparisons
    )

    def update(frame_idx):
        for test_idx, item in enumerate(comparisons):
            _draw_run_panel(axes[0, test_idx], f"Standard GNN Run {test_idx + 1}", item["standard_gnn_run"], frame_idx, task_label_mode)
            _draw_run_panel(axes[1, test_idx], f"HRSGA Run {test_idx + 1}", item["hrsga_run"], frame_idx, task_label_mode)
        fig.suptitle("Standard GNN vs HRSGA Rollout Animation")
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
    parser = argparse.ArgumentParser(description="Compare trained Standard-GNN and HRSGA trajectories in the same environment")
    parser.add_argument("--hrsga_model_path", type=str, default=os.path.join(ROOT_DIR, "models", "hrsga_ball_best.pt"))
    parser.add_argument("--gnn_model_path", type=str, default=os.path.join(ROOT_DIR, "models", "standard_gnn_ball_best.pt"))
    parser.add_argument("--num_tests", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--plot_path", type=str, default=None)
    parser.add_argument("--gif_path", type=str, default=None)
    parser.add_argument("--gif_fps", type=int, default=6)
    parser.add_argument("--summary_path", type=str, default=None)
    parser.add_argument("--show_plot", action="store_true")
    parser.add_argument("--seed", type=int, default=5300)
    parser.add_argument("--num_balls", type=int, default=None)
    parser.add_argument("--num_tasks", type=int, default=None)
    parser.add_argument("--layout_mode", type=str, choices=["representative", "structured", "random"], default="representative")
    parser.add_argument("--task_label_mode", type=str, choices=["ranked", "index", "none"], default="ranked")
    parser.add_argument("--enforce_visit_order", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fixed_layout", action="store_true", help="Use the original structured start/task layout")
    parser.add_argument("--random_layout", action="store_true", help="Use fully random start/task generation")
    args = parser.parse_args()
    layout_mode = "structured" if args.fixed_layout else "random" if args.random_layout else args.layout_mode
    compare_models(
        hrsga_model_path=args.hrsga_model_path,
        gnn_model_path=args.gnn_model_path,
        num_tests=args.num_tests,
        max_steps=args.max_steps,
        plot_path=args.plot_path,
        gif_path=args.gif_path,
        gif_fps=args.gif_fps,
        summary_path=args.summary_path,
        show_plot=args.show_plot,
        seed=args.seed,
        layout_mode=layout_mode,
        num_balls=args.num_balls,
        num_tasks=args.num_tasks,
        task_label_mode=args.task_label_mode,
        enforce_visit_order=args.enforce_visit_order,
    )


if __name__ == "__main__":
    main()