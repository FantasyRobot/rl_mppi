import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
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


def save_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def build_default_output_paths(layout_mode):
    root = os.path.join(ROOT_DIR, "runs", "benchmarks")
    tag = f"compare_trained_gnn_hrsga_{layout_mode}"
    return {
        "plot": os.path.join(root, f"{tag}.png"),
        "summary": os.path.join(root, f"{tag}.json"),
    }


def rollout(agent, seed, max_steps=None, layout_mode="representative"):
    max_steps = agent.max_steps if max_steps is None else max_steps
    env = HRSGABallEnvironment(num_balls=agent.num_balls, max_steps=max_steps)
    snapshot = env.reset(seed=seed, layout_mode=layout_mode)
    trajectories = [[agent_state["pos"]] for agent_state in snapshot["agents"]]
    info = {}
    collision_terminated = False
    while True:
        action = agent.select_action(snapshot)
        snapshot, done, info = env.step(action)
        for ball_idx, agent_state in enumerate(snapshot["agents"]):
            trajectories[ball_idx].append(agent_state["pos"])
        if int(info.get("collisions", 0)) > 0:
            done = True
            collision_terminated = True
        if done:
            break
    info = dict(info)
    info["collision_terminated"] = collision_terminated
    return {"trajectories": [np.asarray(traj, dtype=np.float32) for traj in trajectories], "snapshot": snapshot, "info": info}


def compare_models(
    hrsga_model_path,
    gnn_model_path,
    num_tests=5,
    max_steps=None,
    plot_path=None,
    summary_path=None,
    show_plot=False,
    seed=5300,
    layout_mode="representative",
):
    output_paths = build_default_output_paths(layout_mode)
    plot_path = plot_path or output_paths["plot"]
    summary_path = summary_path or output_paths["summary"]
    hrsga_agent = load_hrsga_agent(hrsga_model_path)
    gnn_agent = load_gnn_agent(gnn_model_path)
    fig, axes = plt.subplots(2, num_tests, figsize=(6 * num_tests, 12), squeeze=False)
    comparisons = []
    print(f"Comparing trained Standard-GNN and HRSGA policies with {layout_mode} start/task layouts...")
    for test_idx in range(num_tests):
        run_seed = seed + test_idx
        gnn_run = rollout(gnn_agent, seed=run_seed, max_steps=max_steps, layout_mode=layout_mode)
        hrsga_run = rollout(hrsga_agent, seed=run_seed, max_steps=max_steps, layout_mode=layout_mode)
        comparisons.append(
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
                ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=2.0)
                ax.scatter(traj[0, 0], traj[0, 1], color=color, marker="o", s=35)
                ax.scatter(traj[-1, 0], traj[-1, 1], color=color, marker="s", s=35)
            for task_idx, task in enumerate(snapshot["tasks"]):
                color = BALL_COLORS[task_idx % len(BALL_COLORS)]
                marker = "*" if task["completed"] else "x"
                ax.scatter(task["pos"][0], task["pos"][1], color=color, marker=marker, s=140, alpha=0.95)
            ax.set_title(f"{title} Run {test_idx + 1}")
            ax.set_xlim(-5.1, 5.1)
            ax.set_ylim(-5.1, 5.1)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.grid(True)
    fig.tight_layout()
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig.savefig(plot_path)
    if show_plot:
        plt.show()
    plt.close(fig)
    save_json(
        summary_path,
        {
            "layout_mode": layout_mode,
            "num_tests": int(num_tests),
            "seed_start": int(seed),
            "hrsga_model_path": hrsga_model_path,
            "gnn_model_path": gnn_model_path,
            "plot_path": plot_path,
            "runs": comparisons,
        },
    )
    print(f"[SAVE] plot={plot_path}")
    print(f"[SAVE] summary={summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare trained Standard-GNN and HRSGA trajectories in the same environment")
    parser.add_argument("--hrsga_model_path", type=str, default=os.path.join(ROOT_DIR, "models", "hrsga_ball_best.pt"))
    parser.add_argument("--gnn_model_path", type=str, default=os.path.join(ROOT_DIR, "models", "standard_gnn_ball_best.pt"))
    parser.add_argument("--num_tests", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--plot_path", type=str, default=None)
    parser.add_argument("--summary_path", type=str, default=None)
    parser.add_argument("--show_plot", action="store_true")
    parser.add_argument("--seed", type=int, default=5300)
    parser.add_argument("--layout_mode", type=str, choices=["representative", "structured", "random"], default="representative")
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
        summary_path=args.summary_path,
        show_plot=args.show_plot,
        seed=args.seed,
        layout_mode=layout_mode,
    )


if __name__ == "__main__":
    main()