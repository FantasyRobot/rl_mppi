import argparse
import json
import os
import sys

import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
ROOT_DIR = os.path.dirname(SCRIPTS_DIR)
for candidate in (ROOT_DIR, os.path.join(SCRIPTS_DIR, "eval")):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from benchmark_policy_ball import aggregate_runs, run_episode
from hrsga_ball_model import load_agent_from_checkpoint


def evaluate_ablation(model_path, label, num_tests, seed, max_steps, layout_mode, strict_collision_stop, num_balls, num_tasks):
    agent = load_agent_from_checkpoint(model_path)
    runs = []
    for test_idx in range(num_tests):
        runs.append(run_episode("hrsga", agent, seed + test_idx, max_steps=max_steps, layout_mode=layout_mode, strict_collision_stop=strict_collision_stop, num_balls=num_balls, num_tasks=num_tasks))
    summary = aggregate_runs("hrsga", model_path, layout_mode, strict_collision_stop, runs)
    return {"label": label, "summary": summary, "runs": runs}


def save_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def render_plot(results, plot_path):
    metrics = [
        ("success_rate", "Success Rate"),
        ("avg_deadline_satisfaction", "Deadline Satisfaction"),
        ("avg_collisions", "Avg Collisions"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.6 * len(metrics), 4.5), squeeze=False)
    labels = [result["label"] for result in results]
    for axis, (key, title) in zip(axes[0], metrics):
        values = [result["summary"][key] for result in results]
        axis.bar(labels, values, color=["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"][: len(values)])
        axis.set_title(title)
        axis.grid(True, axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig.savefig(plot_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare trained HRSGA ablation checkpoints")
    parser.add_argument("--labels", type=str, nargs="+", required=True, help="Display labels for the ablation checkpoints")
    parser.add_argument("--model_paths", type=str, nargs="+", required=True, help="Checkpoint paths matching the labels")
    parser.add_argument("--num_tests", type=int, default=20)
    parser.add_argument("--seed", type=int, default=5300)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--num_balls", type=int, default=None)
    parser.add_argument("--num_tasks", type=int, default=None)
    parser.add_argument("--layout_mode", type=str, choices=["representative", "structured", "random"], default="representative")
    parser.add_argument("--strict_collision_stop", action="store_true")
    parser.add_argument("--summary_path", type=str, default=os.path.join(ROOT_DIR, "runs", "benchmarks", "hrsga_ablations.json"))
    parser.add_argument("--plot_path", type=str, default=os.path.join(ROOT_DIR, "runs", "benchmarks", "hrsga_ablations.png"))
    args = parser.parse_args()

    if len(args.labels) != len(args.model_paths):
        raise ValueError("labels and model_paths must have the same length")

    results = []
    for label, model_path in zip(args.labels, args.model_paths):
        print(f"Evaluating {label} from {model_path}...")
        results.append(
            evaluate_ablation(
                model_path=model_path,
                label=label,
                num_tests=args.num_tests,
                seed=args.seed,
                max_steps=args.max_steps,
                layout_mode=args.layout_mode,
                strict_collision_stop=args.strict_collision_stop,
                num_balls=args.num_balls,
                num_tasks=args.num_tasks,
            )
        )

    save_json(args.summary_path, {"results": results})
    render_plot(results, args.plot_path)
    print(f"[SAVE] summary={args.summary_path}")
    print(f"[SAVE] plot={args.plot_path}")


if __name__ == "__main__":
    main()