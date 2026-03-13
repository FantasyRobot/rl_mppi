import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
ROOT_DIR = os.path.dirname(SCRIPTS_DIR)
for candidate in (ROOT_DIR, os.path.join(SCRIPTS_DIR, "eval")):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from benchmark_policy_ball import aggregate_runs, load_agent, run_episode


def save_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def build_default_output_paths(model_names, layout_mode):
    root = os.path.join(ROOT_DIR, "runs", "benchmarks")
    tag = f"compare_{'_'.join(model_names)}_{layout_mode}"
    return {
        "summary": os.path.join(root, f"{tag}.json"),
        "plot": os.path.join(root, f"{tag}.png"),
    }


def compare_models(model_specs, num_tests, seed, max_steps, layout_mode, strict_collision_stop, num_balls, num_tasks):
    agents = [
        {
            "label": spec["label"],
            "model_type": spec["model_type"],
            "model_path": spec["model_path"],
            "agent": load_agent(spec["model_type"], spec["model_path"]),
            "runs": [],
        }
        for spec in model_specs
    ]

    for test_idx in range(num_tests):
        run_seed = seed + test_idx
        print_parts = [f"Seed {run_seed}:"]
        for item in agents:
            run = run_episode(
                item["model_type"],
                item["agent"],
                run_seed,
                max_steps=max_steps,
                layout_mode=layout_mode,
                strict_collision_stop=strict_collision_stop,
                num_balls=num_balls,
                num_tasks=num_tasks,
            )
            item["runs"].append(run)
            print_parts.append(
                f"{item['label']}(success={run['success']}, completed={run['completed_tasks']}, collisions={run['collisions']})"
            )
        print(" | ".join(print_parts))

    for item in agents:
        item["summary"] = aggregate_runs(item["model_type"], item["model_path"], layout_mode, strict_collision_stop, item["runs"])
    return agents


def render_comparison_plot(model_results, plot_path):
    metrics = [
        ("success_rate", "Success Rate", False),
        ("avg_deadline_satisfaction", "Deadline Satisfaction", False),
        ("avg_collisions", "Avg Collisions", True),
        ("avg_steps", "Avg Steps", True),
        ("avg_inference_ms", "Avg Inference (ms)", True),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.6 * len(metrics), 4.8), squeeze=False)
    labels = [item["label"] for item in model_results]
    colors = ["tab:blue", "tab:orange", "tab:green"][: len(model_results)]
    for axis, (key, title, lower_is_better) in zip(axes[0], metrics):
        values = [item["summary"][key] for item in model_results]
        axis.bar(labels, values, color=colors, alpha=0.85)
        axis.set_title(title)
        axis.grid(True, axis="y", alpha=0.25)
        for index, value in enumerate(values):
            axis.text(index, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
        axis.tick_params(axis="x", rotation=15)
        if lower_is_better:
            axis.set_ylabel("lower is better")
        else:
            axis.set_ylabel("higher is better")
    fig.tight_layout()
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig.savefig(plot_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare trained policy checkpoints with aggregate metrics")
    parser.add_argument("--left_model_type", type=str, choices=["hrsga", "standard_gnn", "td3", "diffusion_poilcy"], default="hrsga")
    parser.add_argument("--left_model_path", type=str, default=os.path.join(ROOT_DIR, "models", "hrsga_ball_best.pt"))
    parser.add_argument("--left_label", type=str, default="HRSGA")
    parser.add_argument("--right_model_type", type=str, choices=["hrsga", "standard_gnn", "td3", "diffusion_poilcy"], default="standard_gnn")
    parser.add_argument("--right_model_path", type=str, default=os.path.join(ROOT_DIR, "models", "standard_gnn_ball_best.pt"))
    parser.add_argument("--right_label", type=str, default="Standard GNN")
    parser.add_argument("--third_model_type", type=str, choices=["hrsga", "standard_gnn", "td3", "diffusion_poilcy"], default=None)
    parser.add_argument("--third_model_path", type=str, default=None)
    parser.add_argument("--third_label", type=str, default="Diffusion Poilcy")
    parser.add_argument("--num_tests", type=int, default=20)
    parser.add_argument("--seed", type=int, default=5300)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--num_balls", type=int, default=None)
    parser.add_argument("--num_tasks", type=int, default=None)
    parser.add_argument("--layout_mode", type=str, choices=["representative", "structured", "random"], default="representative")
    parser.add_argument("--strict_collision_stop", action="store_true")
    parser.add_argument("--summary_path", type=str, default=None)
    parser.add_argument("--plot_path", type=str, default=None)
    args = parser.parse_args()

    model_specs = [
        {"label": args.left_label, "model_type": args.left_model_type, "model_path": args.left_model_path},
        {"label": args.right_label, "model_type": args.right_model_type, "model_path": args.right_model_path},
    ]
    if args.third_model_type and args.third_model_path:
        model_specs.append({"label": args.third_label, "model_type": args.third_model_type, "model_path": args.third_model_path})

    output_paths = build_default_output_paths([spec["model_type"] for spec in model_specs], args.layout_mode)
    summary_path = args.summary_path or output_paths["summary"]
    plot_path = args.plot_path or output_paths["plot"]

    model_results = compare_models(
        model_specs=model_specs,
        num_tests=args.num_tests,
        seed=args.seed,
        max_steps=args.max_steps,
        layout_mode=args.layout_mode,
        strict_collision_stop=args.strict_collision_stop,
        num_balls=args.num_balls,
        num_tasks=args.num_tasks,
    )

    payload = {
        "models": [
            {
                "label": item["label"],
                "model_type": item["model_type"],
                "model_path": item["model_path"],
                "summary": item["summary"],
                "runs": item["runs"],
            }
            for item in model_results
        ]
    }
    save_json(summary_path, payload)
    render_comparison_plot(model_results, plot_path)
    print(json.dumps({item["label"]: item["summary"] for item in model_results}, indent=2, ensure_ascii=False))
    print(f"[SAVE] summary={summary_path}")
    print(f"[SAVE] plot={plot_path}")


if __name__ == "__main__":
    main()