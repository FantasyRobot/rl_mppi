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

from benchmark_policy import aggregate_runs, load_agent, run_episode


def save_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def build_default_output_paths(left_name, right_name, layout_mode):
    root = os.path.join(ROOT_DIR, "runs", "benchmarks")
    tag = f"compare_{left_name}_vs_{right_name}_{layout_mode}"
    return {
        "summary": os.path.join(root, f"{tag}.json"),
        "plot": os.path.join(root, f"{tag}.png"),
    }


def compare_models(
    left_model_type,
    left_model_path,
    right_model_type,
    right_model_path,
    num_tests,
    seed,
    max_steps,
    layout_mode,
    strict_collision_stop,
    xml_path,
):
    left_agent = load_agent(left_model_type, left_model_path)
    right_agent = load_agent(right_model_type, right_model_path)
    left_runs = []
    right_runs = []

    for test_idx in range(num_tests):
        run_seed = seed + test_idx
        left_run = run_episode(left_model_type, left_agent, run_seed, max_steps=max_steps, layout_mode=layout_mode, strict_collision_stop=strict_collision_stop, xml_path=xml_path)
        right_run = run_episode(right_model_type, right_agent, run_seed, max_steps=max_steps, layout_mode=layout_mode, strict_collision_stop=strict_collision_stop, xml_path=xml_path)
        left_runs.append(left_run)
        right_runs.append(right_run)
        print(
            f"Seed {run_seed}: "
            f"{left_model_type}(success={left_run['success']}, completed={left_run['completed_tasks']}, collisions={left_run['collisions']}) | "
            f"{right_model_type}(success={right_run['success']}, completed={right_run['completed_tasks']}, collisions={right_run['collisions']})"
        )

    left_summary = aggregate_runs(left_model_type, left_model_path, layout_mode, strict_collision_stop, left_runs)
    right_summary = aggregate_runs(right_model_type, right_model_path, layout_mode, strict_collision_stop, right_runs)
    return left_runs, right_runs, left_summary, right_summary


def render_comparison_plot(left_label, right_label, left_summary, right_summary, plot_path):
    metrics = [
        ("success_rate", "Success Rate", False),
        ("avg_deadline_satisfaction", "Deadline Satisfaction", False),
        ("avg_collisions", "Avg Collisions", True),
        ("avg_steps", "Avg Steps", True),
        ("avg_inference_ms", "Avg Inference (ms)", True),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.6 * len(metrics), 4.8), squeeze=False)
    for axis, (key, title, lower_is_better) in zip(axes[0], metrics):
        values = [left_summary[key], right_summary[key]]
        colors = ["tab:blue", "tab:orange"]
        axis.bar([left_label, right_label], values, color=colors, alpha=0.85)
        axis.set_title(title)
        axis.grid(True, axis="y", alpha=0.25)
        for index, value in enumerate(values):
            axis.text(index, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
        if lower_is_better:
            axis.set_ylabel("lower is better")
        else:
            axis.set_ylabel("higher is better")
    fig.tight_layout()
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig.savefig(plot_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare two trained policy checkpoints with aggregate metrics")
    parser.add_argument("--left_model_type", type=str, choices=["hrsga", "standard_gnn", "td3"], default="hrsga")
    parser.add_argument("--left_model_path", type=str, default=os.path.join(ROOT_DIR, "models", "hrsga_best.pt"))
    parser.add_argument("--left_label", type=str, default="HRSGA")
    parser.add_argument("--right_model_type", type=str, choices=["hrsga", "standard_gnn", "td3"], default="standard_gnn")
    parser.add_argument("--right_model_path", type=str, default=os.path.join(ROOT_DIR, "models", "standard_gnn_best.pt"))
    parser.add_argument("--right_label", type=str, default="Standard GNN")
    parser.add_argument("--num_tests", type=int, default=20)
    parser.add_argument("--seed", type=int, default=5300)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--layout_mode", type=str, choices=["representative", "structured", "random"], default="representative")
    parser.add_argument("--strict_collision_stop", action="store_true")
    parser.add_argument("--xml_path", type=str, default=None)
    parser.add_argument("--summary_path", type=str, default=None)
    parser.add_argument("--plot_path", type=str, default=None)
    args = parser.parse_args()

    output_paths = build_default_output_paths(args.left_model_type, args.right_model_type, args.layout_mode)
    summary_path = args.summary_path or output_paths["summary"]
    plot_path = args.plot_path or output_paths["plot"]

    left_runs, right_runs, left_summary, right_summary = compare_models(
        left_model_type=args.left_model_type,
        left_model_path=args.left_model_path,
        right_model_type=args.right_model_type,
        right_model_path=args.right_model_path,
        num_tests=args.num_tests,
        seed=args.seed,
        max_steps=args.max_steps,
        layout_mode=args.layout_mode,
        strict_collision_stop=args.strict_collision_stop,
        xml_path=args.xml_path,
    )

    payload = {
        "left": {"label": args.left_label, "summary": left_summary, "runs": left_runs},
        "right": {"label": args.right_label, "summary": right_summary, "runs": right_runs},
    }
    save_json(summary_path, payload)
    render_comparison_plot(args.left_label, args.right_label, left_summary, right_summary, plot_path)
    print(json.dumps({"left": left_summary, "right": right_summary}, indent=2, ensure_ascii=False))
    print(f"[SAVE] summary={summary_path}")
    print(f"[SAVE] plot={plot_path}")


if __name__ == "__main__":
    main()