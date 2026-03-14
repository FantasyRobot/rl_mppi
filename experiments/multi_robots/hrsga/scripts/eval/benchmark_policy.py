import argparse
import json
import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
ROOT_DIR = os.path.dirname(SCRIPTS_DIR)
for candidate in (ROOT_DIR, os.path.join(SCRIPTS_DIR, "train")):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from env_hrsga import HRSGAEnvironment
from hrsga_model import load_agent_from_checkpoint as load_hrsga_agent
from standard_gnn_model import load_agent_from_checkpoint as load_gnn_agent
from train_td3_hrsga import load_agent_from_checkpoint as load_td3_agent


def task_rule_config_from_snapshot(snapshot):
    dwell_values = [int(task.get("dwell_steps", 1)) for task in snapshot["tasks"]]
    return {
        "visit_order_mode": "ranked",
        "min_dwell_steps": min(dwell_values) if dwell_values else 0,
        "max_dwell_steps": max(dwell_values) if dwell_values else 0,
    }

def _resolve_default_model_path(model_type):
    models_dir = os.path.join(ROOT_DIR, "models")
    default_names = {
        "hrsga": "hrsga_best.pt",
        "standard_gnn": "standard_gnn_best.pt",
        "td3": "td3_hrsga_best.pt",
    }
    return os.path.join(models_dir, default_names[model_type])


def load_agent(model_type, model_path):
    if model_type == "hrsga":
        return load_hrsga_agent(model_path)
    if model_type == "standard_gnn":
        return load_gnn_agent(model_path)
    if model_type == "td3":
        return load_td3_agent(model_path)
    raise ValueError(f"Unsupported model_type: {model_type}")


def select_action(model_type, agent, snapshot):
    if model_type == "td3":
        return agent.select_action(snapshot, noise_sigma=0.0)
    return agent.select_action(snapshot)


def run_episode(model_type, agent, seed, max_steps=None, layout_mode="representative", strict_collision_stop=False, xml_path=None):
    max_steps = agent.max_steps if max_steps is None else max_steps
    env = HRSGAEnvironment(
        max_steps=max_steps,
        xml_path=xml_path or getattr(agent, "xml_path", None),
    )
    snapshot = env.reset(seed=seed, layout_mode=layout_mode)
    inference_times_ms = []
    info = {}
    collision_terminated = False

    while True:
        start = time.perf_counter()
        action = select_action(model_type, agent, snapshot)
        inference_times_ms.append((time.perf_counter() - start) * 1000.0)
        snapshot, done, info = env.step(action)
        if strict_collision_stop and int(info.get("collisions", 0)) > 0:
            done = True
            collision_terminated = True
        if done:
            break

    completed_tasks = int(info.get("completed_tasks", 0))
    task_count = max(1, len(snapshot["tasks"]))
    return {
        "seed": int(seed),
        "success": bool(info.get("success", False)),
        "completed_tasks": completed_tasks,
        "completed_fraction": float(completed_tasks / task_count),
        "deadline_satisfaction": float(info.get("deadline_satisfaction", 0.0)),
        "collisions": int(info.get("total_collisions", 0)),
        "mean_task_distance": float(info.get("mean_task_distance", 0.0)),
        "min_pair_distance": float(info.get("min_pair_distance", 0.0)),
        "missed_deadlines": int(info.get("missed_deadlines", 0)),
        "steps": int(snapshot["step"]),
        "collision_terminated": collision_terminated,
        "avg_inference_ms": float(np.mean(inference_times_ms)) if inference_times_ms else 0.0,
        "p95_inference_ms": float(np.percentile(inference_times_ms, 95)) if inference_times_ms else 0.0,
        "task_rules": task_rule_config_from_snapshot(snapshot),
    }


def aggregate_runs(model_type, model_path, layout_mode, strict_collision_stop, runs):
    if not runs:
        raise ValueError("No evaluation runs were collected.")

    def mean(key):
        return float(np.mean([run[key] for run in runs]))

    return {
        "model_type": model_type,
        "model_path": model_path,
        "layout_mode": layout_mode,
        "strict_collision_stop": bool(strict_collision_stop),
        "task_rules": runs[0].get("task_rules", {}),
        "num_tests": int(len(runs)),
        "success_rate": mean("success"),
        "avg_completed_tasks": mean("completed_tasks"),
        "avg_completed_fraction": mean("completed_fraction"),
        "avg_deadline_satisfaction": mean("deadline_satisfaction"),
        "avg_collisions": mean("collisions"),
        "avg_mean_task_distance": mean("mean_task_distance"),
        "avg_min_pair_distance": mean("min_pair_distance"),
        "avg_missed_deadlines": mean("missed_deadlines"),
        "avg_steps": mean("steps"),
        "collision_stop_rate": mean("collision_terminated"),
        "avg_inference_ms": mean("avg_inference_ms"),
        "p95_inference_ms": mean("p95_inference_ms"),
    }


def save_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def save_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_default_output_paths(model_type, layout_mode):
    root = os.path.join(ROOT_DIR, "runs", "benchmarks")
    tag = f"{model_type}_{layout_mode}"
    return {
        "summary": os.path.join(root, f"{tag}_summary.json"),
        "runs": os.path.join(root, f"{tag}_runs.jsonl"),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark a trained policy checkpoint on the multi-robot environment")
    parser.add_argument("--model_type", type=str, choices=["hrsga", "standard_gnn", "td3"], required=True)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--num_tests", type=int, default=20)
    parser.add_argument("--seed", type=int, default=5300)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--layout_mode", type=str, choices=["representative", "structured", "random"], default="representative")
    parser.add_argument("--strict_collision_stop", action="store_true")
    parser.add_argument("--xml_path", type=str, default=None)
    parser.add_argument("--summary_path", type=str, default=None)
    parser.add_argument("--runs_path", type=str, default=None)
    args = parser.parse_args()

    model_path = args.model_path or _resolve_default_model_path(args.model_type)
    agent = load_agent(args.model_type, model_path)
    output_paths = build_default_output_paths(args.model_type, args.layout_mode)
    summary_path = args.summary_path or output_paths["summary"]
    runs_path = args.runs_path or output_paths["runs"]

    runs = []
    print(
        f"Benchmarking {args.model_type} for {args.num_tests} runs on {args.layout_mode} layouts"
        f"{' with collision-stop' if args.strict_collision_stop else ''}..."
    )
    for test_idx in range(args.num_tests):
        run = run_episode(
            model_type=args.model_type,
            agent=agent,
            seed=args.seed + test_idx,
            max_steps=args.max_steps,
            layout_mode=args.layout_mode,
            strict_collision_stop=args.strict_collision_stop,
            xml_path=args.xml_path,
        )
        runs.append(run)
        print(
            f"Seed {run['seed']}: success={run['success']} completed={run['completed_tasks']} collisions={run['collisions']} "
            f"deadline={run['deadline_satisfaction']:.3f} steps={run['steps']} avg_infer_ms={run['avg_inference_ms']:.3f}"
        )

    summary = aggregate_runs(args.model_type, model_path, args.layout_mode, args.strict_collision_stop, runs)
    save_json(summary_path, {"summary": summary, "runs": runs})
    save_jsonl(runs_path, runs)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[SAVE] summary={summary_path}")
    print(f"[SAVE] runs={runs_path}")


if __name__ == "__main__":
    main()