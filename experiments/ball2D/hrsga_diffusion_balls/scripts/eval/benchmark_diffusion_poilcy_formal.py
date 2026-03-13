import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from benchmark_policy_ball import aggregate_runs, load_agent, run_episode, save_json


def main():
    parser = argparse.ArgumentParser(description="Benchmark a diffusion poilcy checkpoint on the formal 4r8t/2r8t/2r4t settings")
    parser.add_argument("--model_path", type=str, default=os.path.join(ROOT_DIR, "models", "hrsga_diffusion_poilcy_best.pt"))
    parser.add_argument("--num_tests", type=int, default=20)
    parser.add_argument("--seed", type=int, default=5300)
    parser.add_argument("--layout_mode", type=str, choices=["representative", "structured", "random"], default="representative")
    parser.add_argument("--strict_collision_stop", action="store_true")
    parser.add_argument("--summary_path", type=str, default=os.path.join(ROOT_DIR, "runs", "benchmarks", "diffusion_poilcy_formal.json"))
    args = parser.parse_args()

    agent = load_agent("diffusion_poilcy", args.model_path)
    settings = [
        {"name": "4r8t", "num_balls": 4, "num_tasks": 8},
        {"name": "2r8t", "num_balls": 2, "num_tasks": 8},
        {"name": "2r4t", "num_balls": 2, "num_tasks": 4},
    ]

    payload = {
        "model_type": "diffusion_poilcy",
        "model_path": args.model_path,
        "layout_mode": args.layout_mode,
        "strict_collision_stop": bool(args.strict_collision_stop),
        "num_tests": int(args.num_tests),
        "seed_start": int(args.seed),
        "results": [],
    }

    for setting in settings:
        print(
            f"Benchmarking diffusion_poilcy on {setting['name']} for {args.num_tests} runs"
            f"{' with collision-stop' if args.strict_collision_stop else ''}..."
        )
        runs = []
        for test_idx in range(args.num_tests):
            run = run_episode(
                model_type="diffusion_poilcy",
                agent=agent,
                seed=args.seed + test_idx,
                max_steps=None,
                layout_mode=args.layout_mode,
                strict_collision_stop=args.strict_collision_stop,
                num_balls=setting["num_balls"],
                num_tasks=setting["num_tasks"],
            )
            runs.append(run)
            print(
                f"Seed {run['seed']}: success={run['success']} completed={run['completed_tasks']} collisions={run['collisions']} "
                f"deadline={run['deadline_satisfaction']:.3f} steps={run['steps']} avg_infer_ms={run['avg_inference_ms']:.3f}"
            )
        summary = aggregate_runs("diffusion_poilcy", args.model_path, args.layout_mode, args.strict_collision_stop, runs)
        summary["setting"] = setting["name"]
        summary["num_balls"] = int(setting["num_balls"])
        summary["num_tasks"] = int(setting["num_tasks"])
        payload["results"].append({"setting": setting, "summary": summary, "runs": runs})

    save_json(args.summary_path, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[SAVE] summary={args.summary_path}")


if __name__ == "__main__":
    main()