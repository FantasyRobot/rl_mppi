import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
ROOT_DIR = os.path.dirname(SCRIPTS_DIR)
for candidate in (ROOT_DIR, SCRIPT_DIR):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from train_hrsga_ball import train


ABLATION_PRESETS = {
    "no_temporal": {
        "disable_temporal_bias": True,
    },
    "dense": {
        "topk_robot": 16,
        "topk_task": 16,
        "topk_obstacle": 8,
    },
    "unified_relation": {
        "shared_relation_attention": True,
    },
    "no_geometric": {
        "disable_geometric_bias": True,
    },
}


def _explicit_cli_keys(argv):
    keys = set()
    for token in argv:
        if not token.startswith("--"):
            continue
        name = token[2:]
        if "=" in name:
            name = name.split("=", 1)[0]
        keys.add(name.replace("-", "_"))
    return keys


def _apply_config_overrides(args, parser, argv):
    if not args.config_path:
        return args
    with open(args.config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    if isinstance(config, dict) and isinstance(config.get("config"), dict):
        config = config["config"]
    valid_dests = {action.dest for action in parser._actions}
    explicit_keys = _explicit_cli_keys(argv)
    for key, value in config.items():
        if key not in valid_dests or key in explicit_keys or key == "config_path":
            continue
        setattr(args, key, value)
    return args


def build_train_kwargs(args):
    kwargs = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "train_episodes": args.train_episodes,
        "val_episodes": args.val_episodes,
        "rollout_episodes": args.rollout_episodes,
        "eval_interval": args.eval_interval,
        "save_interval": args.save_interval,
        "num_balls": args.num_balls,
        "num_tasks": args.num_tasks,
        "max_steps": args.max_steps,
        "hidden_dim": args.hidden_dim,
        "num_heads": args.num_heads,
        "topk_robot": args.topk_robot,
        "topk_task": args.topk_task,
        "topk_obstacle": args.topk_obstacle,
        "disable_temporal_bias": args.disable_temporal_bias,
        "disable_geometric_bias": args.disable_geometric_bias,
        "shared_relation_attention": args.shared_relation_attention,
        "use_dense_residual": args.use_dense_residual,
        "dataset_layout_pattern": args.dataset_layout_pattern,
        "expert_max_collisions": args.expert_max_collisions,
        "eval_layout_mode": args.eval_layout_mode,
        "eval_strict_collision_stop": args.eval_strict_collision_stop,
        "resume_path": args.resume_path,
        "seed": args.seed,
        "run_name": args.run_name,
        "output_root": args.output_root,
    }
    preset = ABLATION_PRESETS[args.ablation_name]
    kwargs.update(preset)
    if args.run_name is None:
        kwargs["run_name"] = f"hrsga_{args.ablation_name}"
    return kwargs


def main():
    parser = argparse.ArgumentParser(description="Train an HRSGA ablation variant")
    parser.add_argument("--ablation_name", type=str, choices=sorted(ABLATION_PRESETS.keys()), required=True)
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--train_episodes", type=int, default=160)
    parser.add_argument("--val_episodes", type=int, default=40)
    parser.add_argument("--rollout_episodes", type=int, default=20)
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--num_balls", type=int, default=4)
    parser.add_argument("--num_tasks", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=180)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--topk_robot", type=int, default=2)
    parser.add_argument("--topk_task", type=int, default=2)
    parser.add_argument("--topk_obstacle", type=int, default=1)
    parser.add_argument("--disable_temporal_bias", action="store_true")
    parser.add_argument("--disable_geometric_bias", action="store_true")
    parser.add_argument("--shared_relation_attention", action="store_true")
    parser.add_argument("--use_dense_residual", action="store_true")
    parser.add_argument("--dataset_layout_pattern", type=str, default="structured,structured,random,representative")
    parser.add_argument("--expert_max_collisions", type=int, default=None)
    parser.add_argument("--eval_layout_mode", type=str, default="representative", choices=["structured", "random", "representative"])
    parser.add_argument("--eval_strict_collision_stop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=4300)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=os.path.join(ROOT_DIR, "runs"))
    args = parser.parse_args()
    args = _apply_config_overrides(args, parser, sys.argv[1:])
    train(**build_train_kwargs(args))


if __name__ == "__main__":
    main()