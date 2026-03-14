import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
ROOT_DIR = os.path.dirname(SCRIPTS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

DEFAULT_CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "standard_gnn.json")

from env_hrsga import HRSGAEnvironment
from hrsga_model import build_behavior_dataset, compute_action_loss, iterate_minibatches
from standard_gnn_model import StandardGNNAgent, evaluate_agent


def _save_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _append_jsonl(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _safe_save_checkpoint(agent, path, *, epoch, best_score, optimizer, required=True):
    try:
        agent.save(path, epoch=epoch, best_score=best_score, optimizer=optimizer)
        return True
    except Exception as error:
        if required:
            raise
        print(f"[WARN] checkpoint sync skipped path={path} error={error}")
        return False


def _build_run_paths(output_root, run_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    resolved_run_name = run_name or f"standard_gnn_run_{timestamp}"
    run_dir = os.path.join(output_root, resolved_run_name)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    log_dir = os.path.join(run_dir, "logs")
    return {
        "run_name": resolved_run_name,
        "run_dir": run_dir,
        "checkpoint_dir": checkpoint_dir,
        "log_dir": log_dir,
        "config_path": os.path.join(run_dir, "config.json"),
        "metrics_path": os.path.join(log_dir, "metrics.jsonl"),
        "summary_path": os.path.join(run_dir, "summary.json"),
        "best_path": os.path.join(checkpoint_dir, "standard_gnn_best.pt"),
        "latest_path": os.path.join(checkpoint_dir, "standard_gnn_latest.pt"),
        "canonical_best_path": os.path.join(ROOT_DIR, "models", "standard_gnn_best.pt"),
        "canonical_latest_path": os.path.join(ROOT_DIR, "models", "standard_gnn_latest.pt"),
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


def _task_rule_config(max_steps, xml_path=None, enforce_visit_order=None):
    env = HRSGAEnvironment(max_steps=max_steps, xml_path=xml_path, enforce_visit_order=enforce_visit_order)
    return env.task_rule_config()


def _split_dataset(dataset, train_ratio):
    sample_count = dataset.robot_features.shape[0]
    split_index = max(1, min(sample_count - 1, int(sample_count * train_ratio)))
    train_slice = slice(0, split_index)
    val_slice = slice(split_index, sample_count)
    train_dataset = type(dataset)(*[getattr(dataset, field)[train_slice] for field in dataset.__dataclass_fields__])
    val_dataset = type(dataset)(*[getattr(dataset, field)[val_slice] for field in dataset.__dataclass_fields__])
    return train_dataset, val_dataset


def _dataset_summary(stats):
    if not stats:
        return "no episodes"
    completed = np.mean([item["completed_tasks"] for item in stats])
    deadline = np.mean([item["deadline_satisfaction"] for item in stats])
    collisions = np.mean([item["total_collisions"] for item in stats])
    return f"completed={completed:.2f} deadline_satisfaction={deadline:.3f} collisions={collisions:.2f}"


def _filter_summary(metadata):
    if not metadata:
        return "filter=disabled"
    requested = int(metadata.get("requested_episode_count", 0))
    attempted = int(metadata.get("attempted_episode_count", requested))
    kept = int(metadata.get("kept_episode_count", 0))
    dropped = int(metadata.get("dropped_episode_count", 0))
    threshold = metadata.get("expert_max_collisions")
    if threshold is None:
        return f"filter=disabled requested={requested} attempted={attempted} kept={kept}"
    return (
        f"filter=max_collisions<={threshold} requested={requested} "
        f"attempted={attempted} kept={kept} dropped={dropped}"
    )


def train(
    *,
    epochs=80,
    batch_size=64,
    learning_rate=3e-4,
    weight_decay=1e-5,
    train_episodes=160,
    val_episodes=40,
    rollout_episodes=20,
    eval_interval=5,
    save_interval=10,
    max_steps=280,
    hidden_dim=128,
    edge_hidden_dim=64,
    dataset_layout_pattern="structured,structured,random,representative",
    expert_max_collisions=None,
    eval_layout_mode="representative",
    eval_strict_collision_stop=True,
    enforce_visit_order=None,
    resume_path=None,
    seed=4300,
    run_name=None,
    output_root=None,
    xml_path=None,
):
    output_root = output_root or os.path.join(ROOT_DIR, "runs")
    run_paths = _build_run_paths(output_root, run_name)
    os.makedirs(run_paths["checkpoint_dir"], exist_ok=True)
    os.makedirs(run_paths["log_dir"], exist_ok=True)

    env_summary = HRSGAEnvironment(max_steps=max_steps, xml_path=xml_path, enforce_visit_order=enforce_visit_order)
    num_robots = int(env_summary.num_robots)
    num_tasks = int(env_summary.num_tasks)

    dataset, episode_stats, dataset_metadata = build_behavior_dataset(
        num_episodes=train_episodes + val_episodes,
        num_robots=num_robots,
        num_tasks=num_tasks,
        max_steps=max_steps,
        seed_start=seed,
        layout_pattern=dataset_layout_pattern,
        expert_max_collisions=expert_max_collisions,
        return_metadata=True,
        xml_path=xml_path,
        enforce_visit_order=enforce_visit_order,
    )
    train_dataset, val_dataset = _split_dataset(dataset, train_episodes / max(train_episodes + val_episodes, 1))
    mean_completed_tasks = float(np.mean([item["completed_tasks"] for item in episode_stats])) if episode_stats else 0.0
    mean_completed_fraction = mean_completed_tasks / max(float(num_tasks), 1.0)
    mean_deadline = float(np.mean([item["deadline_satisfaction"] for item in episode_stats])) if episode_stats else 0.0
    mean_collisions = float(np.mean([item["total_collisions"] for item in episode_stats])) if episode_stats else 0.0
    dataset_quality_warnings = []
    if mean_completed_fraction < 0.5:
        dataset_quality_warnings.append(
            f"low expert completion fraction ({mean_completed_fraction:.3f}); imitation targets may be too weak for this XML scene"
        )
    if mean_collisions > 5.0:
        dataset_quality_warnings.append(
            f"high expert collision count ({mean_collisions:.2f}); consider tightening expert_max_collisions or improving the expert policy"
        )
    if mean_deadline < 0.5:
        dataset_quality_warnings.append(
            f"low expert deadline satisfaction ({mean_deadline:.3f}); rollout evaluation is unlikely to improve from pure behavior cloning"
        )

    dataset_summary = {
        "train_samples": int(train_dataset.robot_features.shape[0]),
        "val_samples": int(val_dataset.robot_features.shape[0]),
        "expert_episode_count": int(len(episode_stats)),
        "expert_stats": _dataset_summary(episode_stats),
        "expert_completed_fraction": mean_completed_fraction,
        "expert_deadline_satisfaction": mean_deadline,
        "expert_collisions": mean_collisions,
        "dataset_filter": _filter_summary(dataset_metadata),
        "requested_episode_count": int(dataset_metadata.get("requested_episode_count", len(episode_stats))),
        "attempted_episode_count": int(dataset_metadata.get("attempted_episode_count", len(episode_stats))),
        "dropped_episode_count": int(dataset_metadata.get("dropped_episode_count", 0)),
        "expert_max_collisions": dataset_metadata.get("expert_max_collisions"),
        "dropped_expert_stats": _dataset_summary(dataset_metadata.get("dropped_expert_stats", [])),
        "warnings": dataset_quality_warnings,
    }
    print(
        f"[DATA] train_samples={dataset_summary['train_samples']} val_samples={dataset_summary['val_samples']} "
        f"expert_stats={dataset_summary['expert_stats']} {dataset_summary['dataset_filter']} "
        f"run_dir={run_paths['run_dir']}"
    )
    print(
        f"[DATA] expert_completed_fraction={dataset_summary['expert_completed_fraction']:.3f} "
        f"expert_deadline={dataset_summary['expert_deadline_satisfaction']:.3f} "
        f"expert_collisions={dataset_summary['expert_collisions']:.2f}"
    )
    for warning in dataset_quality_warnings:
        print(f"[WARN] {warning}")
    task_rules = _task_rule_config(max_steps=max_steps, xml_path=xml_path, enforce_visit_order=enforce_visit_order)

    resolved_config = {
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "train_episodes": int(train_episodes),
        "val_episodes": int(val_episodes),
        "rollout_episodes": int(rollout_episodes),
        "eval_interval": int(eval_interval),
        "save_interval": int(save_interval),
        "max_steps": int(max_steps),
        "hidden_dim": int(hidden_dim),
        "edge_hidden_dim": int(edge_hidden_dim),
        "dataset_layout_pattern": str(dataset_layout_pattern),
        "expert_max_collisions": None if expert_max_collisions is None else int(expert_max_collisions),
        "eval_layout_mode": str(eval_layout_mode),
        "eval_strict_collision_stop": bool(eval_strict_collision_stop),
        "enforce_visit_order": None if enforce_visit_order is None else bool(enforce_visit_order),
        "resume_path": resume_path,
        "seed": int(seed),
        "run_name": run_paths["run_name"],
        "output_root": output_root,
        "xml_path": xml_path,
    }
    _save_json(
        run_paths["config_path"],
        {
            "config": resolved_config,
            "dataset": dataset_summary,
            "policy_target": "joint_targets_from_xml_layout",
            "task_rules": task_rules,
        },
    )

    agent = StandardGNNAgent(
        num_robots=num_robots,
        num_tasks=num_tasks,
        max_steps=max_steps,
        hidden_dim=hidden_dim,
        edge_hidden_dim=edge_hidden_dim,
        xml_path=xml_path,
    )
    optimizer = torch.optim.AdamW(agent.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    start_epoch = 0
    best_score = -np.inf
    if resume_path:
        metadata = agent.load(resume_path, optimizer=optimizer)
        start_epoch = metadata["epoch"]
        best_score = metadata["best_score"]
        print(f"[RESUME] loaded {resume_path} epoch={start_epoch} best_score={best_score:.4f}")

    best_eval_metrics = None
    for epoch in range(start_epoch + 1, start_epoch + epochs + 1):
        agent.model.train()
        train_losses = []
        for batch in iterate_minibatches(train_dataset, batch_size=batch_size, shuffle=True, seed=seed + epoch):
            optimizer.zero_grad()
            pred_actions = agent.forward_batch(batch)
            target_actions = torch.from_numpy(batch["action_targets"]).to(agent.device, dtype=torch.float32)
            active_mask = torch.from_numpy(batch["active_mask"]).to(agent.device, dtype=torch.float32)
            loss = compute_action_loss(pred_actions, target_actions, active_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.model.parameters(), max_norm=2.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        agent.model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in iterate_minibatches(val_dataset, batch_size=batch_size, shuffle=False):
                pred_actions = agent.forward_batch(batch)
                target_actions = torch.from_numpy(batch["action_targets"]).to(agent.device, dtype=torch.float32)
                active_mask = torch.from_numpy(batch["active_mask"]).to(agent.device, dtype=torch.float32)
                loss = compute_action_loss(pred_actions, target_actions, active_mask)
                val_losses.append(float(loss.detach().cpu().item()))

        epoch_record = {"epoch": int(epoch), "train_loss": float(np.mean(train_losses)), "val_loss": float(np.mean(val_losses))}
        log_line = f"epoch={epoch} train_loss={epoch_record['train_loss']:.6f} val_loss={epoch_record['val_loss']:.6f}"
        if eval_interval > 0 and epoch % eval_interval == 0:
            metrics = evaluate_agent(
                agent,
                num_episodes=rollout_episodes,
                max_steps=max_steps,
                base_seed=seed + 10000,
                layout_mode=eval_layout_mode,
                strict_collision_stop=eval_strict_collision_stop,
                xml_path=xml_path,
                enforce_visit_order=enforce_visit_order,
            )
            score = metrics["success_rate"] + 0.35 * metrics["avg_deadline_satisfaction"] - 0.05 * metrics["avg_collisions"]
            epoch_record["eval"] = {key: float(value) for key, value in metrics.items()}
            epoch_record["selection_score"] = float(score)
            log_line += (
                f" eval_success={metrics['success_rate']:.3f}"
                f" eval_deadline={metrics['avg_deadline_satisfaction']:.3f}"
                f" eval_collisions={metrics['avg_collisions']:.3f}"
                f" eval_task_dist={metrics['avg_mean_task_distance']:.3f}"
            )
            if score > best_score:
                best_score = score
                best_eval_metrics = epoch_record["eval"]
                _safe_save_checkpoint(agent, run_paths["best_path"], epoch=epoch, best_score=best_score, optimizer=optimizer, required=True)
                _safe_save_checkpoint(agent, run_paths["canonical_best_path"], epoch=epoch, best_score=best_score, optimizer=optimizer, required=False)
                log_line += " best=updated"

        _append_jsonl(run_paths["metrics_path"], epoch_record)
        print(log_line)

        if save_interval > 0 and epoch % save_interval == 0:
            _safe_save_checkpoint(agent, run_paths["latest_path"], epoch=epoch, best_score=best_score, optimizer=optimizer, required=True)
            _safe_save_checkpoint(agent, run_paths["canonical_latest_path"], epoch=epoch, best_score=best_score, optimizer=optimizer, required=False)
            print(f"[SAVE] epoch={epoch} latest checkpoint saved to {run_paths['latest_path']}")

    _safe_save_checkpoint(agent, run_paths["latest_path"], epoch=start_epoch + epochs, best_score=best_score, optimizer=optimizer, required=True)
    _safe_save_checkpoint(agent, run_paths["canonical_latest_path"], epoch=start_epoch + epochs, best_score=best_score, optimizer=optimizer, required=False)
    _save_json(
        run_paths["summary_path"],
        {
            "run_name": run_paths["run_name"],
            "run_dir": run_paths["run_dir"],
            "best_score": float(best_score),
            "best_eval": best_eval_metrics,
            "task_rules": task_rules,
            "final_epoch": int(start_epoch + epochs),
            "best_checkpoint": run_paths["best_path"],
            "latest_checkpoint": run_paths["latest_path"],
        },
    )
    print("训练完成，模型已保存。")


def main():
    parser = argparse.ArgumentParser(description="Standard GNN imitation training for multi-robot task planning")
    parser.add_argument("--config_path", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--train_episodes", type=int, default=160)
    parser.add_argument("--val_episodes", type=int, default=40)
    parser.add_argument("--rollout_episodes", type=int, default=20)
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=280)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--edge_hidden_dim", type=int, default=64)
    parser.add_argument("--dataset_layout_pattern", type=str, default="structured,structured,random,representative")
    parser.add_argument("--expert_max_collisions", type=int, default=None)
    parser.add_argument("--eval_layout_mode", type=str, choices=["representative", "structured", "random"], default="representative")
    parser.add_argument("--eval_strict_collision_stop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enforce_visit_order", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=4300)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=os.path.join(ROOT_DIR, "runs"))
    parser.add_argument("--xml_path", type=str, default=None)
    args = parser.parse_args()
    args = _apply_config_overrides(args, parser, sys.argv[1:])
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        train_episodes=args.train_episodes,
        val_episodes=args.val_episodes,
        rollout_episodes=args.rollout_episodes,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        max_steps=args.max_steps,
        hidden_dim=args.hidden_dim,
        edge_hidden_dim=args.edge_hidden_dim,
        dataset_layout_pattern=args.dataset_layout_pattern,
        expert_max_collisions=args.expert_max_collisions,
        eval_layout_mode=args.eval_layout_mode,
        eval_strict_collision_stop=args.eval_strict_collision_stop,
        enforce_visit_order=args.enforce_visit_order,
        resume_path=args.resume_path,
        seed=args.seed,
        run_name=args.run_name,
        output_root=args.output_root,
        xml_path=args.xml_path,
    )


if __name__ == "__main__":
    main()