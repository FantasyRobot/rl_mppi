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

from hrsga_ball_model import build_behavior_dataset, compute_action_loss, iterate_minibatches
from standard_gnn_ball_model import StandardGNNBallAgent, evaluate_agent


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
        "best_path": os.path.join(checkpoint_dir, "standard_gnn_ball_best.pt"),
        "latest_path": os.path.join(checkpoint_dir, "standard_gnn_ball_latest.pt"),
        "canonical_best_path": os.path.join(ROOT_DIR, "models", "standard_gnn_ball_best.pt"),
        "canonical_latest_path": os.path.join(ROOT_DIR, "models", "standard_gnn_ball_latest.pt"),
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
    valid_dests = {action.dest for action in parser._actions}
    explicit_keys = _explicit_cli_keys(argv)
    for key, value in config.items():
        if key not in valid_dests or key in explicit_keys or key == "config_path":
            continue
        setattr(args, key, value)
    return args


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
    num_balls=4,
    num_tasks=8,
    max_steps=180,
    hidden_dim=128,
    edge_hidden_dim=64,
    resume_path=None,
    seed=4300,
    run_name=None,
    output_root=None,
):
    output_root = output_root or os.path.join(ROOT_DIR, "runs")
    run_paths = _build_run_paths(output_root, run_name)
    os.makedirs(run_paths["checkpoint_dir"], exist_ok=True)
    os.makedirs(run_paths["log_dir"], exist_ok=True)

    dataset, episode_stats = build_behavior_dataset(
        num_episodes=train_episodes + val_episodes,
        num_balls=num_balls,
        num_tasks=num_tasks,
        max_steps=max_steps,
        seed_start=seed,
    )
    train_dataset, val_dataset = _split_dataset(dataset, train_episodes / max(train_episodes + val_episodes, 1))
    print(
        f"[DATA] train_samples={train_dataset.robot_features.shape[0]} val_samples={val_dataset.robot_features.shape[0]} "
        f"expert_stats={_dataset_summary(episode_stats)} run_dir={run_paths['run_dir']}"
    )

    _save_json(
        run_paths["config_path"],
        {
            "config": {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "train_episodes": train_episodes,
                "val_episodes": val_episodes,
                "rollout_episodes": rollout_episodes,
                "eval_interval": eval_interval,
                "save_interval": save_interval,
                "num_balls": num_balls,
                "num_tasks": num_tasks,
                "max_steps": max_steps,
                "hidden_dim": hidden_dim,
                "edge_hidden_dim": edge_hidden_dim,
                "resume_path": resume_path,
                "seed": seed,
                "run_name": run_paths["run_name"],
                "output_root": output_root,
            }
        },
    )

    agent = StandardGNNBallAgent(num_balls=num_balls, num_tasks=num_tasks, max_steps=max_steps, hidden_dim=hidden_dim, edge_hidden_dim=edge_hidden_dim)
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
            metrics = evaluate_agent(agent, num_episodes=rollout_episodes, num_balls=num_balls, num_tasks=num_tasks, max_steps=max_steps, base_seed=seed + 10000)
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
            "final_epoch": int(start_epoch + epochs),
            "best_checkpoint": run_paths["best_path"],
            "latest_checkpoint": run_paths["latest_path"],
        },
    )
    print("训练完成，模型已保存。")


def main():
    parser = argparse.ArgumentParser(description="Standard GNN imitation training for multi-ball task planning")
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
    parser.add_argument("--edge_hidden_dim", type=int, default=64)
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=4300)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=os.path.join(ROOT_DIR, "runs"))
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
        num_balls=args.num_balls,
        num_tasks=args.num_tasks,
        max_steps=args.max_steps,
        hidden_dim=args.hidden_dim,
        edge_hidden_dim=args.edge_hidden_dim,
        resume_path=args.resume_path,
        seed=args.seed,
        run_name=args.run_name,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()