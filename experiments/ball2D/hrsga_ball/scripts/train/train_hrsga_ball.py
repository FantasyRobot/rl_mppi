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

from hrsga_ball_model import HRSGABallAgent, build_behavior_dataset, compute_action_loss, evaluate_agent, iterate_minibatches


def _save_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _append_jsonl(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _build_run_paths(output_root, run_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    resolved_run_name = run_name or f"hrsga_run_{timestamp}"
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
        "best_path": os.path.join(checkpoint_dir, "hrsga_ball_best.pt"),
        "latest_path": os.path.join(checkpoint_dir, "hrsga_ball_latest.pt"),
        "canonical_best_path": os.path.join(ROOT_DIR, "models", "hrsga_ball_best.pt"),
        "canonical_latest_path": os.path.join(ROOT_DIR, "models", "hrsga_ball_latest.pt"),
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
    train_dataset = type(dataset)(
        robot_features=dataset.robot_features[train_slice],
        task_features=dataset.task_features[train_slice],
        obstacle_features=dataset.obstacle_features[train_slice],
        rr_edges=dataset.rr_edges[train_slice],
        tr_edges=dataset.tr_edges[train_slice],
        or_edges=dataset.or_edges[train_slice],
        rr_mask=dataset.rr_mask[train_slice],
        tr_mask=dataset.tr_mask[train_slice],
        or_mask=dataset.or_mask[train_slice],
        active_mask=dataset.active_mask[train_slice],
        action_targets=dataset.action_targets[train_slice],
    )
    val_dataset = type(dataset)(
        robot_features=dataset.robot_features[val_slice],
        task_features=dataset.task_features[val_slice],
        obstacle_features=dataset.obstacle_features[val_slice],
        rr_edges=dataset.rr_edges[val_slice],
        tr_edges=dataset.tr_edges[val_slice],
        or_edges=dataset.or_edges[val_slice],
        rr_mask=dataset.rr_mask[val_slice],
        tr_mask=dataset.tr_mask[val_slice],
        or_mask=dataset.or_mask[val_slice],
        active_mask=dataset.active_mask[val_slice],
        action_targets=dataset.action_targets[val_slice],
    )
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
    num_balls=4,
    max_steps=180,
    hidden_dim=128,
    num_heads=4,
    topk_robot=2,
    topk_task=2,
    topk_obstacle=1,
    disable_temporal_bias=False,
    disable_geometric_bias=False,
    shared_relation_attention=False,
    use_dense_residual=False,
    dataset_layout_pattern="structured,structured,random,representative",
    expert_max_collisions=None,
    eval_layout_mode="representative",
    eval_strict_collision_stop=True,
    resume_path=None,
    seed=4300,
    run_name=None,
    output_root=None,
):
    output_root = output_root or os.path.join(ROOT_DIR, "runs")
    run_paths = _build_run_paths(output_root, run_name)
    os.makedirs(run_paths["checkpoint_dir"], exist_ok=True)
    os.makedirs(run_paths["log_dir"], exist_ok=True)

    dataset, episode_stats, dataset_metadata = build_behavior_dataset(
        num_episodes=train_episodes + val_episodes,
        num_balls=num_balls,
        max_steps=max_steps,
        seed_start=seed,
        layout_pattern=dataset_layout_pattern,
        expert_max_collisions=expert_max_collisions,
        return_metadata=True,
    )
    train_dataset, val_dataset = _split_dataset(dataset, train_episodes / max(train_episodes + val_episodes, 1))
    dataset_summary = {
        "train_samples": int(train_dataset.robot_features.shape[0]),
        "val_samples": int(val_dataset.robot_features.shape[0]),
        "expert_episode_count": int(len(episode_stats)),
        "expert_stats": _dataset_summary(episode_stats),
        "dataset_filter": _filter_summary(dataset_metadata),
        "requested_episode_count": int(dataset_metadata.get("requested_episode_count", len(episode_stats))),
        "attempted_episode_count": int(dataset_metadata.get("attempted_episode_count", len(episode_stats))),
        "dropped_episode_count": int(dataset_metadata.get("dropped_episode_count", 0)),
        "expert_max_collisions": dataset_metadata.get("expert_max_collisions"),
        "dropped_expert_stats": _dataset_summary(dataset_metadata.get("dropped_expert_stats", [])),
    }
    print(
        f"[DATA] train_samples={dataset_summary['train_samples']} val_samples={dataset_summary['val_samples']} "
        f"expert_stats={dataset_summary['expert_stats']} {dataset_summary['dataset_filter']} "
        f"run_dir={run_paths['run_dir']}"
    )

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
        "num_balls": int(num_balls),
        "max_steps": int(max_steps),
        "hidden_dim": int(hidden_dim),
        "num_heads": int(num_heads),
        "topk_robot": int(topk_robot),
        "topk_task": int(topk_task),
        "topk_obstacle": int(topk_obstacle),
        "disable_temporal_bias": bool(disable_temporal_bias),
        "disable_geometric_bias": bool(disable_geometric_bias),
        "shared_relation_attention": bool(shared_relation_attention),
        "use_dense_residual": bool(use_dense_residual),
        "dataset_layout_pattern": str(dataset_layout_pattern),
        "expert_max_collisions": None if expert_max_collisions is None else int(expert_max_collisions),
        "eval_layout_mode": str(eval_layout_mode),
        "eval_strict_collision_stop": bool(eval_strict_collision_stop),
        "resume_path": resume_path,
        "seed": int(seed),
        "run_name": run_paths["run_name"],
        "output_root": output_root,
    }
    _save_json(
        run_paths["config_path"],
        {
            "config": resolved_config,
            "dataset": dataset_summary,
        },
    )

    agent = HRSGABallAgent(
        num_balls=num_balls,
        max_steps=max_steps,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        topk_robot=topk_robot,
        topk_task=topk_task,
        topk_obstacle=topk_obstacle,
        disable_temporal_bias=disable_temporal_bias,
        disable_geometric_bias=disable_geometric_bias,
        shared_relation_attention=shared_relation_attention,
        use_dense_residual=use_dense_residual,
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

        epoch_record = {
            "epoch": int(epoch),
            "train_loss": float(np.mean(train_losses)),
            "val_loss": float(np.mean(val_losses)),
        }
        log_line = f"epoch={epoch} train_loss={epoch_record['train_loss']:.6f} val_loss={epoch_record['val_loss']:.6f}"
        metrics = None
        if eval_interval > 0 and epoch % eval_interval == 0:
            metrics = evaluate_agent(
                agent,
                num_episodes=rollout_episodes,
                num_balls=num_balls,
                max_steps=max_steps,
                base_seed=seed + 10000,
                layout_mode=eval_layout_mode,
                strict_collision_stop=eval_strict_collision_stop,
            )
            score = (
                1.0 * metrics["success_rate"]
                + 0.35 * metrics["avg_deadline_satisfaction"]
                - 0.10 * metrics["avg_collisions"]
                - 0.25 * metrics.get("collision_stop_rate", 0.0)
            )
            epoch_record["eval"] = {key: float(value) for key, value in metrics.items()}
            epoch_record["selection_score"] = float(score)
            log_line += (
                f" eval_success={metrics['success_rate']:.3f}"
                f" eval_deadline={metrics['avg_deadline_satisfaction']:.3f}"
                f" eval_collisions={metrics['avg_collisions']:.3f}"
                f" eval_collision_stop={metrics.get('collision_stop_rate', 0.0):.3f}"
                f" eval_task_dist={metrics['avg_mean_task_distance']:.3f}"
            )
            if score > best_score:
                best_score = score
                best_eval_metrics = epoch_record.get("eval")
                agent.save(run_paths["best_path"], epoch=epoch, best_score=best_score, optimizer=optimizer)
                agent.save(run_paths["canonical_best_path"], epoch=epoch, best_score=best_score, optimizer=optimizer)
                log_line += " best=updated"
                epoch_record["best_updated"] = True

        _append_jsonl(run_paths["metrics_path"], epoch_record)
        print(log_line)

        if save_interval > 0 and epoch % save_interval == 0:
            agent.save(run_paths["latest_path"], epoch=epoch, best_score=best_score, optimizer=optimizer)
            agent.save(run_paths["canonical_latest_path"], epoch=epoch, best_score=best_score, optimizer=optimizer)
            print(f"[SAVE] epoch={epoch} latest checkpoint saved to {run_paths['latest_path']}")

    agent.save(run_paths["latest_path"], epoch=start_epoch + epochs, best_score=best_score, optimizer=optimizer)
    agent.save(run_paths["canonical_latest_path"], epoch=start_epoch + epochs, best_score=best_score, optimizer=optimizer)
    _save_json(
        run_paths["summary_path"],
        {
            "run_name": run_paths["run_name"],
            "run_dir": run_paths["run_dir"],
            "best_score": float(best_score),
            "best_eval": best_eval_metrics,
            "final_epoch": int(start_epoch + epochs),
            "config_path": run_paths["config_path"],
            "metrics_path": run_paths["metrics_path"],
            "best_checkpoint": run_paths["best_path"],
            "latest_checkpoint": run_paths["latest_path"],
            "canonical_best_checkpoint": run_paths["canonical_best_path"],
            "canonical_latest_checkpoint": run_paths["canonical_latest_path"],
        },
    )
    print("训练完成，模型已保存。")


def main():
    parser = argparse.ArgumentParser(description="HRSGA imitation training for multi-ball task planning")
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
        max_steps=args.max_steps,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        topk_robot=args.topk_robot,
        topk_task=args.topk_task,
        topk_obstacle=args.topk_obstacle,
        disable_temporal_bias=args.disable_temporal_bias,
        disable_geometric_bias=args.disable_geometric_bias,
        shared_relation_attention=args.shared_relation_attention,
        use_dense_residual=args.use_dense_residual,
        dataset_layout_pattern=args.dataset_layout_pattern,
        expert_max_collisions=args.expert_max_collisions,
        eval_layout_mode=args.eval_layout_mode,
        eval_strict_collision_stop=args.eval_strict_collision_stop,
        resume_path=args.resume_path,
        seed=args.seed,
        run_name=args.run_name,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()