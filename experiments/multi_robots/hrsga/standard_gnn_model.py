import os
import time

import numpy as np
import torch
from torch import nn

try:
    from .env_hrsga import HRSGAEnvironment
    from .hrsga_model import build_behavior_dataset, iterate_minibatches, compute_action_loss, snapshot_to_model_inputs
except ImportError:
    from env_hrsga import HRSGAEnvironment
    from hrsga_model import build_behavior_dataset, iterate_minibatches, compute_action_loss, snapshot_to_model_inputs


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation=nn.ReLU):
        super().__init__()
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(activation())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.net(inputs)


class SharedMessagePassing(nn.Module):
    def __init__(self, hidden_dim, edge_hidden_dim):
        super().__init__()
        self.edge_rr = MLP(8, [edge_hidden_dim], hidden_dim)
        self.edge_tr = MLP(14, [edge_hidden_dim], hidden_dim)
        self.edge_or = MLP(5, [edge_hidden_dim], hidden_dim)
        self.message_mlp = MLP(hidden_dim * 3, [hidden_dim, hidden_dim], hidden_dim)
        self.update_mlp = MLP(hidden_dim * 2, [hidden_dim * 2, hidden_dim], hidden_dim)

    def _aggregate_relation(self, target_hidden, source_hidden, edge_features, edge_mask, edge_encoder):
        batch_size, target_count, _ = target_hidden.shape
        source_count = source_hidden.shape[1]
        if source_count == 0:
            return torch.zeros_like(target_hidden)
        edge_hidden = edge_encoder(edge_features)
        target_expand = target_hidden.unsqueeze(2).expand(-1, -1, source_count, -1)
        source_expand = source_hidden.unsqueeze(1).expand(-1, target_count, -1, -1)
        messages = self.message_mlp(torch.cat([target_expand, source_expand, edge_hidden], dim=-1))
        masked_messages = messages * edge_mask.unsqueeze(-1)
        denom = edge_mask.sum(dim=2, keepdim=True).clamp_min(1.0)
        return masked_messages.sum(dim=2) / denom

    def forward(self, robot_hidden, task_hidden, obstacle_hidden, rr_edges, tr_edges, or_edges, rr_mask, tr_mask, or_mask):
        rr_context = self._aggregate_relation(robot_hidden, robot_hidden, rr_edges, rr_mask.float(), self.edge_rr)
        tr_context = self._aggregate_relation(robot_hidden, task_hidden, tr_edges, tr_mask.float(), self.edge_tr)
        or_context = self._aggregate_relation(robot_hidden, obstacle_hidden, or_edges, or_mask.float(), self.edge_or)
        merged = (rr_context + tr_context + or_context) / 3.0
        return self.update_mlp(torch.cat([robot_hidden, merged], dim=-1))


class StandardGNNPolicyNetwork(nn.Module):
    def __init__(self, hidden_dim=128, edge_hidden_dim=64, action_scale=4.5):
        super().__init__()
        self.action_scale = action_scale
        self.robot_encoder = MLP(19, [hidden_dim, hidden_dim], hidden_dim)
        self.task_encoder = MLP(14, [hidden_dim, hidden_dim], hidden_dim)
        self.obstacle_encoder = MLP(3, [hidden_dim, hidden_dim], hidden_dim)
        self.message_passing = SharedMessagePassing(hidden_dim=hidden_dim, edge_hidden_dim=edge_hidden_dim)
        self.action_head = MLP(hidden_dim, [hidden_dim, hidden_dim], 6)

    def forward(self, robot_features, task_features, obstacle_features, rr_edges, tr_edges, or_edges, rr_mask, tr_mask, or_mask, active_mask):
        robot_hidden = self.robot_encoder(robot_features)
        task_hidden = self.task_encoder(task_features)
        obstacle_hidden = self.obstacle_encoder(obstacle_features)
        updated_hidden = self.message_passing(robot_hidden, task_hidden, obstacle_hidden, rr_edges, tr_edges, or_edges, rr_mask, tr_mask, or_mask)
        actions = torch.tanh(self.action_head(updated_hidden)) * self.action_scale
        return actions * active_mask.unsqueeze(-1)


class StandardGNNAgent:
    def __init__(self, num_robots, max_steps, num_tasks=None, hidden_dim=128, edge_hidden_dim=64, device=None, xml_path=None):
        self.num_robots = int(num_robots)
        self.max_steps = int(max_steps)
        self.num_tasks = None if num_tasks is None else int(num_tasks)
        self.hidden_dim = int(hidden_dim)
        self.edge_hidden_dim = int(edge_hidden_dim)
        self.xml_path = xml_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = StandardGNNPolicyNetwork(hidden_dim=self.hidden_dim, edge_hidden_dim=self.edge_hidden_dim).to(self.device)

    def _to_tensor(self, value):
        if isinstance(value, torch.Tensor):
            return value.to(self.device)
        dtype = torch.bool if value.dtype == np.bool_ else torch.float32
        return torch.from_numpy(value).to(self.device, dtype=dtype)

    def forward_batch(self, batch):
        return self.model(
            self._to_tensor(batch["robot_features"]),
            self._to_tensor(batch["task_features"]),
            self._to_tensor(batch["obstacle_features"]),
            self._to_tensor(batch["rr_edges"]),
            self._to_tensor(batch["tr_edges"]),
            self._to_tensor(batch["or_edges"]),
            self._to_tensor(batch["rr_mask"]),
            self._to_tensor(batch["tr_mask"]),
            self._to_tensor(batch["or_mask"]),
            self._to_tensor(batch["active_mask"]),
        )

    def select_action(self, snapshot):
        batch = snapshot_to_model_inputs(snapshot, max_steps=self.max_steps)
        batch = {key: value[None, ...] for key, value in batch.items()}
        self.model.eval()
        with torch.no_grad():
            actions = self.forward_batch(batch)[0].cpu().numpy()
        return actions.astype(np.float32)

    def save(self, path, epoch=0, best_score=-np.inf, optimizer=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            "model_state": self.model.state_dict(),
            "epoch": int(epoch),
            "best_score": float(best_score),
            "num_robots": self.num_robots,
            "num_tasks": self.num_tasks,
            "max_steps": self.max_steps,
            "hidden_dim": self.hidden_dim,
            "edge_hidden_dim": self.edge_hidden_dim,
            "xml_path": self.xml_path,
        }
        if optimizer is not None:
            checkpoint["optimizer_state"] = optimizer.state_dict()
        _save_checkpoint_with_retry(checkpoint, path)

    def load(self, path, optimizer=None):
        checkpoint = torch.load(path, map_location=self.device)
        adapted_state = _adapt_state_dict_for_model(self.model.state_dict(), checkpoint["model_state"])
        self.model.load_state_dict(adapted_state)
        if optimizer is not None and "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        return {
            "epoch": int(checkpoint.get("epoch", 0)),
            "best_score": float(checkpoint.get("best_score", -np.inf)),
        }


def _adapt_state_dict_for_model(model_state, checkpoint_state):
    adapted = dict(model_state)
    for key, value in checkpoint_state.items():
        if key not in adapted:
            continue
        target = adapted[key]
        if target.shape == value.shape:
            adapted[key] = value
            continue
        if target.ndim != value.ndim:
            continue
        if any(target.shape[index] != value.shape[index] for index in range(target.ndim - 1)):
            continue
        if target.shape[-1] < value.shape[-1]:
            continue
        patched = target.clone()
        slices = tuple(slice(0, value.shape[index]) for index in range(value.ndim))
        patched[slices] = value
        adapted[key] = patched
    return adapted


def load_agent_from_checkpoint(path, device=None):
    checkpoint = torch.load(path, map_location=device or ("cuda" if torch.cuda.is_available() else "cpu"))
    agent = StandardGNNAgent(
        num_robots=int(checkpoint["num_robots"]),
        num_tasks=checkpoint.get("num_tasks"),
        max_steps=int(checkpoint["max_steps"]),
        hidden_dim=int(checkpoint.get("hidden_dim", 128)),
        edge_hidden_dim=int(checkpoint.get("edge_hidden_dim", 64)),
        device=device,
        xml_path=checkpoint.get("xml_path"),
    )
    adapted_state = _adapt_state_dict_for_model(agent.model.state_dict(), checkpoint["model_state"])
    agent.model.load_state_dict(adapted_state)
    agent.model.eval()
    return agent


def _save_checkpoint_with_retry(checkpoint, path, attempts=6, retry_delay=0.6):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            torch.save(checkpoint, tmp_path)
            os.replace(tmp_path, path)
            return
        except (OSError, RuntimeError) as error:
            last_error = error
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
        if attempt < attempts:
            time.sleep(retry_delay)
    raise RuntimeError(f"Failed to save checkpoint to {path} after {attempts} attempts") from last_error


def evaluate_agent(agent, num_episodes=10, max_steps=None, base_seed=5300, layout_mode="structured", strict_collision_stop=False, xml_path=None, enforce_visit_order=None):
    max_steps = int(agent.max_steps if max_steps is None else max_steps)
    returns = {
        "success_rate": 0.0,
        "avg_completed_fraction": 0.0,
        "avg_deadline_satisfaction": 0.0,
        "avg_collisions": 0.0,
        "avg_mean_task_distance": 0.0,
        "avg_min_pair_distance": 0.0,
        "avg_missed_deadlines": 0.0,
        "avg_steps": 0.0,
        "collision_stop_rate": 0.0,
    }
    runs = []
    for episode_index in range(num_episodes):
        env = HRSGAEnvironment(
            max_steps=max_steps,
            xml_path=xml_path or getattr(agent, "xml_path", None),
            enforce_visit_order=enforce_visit_order,
        )
        snapshot = env.reset(seed=base_seed + episode_index, layout_mode=layout_mode)
        collision_terminated = False
        while True:
            actions = agent.select_action(snapshot)
            snapshot, done, info = env.step(actions)
            if strict_collision_stop and int(info.get("collisions", 0)) > 0:
                done = True
                collision_terminated = True
            if done:
                runs.append(
                    {
                        "success": float(info["success"]),
                        "completed_fraction": info["completed_tasks"] / max(1, len(snapshot["tasks"])),
                        "deadline_satisfaction": float(info["deadline_satisfaction"]),
                        "collisions": float(info["total_collisions"]),
                        "mean_task_distance": float(info["mean_task_distance"]),
                        "min_pair_distance": float(info["min_pair_distance"]),
                        "missed_deadlines": float(info["missed_deadlines"]),
                        "steps": float(snapshot["step"]),
                        "collision_terminated": float(collision_terminated),
                    }
                )
                break

    if not runs:
        return returns

    returns["success_rate"] = float(np.mean([run["success"] for run in runs]))
    returns["avg_completed_fraction"] = float(np.mean([run["completed_fraction"] for run in runs]))
    returns["avg_deadline_satisfaction"] = float(np.mean([run["deadline_satisfaction"] for run in runs]))
    returns["avg_collisions"] = float(np.mean([run["collisions"] for run in runs]))
    returns["avg_mean_task_distance"] = float(np.mean([run["mean_task_distance"] for run in runs]))
    returns["avg_min_pair_distance"] = float(np.mean([run["min_pair_distance"] for run in runs]))
    returns["avg_missed_deadlines"] = float(np.mean([run["missed_deadlines"] for run in runs]))
    returns["avg_steps"] = float(np.mean([run["steps"] for run in runs]))
    returns["collision_stop_rate"] = float(np.mean([run.get("collision_terminated", 0.0) for run in runs]))
    return returns
