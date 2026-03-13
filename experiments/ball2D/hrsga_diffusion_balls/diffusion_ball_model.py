import math
import os
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

try:
    from .envball_hrsga import HRSGABallEnvironment
    from .hrsga_ball_model import (
        HRSGAExpertController,
        HRSGAStateEncoder,
        DatasetSplit,
        _adapt_state_dict_for_model,
        _normalize_layout_pattern,
        _save_checkpoint_with_retry,
        snapshot_to_sample,
    )
except ImportError:
    from envball_hrsga import HRSGABallEnvironment
    from hrsga_ball_model import (
        HRSGAExpertController,
        HRSGAStateEncoder,
        DatasetSplit,
        _adapt_state_dict_for_model,
        _normalize_layout_pattern,
        _save_checkpoint_with_retry,
        snapshot_to_sample,
    )


@dataclass
class DiffusionDatasetSplit:
    robot_features: np.ndarray
    task_features: np.ndarray
    obstacle_features: np.ndarray
    rr_edges: np.ndarray
    tr_edges: np.ndarray
    or_edges: np.ndarray
    rr_mask: np.ndarray
    tr_mask: np.ndarray
    or_mask: np.ndarray
    active_mask: np.ndarray
    action_targets: np.ndarray
    action_chunk_targets: np.ndarray


def _stack_diffusion_samples(samples):
    if not samples:
        raise ValueError("No diffusion samples collected.")
    stacked = {}
    for key in samples[0]:
        stacked[key] = np.stack([sample[key] for sample in samples], axis=0)
    return DiffusionDatasetSplit(**stacked)


def _split_diffusion_dataset(dataset, train_ratio):
    sample_count = dataset.robot_features.shape[0]
    split_index = max(1, min(sample_count - 1, int(sample_count * train_ratio)))
    train_slice = slice(0, split_index)
    val_slice = slice(split_index, sample_count)
    train_dataset = DiffusionDatasetSplit(
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
        action_chunk_targets=dataset.action_chunk_targets[train_slice],
    )
    val_dataset = DiffusionDatasetSplit(
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
        action_chunk_targets=dataset.action_chunk_targets[val_slice],
    )
    return train_dataset, val_dataset


def iterate_diffusion_minibatches(dataset, batch_size, shuffle=True, seed=0):
    sample_count = dataset.robot_features.shape[0]
    indices = np.arange(sample_count)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    for start in range(0, sample_count, batch_size):
        batch_indices = indices[start:start + batch_size]
        yield {
            "robot_features": dataset.robot_features[batch_indices],
            "task_features": dataset.task_features[batch_indices],
            "obstacle_features": dataset.obstacle_features[batch_indices],
            "rr_edges": dataset.rr_edges[batch_indices],
            "tr_edges": dataset.tr_edges[batch_indices],
            "or_edges": dataset.or_edges[batch_indices],
            "rr_mask": dataset.rr_mask[batch_indices],
            "tr_mask": dataset.tr_mask[batch_indices],
            "or_mask": dataset.or_mask[batch_indices],
            "active_mask": dataset.active_mask[batch_indices],
            "action_targets": dataset.action_targets[batch_indices],
            "action_chunk_targets": dataset.action_chunk_targets[batch_indices],
        }


def build_diffusion_behavior_dataset(
    num_episodes,
    num_balls,
    max_steps,
    num_tasks=None,
    horizon=4,
    seed_start=4300,
    layout_pattern="structured",
    expert_max_collisions=None,
    return_metadata=False,
):
    controller = HRSGAExpertController()
    resolved_pattern = _normalize_layout_pattern(layout_pattern)
    samples = []
    episode_stats = []
    dropped_episode_stats = []
    target_episode_count = int(num_episodes)
    kept_episode_count = 0
    attempt_index = 0
    horizon = int(max(1, horizon))
    max_attempts = max(target_episode_count, 1) * 20 if expert_max_collisions is not None else target_episode_count

    while kept_episode_count < target_episode_count:
        if attempt_index >= max_attempts:
            raise ValueError(
                "Unable to collect enough expert episodes under the collision filter. "
                f"requested={target_episode_count} kept={kept_episode_count} attempted={attempt_index} "
                f"expert_max_collisions={expert_max_collisions}"
            )
        env = HRSGABallEnvironment(num_balls=num_balls, num_tasks=num_tasks, max_steps=max_steps)
        layout_mode = resolved_pattern[attempt_index % len(resolved_pattern)]
        snapshot = env.reset(seed=seed_start + attempt_index, layout_mode=layout_mode)
        episode_samples = []
        episode_actions = []
        while True:
            expert_actions = np.asarray(controller.act(snapshot), dtype=np.float32)
            episode_samples.append(snapshot_to_sample(snapshot, max_steps=max_steps, action_targets=expert_actions))
            episode_actions.append(expert_actions)
            snapshot, done, info = env.step(expert_actions)
            if done:
                info = dict(info)
                info["layout_mode"] = layout_mode
                total_collisions = int(info.get("total_collisions", 0))
                keep_episode = expert_max_collisions is None or total_collisions <= int(expert_max_collisions)
                if keep_episode:
                    for step_index, sample in enumerate(episode_samples):
                        action_chunk = []
                        for offset in range(horizon):
                            future_index = min(step_index + offset, len(episode_actions) - 1)
                            action_chunk.append(episode_actions[future_index])
                        enriched = dict(sample)
                        enriched["action_chunk_targets"] = np.stack(action_chunk, axis=0).astype(np.float32)
                        samples.append(enriched)
                    episode_stats.append(info)
                    kept_episode_count += 1
                else:
                    dropped_episode_stats.append(info)
                break
        attempt_index += 1

    dataset = _stack_diffusion_samples(samples)
    if not return_metadata:
        return dataset, episode_stats

    metadata = {
        "requested_episode_count": target_episode_count,
        "attempted_episode_count": int(attempt_index),
        "kept_episode_count": int(len(episode_stats)),
        "dropped_episode_count": int(len(dropped_episode_stats)),
        "expert_max_collisions": None if expert_max_collisions is None else int(expert_max_collisions),
        "horizon": horizon,
        "kept_expert_stats": episode_stats,
        "dropped_expert_stats": dropped_episode_stats,
    }
    return dataset, episode_stats, metadata


def sinusoidal_embedding(timesteps, dim, device):
    half_dim = dim // 2
    exponent = -math.log(10000.0) / max(half_dim - 1, 1)
    frequencies = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * exponent)
    angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=1)
    return embedding


class DiffusionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        current = input_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(current, hidden))
            layers.append(nn.SiLU())
            current = hidden
        layers.append(nn.Linear(current, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.net(inputs)


class HRSGADiffusionPoilcyNetwork(nn.Module):
    def __init__(
        self,
        num_balls,
        horizon,
        robot_dim,
        task_dim,
        obstacle_dim,
        rr_edge_dim,
        tr_edge_dim,
        or_edge_dim,
        hidden_dim=128,
        num_heads=4,
        topk_robot=2,
        topk_task=2,
        topk_obstacle=1,
        action_scale=1.4,
        diffusion_steps=16,
        disable_temporal_bias=False,
        disable_geometric_bias=False,
        shared_relation_attention=False,
        use_dense_residual=False,
    ):
        super().__init__()
        self.num_balls = int(num_balls)
        self.horizon = int(horizon)
        self.action_scale = float(action_scale)
        self.diffusion_steps = int(diffusion_steps)
        self.chunk_dim = self.horizon * self.num_balls * 2
        self.hidden_dim = int(hidden_dim)
        self.state_encoder = HRSGAStateEncoder(
            robot_dim=robot_dim,
            task_dim=task_dim,
            obstacle_dim=obstacle_dim,
            rr_edge_dim=rr_edge_dim,
            tr_edge_dim=tr_edge_dim,
            or_edge_dim=or_edge_dim,
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
        self.action_encoder = DiffusionMLP(self.chunk_dim, [hidden_dim * 2, hidden_dim], hidden_dim)
        self.time_encoder = DiffusionMLP(hidden_dim, [hidden_dim], hidden_dim)
        self.noise_head = DiffusionMLP(hidden_dim * 4, [hidden_dim * 4, hidden_dim * 2], self.chunk_dim)

        betas = torch.linspace(1e-4, 2e-2, self.diffusion_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))

    def q_sample(self, clean_actions, timesteps, noise):
        scale_clean = self.sqrt_alpha_bars[timesteps].view(-1, 1, 1, 1)
        scale_noise = self.sqrt_one_minus_alpha_bars[timesteps].view(-1, 1, 1, 1)
        return scale_clean * clean_actions + scale_noise * noise

    def forward(
        self,
        robot_features,
        task_features,
        obstacle_features,
        rr_edges,
        tr_edges,
        or_edges,
        rr_mask,
        tr_mask,
        or_mask,
        active_mask,
        timesteps,
        noisy_action_chunk,
    ):
        fused = self.state_encoder(robot_features, task_features, obstacle_features, rr_edges, tr_edges, or_edges, rr_mask, tr_mask, or_mask)
        active = active_mask.unsqueeze(-1).float()
        denom = active.sum(dim=1).clamp_min(1.0)
        pooled_mean = (fused * active).sum(dim=1) / denom
        masked_fused = fused.masked_fill(~active_mask.unsqueeze(-1).bool(), -1e9)
        pooled_max = masked_fused.max(dim=1).values
        pooled_max = torch.where(torch.isfinite(pooled_max), pooled_max, torch.zeros_like(pooled_mean))
        time_hidden = self.time_encoder(sinusoidal_embedding(timesteps, self.hidden_dim, fused.device))
        action_hidden = self.action_encoder(noisy_action_chunk.reshape(noisy_action_chunk.shape[0], -1))
        pred_noise = self.noise_head(torch.cat([pooled_mean, pooled_max, time_hidden, action_hidden], dim=-1))
        pred_noise = pred_noise.view(-1, self.horizon, self.num_balls, 2)
        return pred_noise * active_mask.unsqueeze(1).unsqueeze(-1).float()


class HRSGADiffusionPoilcyAgent:
    def __init__(
        self,
        num_balls,
        max_steps,
        num_tasks=None,
        horizon=4,
        hidden_dim=128,
        num_heads=4,
        topk_robot=2,
        topk_task=2,
        topk_obstacle=1,
        action_scale=1.4,
        diffusion_steps=16,
        disable_temporal_bias=False,
        disable_geometric_bias=False,
        shared_relation_attention=False,
        use_dense_residual=False,
        device=None,
    ):
        self.num_balls = int(num_balls)
        self.max_steps = int(max_steps)
        self.num_tasks = None if num_tasks is None else int(num_tasks)
        self.horizon = int(horizon)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.topk_robot = int(topk_robot)
        self.topk_task = int(topk_task)
        self.topk_obstacle = int(topk_obstacle)
        self.action_scale = float(action_scale)
        self.diffusion_steps = int(diffusion_steps)
        self.disable_temporal_bias = bool(disable_temporal_bias)
        self.disable_geometric_bias = bool(disable_geometric_bias)
        self.shared_relation_attention = bool(shared_relation_attention)
        self.use_dense_residual = bool(use_dense_residual)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.guidance_controller = HRSGAExpertController(
            topk_robot=self.topk_robot,
            topk_task=self.topk_task,
            topk_obstacle=self.topk_obstacle,
        )
        self.base_guidance_mix = 0.12
        self.risk_guidance_mix = 0.42
        self.misalignment_guidance_mix = 0.22
        self.idle_guidance_mix = 0.20
        self.warm_start_noise_scale = 0.03
        self.rollout_dt = 0.12
        self.rollout_damping = 0.98
        self.rollout_max_speed = 1.25
        self.obstacle_safety_margin = 0.24
        self.agent_safety_margin = 0.18
        self.cost_guidance_scale = 0.18
        self.cost_guidance_task_weight = 1.0
        self.cost_guidance_obstacle_weight = 1.8
        self.cost_guidance_agent_weight = 1.6
        self.cost_guidance_smoothness_weight = 0.08
        self._cached_action_chunk = None
        self._cached_step = None
        self.model = HRSGADiffusionPoilcyNetwork(
            num_balls=self.num_balls,
            horizon=self.horizon,
            robot_dim=13,
            task_dim=14,
            obstacle_dim=3,
            rr_edge_dim=8,
            tr_edge_dim=14,
            or_edge_dim=5,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            topk_robot=self.topk_robot,
            topk_task=self.topk_task,
            topk_obstacle=self.topk_obstacle,
            action_scale=self.action_scale,
            diffusion_steps=self.diffusion_steps,
            disable_temporal_bias=self.disable_temporal_bias,
            disable_geometric_bias=self.disable_geometric_bias,
            shared_relation_attention=self.shared_relation_attention,
            use_dense_residual=self.use_dense_residual,
        ).to(self.device)

    def _to_tensor(self, value):
        if isinstance(value, torch.Tensor):
            return value.to(self.device)
        if value.dtype == np.bool_:
            return torch.from_numpy(value).to(device=self.device, dtype=torch.bool)
        return torch.from_numpy(value).to(device=self.device, dtype=torch.float32)

    def _pad_snapshot_batch_to_teacher_shape(self, batch):
        target_agents = self.num_balls
        current_agents = int(batch["robot_features"].shape[1])
        if current_agents == target_agents:
            return batch, current_agents
        if current_agents > target_agents:
            raise ValueError(
                f"Teacher was trained with num_balls={target_agents}, but received {current_agents} agents."
            )

        padded = {}
        for key, value in batch.items():
            if key == "robot_features":
                pad = np.zeros((value.shape[0], target_agents, value.shape[2]), dtype=value.dtype)
                pad[:, :current_agents, :] = value
                padded[key] = pad
            elif key == "rr_edges":
                pad = np.zeros((value.shape[0], target_agents, target_agents, value.shape[3]), dtype=value.dtype)
                pad[:, :current_agents, :current_agents, :] = value
                padded[key] = pad
            elif key == "rr_mask":
                pad = np.zeros((value.shape[0], target_agents, target_agents), dtype=value.dtype)
                pad[:, :current_agents, :current_agents] = value
                padded[key] = pad
            elif key in ("tr_edges", "or_edges"):
                pad = np.zeros((value.shape[0], target_agents, value.shape[2], value.shape[3]), dtype=value.dtype)
                pad[:, :current_agents, :, :] = value
                padded[key] = pad
            elif key in ("tr_mask", "or_mask"):
                pad = np.zeros((value.shape[0], target_agents, value.shape[2]), dtype=value.dtype)
                pad[:, :current_agents, :] = value
                padded[key] = pad
            elif key == "active_mask":
                pad = np.zeros((value.shape[0], target_agents), dtype=value.dtype)
                pad[:, :current_agents] = value
                padded[key] = pad
            elif key == "action_targets":
                pad = np.zeros((value.shape[0], target_agents, value.shape[2]), dtype=value.dtype)
                pad[:, :current_agents, :] = value
                padded[key] = pad
            elif key == "action_chunk_targets":
                pad = np.zeros((value.shape[0], value.shape[1], target_agents, value.shape[3]), dtype=value.dtype)
                pad[:, :, :current_agents, :] = value
                padded[key] = pad
            else:
                padded[key] = value
        return padded, current_agents

    def diffusion_loss(self, batch):
        clean_chunk = self._to_tensor(batch["action_chunk_targets"])
        active_mask = self._to_tensor(batch["active_mask"])
        batch_size = clean_chunk.shape[0]
        timesteps = torch.randint(0, self.diffusion_steps, (batch_size,), device=self.device, dtype=torch.long)
        noise = torch.randn_like(clean_chunk)
        noisy_chunk = self.model.q_sample(clean_chunk, timesteps, noise)
        pred_noise = self.model(
            self._to_tensor(batch["robot_features"]),
            self._to_tensor(batch["task_features"]),
            self._to_tensor(batch["obstacle_features"]),
            self._to_tensor(batch["rr_edges"]),
            self._to_tensor(batch["tr_edges"]),
            self._to_tensor(batch["or_edges"]),
            self._to_tensor(batch["rr_mask"]),
            self._to_tensor(batch["tr_mask"]),
            self._to_tensor(batch["or_mask"]),
            active_mask,
            timesteps,
            noisy_chunk,
        )
        weight = active_mask.unsqueeze(1).unsqueeze(-1).float()
        denom = weight.sum().clamp_min(1.0) * clean_chunk.shape[1] * clean_chunk.shape[-1]
        return (((pred_noise - noise) ** 2) * weight).sum() / denom

    def _clip_action_norm_torch(self, actions):
        norms = torch.linalg.norm(actions, dim=-1, keepdim=True).clamp_min(1e-6)
        scales = torch.clamp(self.action_scale / norms, max=1.0)
        return actions * scales

    def _build_cost_guidance_state(self, snapshot, active_mask):
        current_positions = np.zeros((1, self.num_balls, 2), dtype=np.float32)
        current_velocities = np.zeros((1, self.num_balls, 2), dtype=np.float32)
        target_positions = np.zeros((1, self.num_balls, 2), dtype=np.float32)
        target_mask = np.zeros((1, self.num_balls), dtype=np.float32)
        agents = snapshot["agents"]
        tasks = snapshot["tasks"]
        for agent_index in range(min(len(agents), self.num_balls)):
            agent = agents[agent_index]
            current_positions[0, agent_index] = np.asarray(agent["pos"], dtype=np.float32)
            current_velocities[0, agent_index] = np.asarray(agent["vel"], dtype=np.float32)
            task_index = int(agent.get("service_task_id", -1))
            if task_index < 0:
                task_index = int(agent.get("task_id", -1))
            if 0 <= task_index < len(tasks) and not tasks[task_index].get("completed", False):
                target_positions[0, agent_index] = np.asarray(tasks[task_index]["pos"], dtype=np.float32)
                target_mask[0, agent_index] = 1.0
            else:
                target_positions[0, agent_index] = current_positions[0, agent_index]

        obstacle_centers = np.asarray([obstacle["center"] for obstacle in snapshot["obstacles"]], dtype=np.float32)
        obstacle_radii = np.asarray([obstacle["radius"] for obstacle in snapshot["obstacles"]], dtype=np.float32)
        return {
            "current_positions": torch.from_numpy(current_positions).to(self.device),
            "current_velocities": torch.from_numpy(current_velocities).to(self.device),
            "target_positions": torch.from_numpy(target_positions).to(self.device),
            "target_mask": torch.from_numpy(target_mask).to(self.device),
            "obstacle_centers": torch.from_numpy(obstacle_centers).to(self.device) if obstacle_centers.size else torch.zeros((0, 2), device=self.device),
            "obstacle_radii": torch.from_numpy(obstacle_radii).to(self.device) if obstacle_radii.size else torch.zeros((0,), device=self.device),
            "ball_radius": float(snapshot.get("ball_radius", 0.18)),
            "active_mask": active_mask.float(),
        }

    def _rollout_chunk_positions(self, action_chunk, guidance_state):
        positions = guidance_state["current_positions"]
        velocities = guidance_state["current_velocities"]
        active = guidance_state["active_mask"].unsqueeze(-1)
        trajectory = []
        for horizon_index in range(action_chunk.shape[1]):
            accel = self._clip_action_norm_torch(action_chunk[:, horizon_index]) * active
            velocities = velocities + accel * self.rollout_dt
            vel_norms = torch.linalg.norm(velocities, dim=-1, keepdim=True).clamp_min(1e-6)
            velocities = velocities * torch.clamp(self.rollout_max_speed / vel_norms, max=1.0)
            velocities = velocities * self.rollout_damping * active
            positions = positions + velocities * self.rollout_dt
            trajectory.append(positions)
        return torch.stack(trajectory, dim=1)

    def _trajectory_guidance_cost(self, action_chunk, guidance_state):
        trajectory = self._rollout_chunk_positions(action_chunk, guidance_state)
        active = guidance_state["active_mask"]
        target_mask = guidance_state["target_mask"] * active
        time_weights = torch.linspace(0.7, 1.3, steps=self.horizon, device=self.device).view(1, self.horizon, 1)

        task_delta = trajectory - guidance_state["target_positions"].unsqueeze(1)
        task_dist = torch.linalg.norm(task_delta, dim=-1)
        task_cost = ((task_dist * time_weights) * target_mask.unsqueeze(1)).sum() / target_mask.sum().clamp_min(1.0)

        obstacle_cost = torch.tensor(0.0, device=self.device)
        if guidance_state["obstacle_centers"].shape[0] > 0:
            obstacle_delta = trajectory.unsqueeze(3) - guidance_state["obstacle_centers"].view(1, 1, 1, -1, 2)
            obstacle_dist = torch.linalg.norm(obstacle_delta, dim=-1)
            safe_clearance = guidance_state["obstacle_radii"].view(1, 1, 1, -1) + guidance_state["ball_radius"] + self.obstacle_safety_margin
            obstacle_penalty = torch.relu(safe_clearance - obstacle_dist) ** 2
            obstacle_cost = (obstacle_penalty * active.unsqueeze(1).unsqueeze(-1)).sum() / active.sum().clamp_min(1.0)

        agent_cost = torch.tensor(0.0, device=self.device)
        if self.num_balls > 1:
            pair_delta = trajectory.unsqueeze(3) - trajectory.unsqueeze(2)
            pair_dist = torch.linalg.norm(pair_delta, dim=-1)
            active_pairs = active.unsqueeze(1).unsqueeze(-1) * active.unsqueeze(1).unsqueeze(-2)
            upper_mask = torch.triu(torch.ones((self.num_balls, self.num_balls), device=self.device), diagonal=1)
            safe_pair_dist = 2.0 * guidance_state["ball_radius"] + self.agent_safety_margin
            pair_penalty = torch.relu(safe_pair_dist - pair_dist) ** 2
            agent_cost = (pair_penalty * active_pairs * upper_mask.view(1, 1, self.num_balls, self.num_balls)).sum() / upper_mask.sum().clamp_min(1.0)

        smoothness_cost = torch.tensor(0.0, device=self.device)
        if self.horizon > 1:
            smoothness_cost = ((action_chunk[:, 1:] - action_chunk[:, :-1]) ** 2).mean()

        return (
            self.cost_guidance_task_weight * task_cost
            + self.cost_guidance_obstacle_weight * obstacle_cost
            + self.cost_guidance_agent_weight * agent_cost
            + self.cost_guidance_smoothness_weight * smoothness_cost
        )

    def _apply_cost_guidance(self, chunk, guidance_state, step):
        if guidance_state is None or self.cost_guidance_scale <= 0.0:
            return chunk
        progress = 1.0 - (float(step) / max(1.0, float(self.diffusion_steps - 1)))
        guidance_strength = self.cost_guidance_scale * (0.35 + 0.65 * progress)
        guided_chunk = chunk.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            guidance_cost = self._trajectory_guidance_cost(guided_chunk, guidance_state)
        gradient = torch.autograd.grad(guidance_cost, guided_chunk, allow_unused=False)[0]
        guided_chunk = (guided_chunk - guidance_strength * gradient).detach()
        guided_chunk = self._clip_action_norm_torch(guided_chunk)
        return guided_chunk * guidance_state["active_mask"].unsqueeze(1).unsqueeze(-1)

    def sample_action_chunk_batch(self, batch, temperature=0.0, initial_chunk=None, guidance_state=None):
        active_mask = self._to_tensor(batch["active_mask"])
        batch_size = active_mask.shape[0]
        if initial_chunk is not None:
            chunk = self._to_tensor(initial_chunk).clone()
            if temperature > 0.0:
                noise_scale = float(max(temperature, self.warm_start_noise_scale))
                chunk = chunk + torch.randn_like(chunk) * noise_scale
        elif temperature > 0.0:
            chunk = torch.randn(batch_size, self.horizon, self.num_balls, 2, device=self.device) * float(temperature)
        else:
            chunk = torch.zeros(batch_size, self.horizon, self.num_balls, 2, device=self.device)

        for step in reversed(range(self.diffusion_steps)):
            timesteps = torch.full((batch_size,), step, device=self.device, dtype=torch.long)
            pred_noise = self.model(
                self._to_tensor(batch["robot_features"]),
                self._to_tensor(batch["task_features"]),
                self._to_tensor(batch["obstacle_features"]),
                self._to_tensor(batch["rr_edges"]),
                self._to_tensor(batch["tr_edges"]),
                self._to_tensor(batch["or_edges"]),
                self._to_tensor(batch["rr_mask"]),
                self._to_tensor(batch["tr_mask"]),
                self._to_tensor(batch["or_mask"]),
                active_mask,
                timesteps,
                chunk,
            )
            alpha = self.model.alphas[step]
            alpha_bar = self.model.alpha_bars[step]
            beta = self.model.betas[step]
            chunk = (chunk - ((1.0 - alpha) / torch.sqrt(1.0 - alpha_bar)) * pred_noise) / torch.sqrt(alpha)
            chunk = self._apply_cost_guidance(chunk, guidance_state, step)
            if step > 0 and temperature > 0.0:
                chunk = chunk + torch.sqrt(beta) * torch.randn_like(chunk) * float(temperature)

        chunk = torch.clamp(chunk, -self.action_scale, self.action_scale)
        return chunk * active_mask.unsqueeze(1).unsqueeze(-1).float()

    def _reset_inference_cache_if_needed(self, snapshot):
        current_step = int(snapshot.get("step", 0))
        if self._cached_step is None or current_step <= 0 or current_step <= self._cached_step:
            self._cached_action_chunk = None
        self._cached_step = current_step

    def _build_initial_chunk(self, active_mask):
        if self._cached_action_chunk is None:
            return None
        cached = np.asarray(self._cached_action_chunk, dtype=np.float32)
        if cached.shape != (self.horizon, self.num_balls, 2):
            return None
        shifted = np.zeros_like(cached)
        shifted[:-1] = cached[1:]
        shifted[-1] = cached[-1]
        shifted *= active_mask[0][None, :, None].astype(np.float32)
        return shifted[None, ...]

    def _target_direction(self, snapshot, agent_index):
        agent = snapshot["agents"][agent_index]
        task_index = int(agent.get("service_task_id", -1))
        if task_index < 0:
            task_index = int(agent.get("task_id", -1))
        if 0 <= task_index < len(snapshot["tasks"]):
            task = snapshot["tasks"][task_index]
            if not task.get("completed", False):
                direction = np.asarray(task["pos"], dtype=np.float32) - np.asarray(agent["pos"], dtype=np.float32)
                norm = float(np.linalg.norm(direction))
                if norm > 1e-6:
                    return direction / norm
        return None

    def _local_risk(self, snapshot, agent_index):
        agent = snapshot["agents"][agent_index]
        agent_pos = np.asarray(agent["pos"], dtype=np.float32)
        ball_radius = float(snapshot.get("ball_radius", 0.18))
        obstacle_risk = 0.0
        for obstacle in snapshot["obstacles"]:
            center = np.asarray(obstacle["center"], dtype=np.float32)
            clearance = float(np.linalg.norm(agent_pos - center) - obstacle["radius"] - ball_radius)
            obstacle_risk = max(obstacle_risk, max(0.0, 0.85 - clearance) / 0.85)
        agent_risk = 0.0
        for other_index, other in enumerate(snapshot["agents"]):
            if other_index == agent_index:
                continue
            other_pos = np.asarray(other["pos"], dtype=np.float32)
            clearance = float(np.linalg.norm(agent_pos - other_pos) - 2.0 * ball_radius)
            agent_risk = max(agent_risk, max(0.0, 0.65 - clearance) / 0.65)
        return float(np.clip(max(obstacle_risk, agent_risk), 0.0, 1.0))

    def _blend_with_guidance(self, snapshot, sampled_action):
        guidance_actions = np.asarray(self.guidance_controller.act(snapshot), dtype=np.float32)
        blended = np.asarray(sampled_action, dtype=np.float32).copy()
        for agent_index, agent in enumerate(snapshot["agents"]):
            if agent.get("reached", False) or agent.get("is_servicing", False):
                blended[agent_index] = np.zeros(2, dtype=np.float32)
                continue
            target_direction = self._target_direction(snapshot, agent_index)
            risk = self._local_risk(snapshot, agent_index)
            sample_vec = blended[agent_index]
            guidance_vec = guidance_actions[agent_index]
            mix = self.base_guidance_mix + self.risk_guidance_mix * risk
            if target_direction is not None:
                sample_norm = float(np.linalg.norm(sample_vec))
                if sample_norm > 1e-6:
                    alignment = float(np.dot(sample_vec / sample_norm, target_direction))
                    if alignment < 0.2:
                        mix += self.misalignment_guidance_mix * (0.2 - alignment) / 1.2
                elif np.linalg.norm(guidance_vec) > 1e-6:
                    mix += self.idle_guidance_mix
            mix = float(np.clip(mix, 0.0, 0.72))
            blended[agent_index] = (1.0 - mix) * sample_vec + mix * guidance_vec
        norms = np.linalg.norm(blended, axis=1, keepdims=True)
        scales = np.where(norms > self.action_scale, self.action_scale / np.maximum(norms, 1e-6), 1.0)
        return (blended * scales).astype(np.float32)

    def _apply_safety_projection(self, snapshot, actions):
        projected = np.asarray(actions, dtype=np.float32).copy()
        ball_radius = float(snapshot.get("ball_radius", 0.18))
        dt = 0.12
        for agent_index, agent in enumerate(snapshot["agents"]):
            if agent.get("reached", False) or agent.get("is_servicing", False):
                projected[agent_index] = np.zeros(2, dtype=np.float32)
                continue
            position = np.asarray(agent["pos"], dtype=np.float32)
            safety_push = np.zeros(2, dtype=np.float32)
            for other_index, other in enumerate(snapshot["agents"]):
                if other_index == agent_index:
                    continue
                other_pos = np.asarray(other["pos"], dtype=np.float32)
                diff = position - other_pos
                dist = float(np.linalg.norm(diff))
                safe_dist = 2.0 * ball_radius + self.agent_safety_margin
                if dist < safe_dist:
                    direction = diff / max(dist, 1e-6)
                    safety_push += direction * ((safe_dist - dist) / safe_dist)
            for obstacle in snapshot["obstacles"]:
                center = np.asarray(obstacle["center"], dtype=np.float32)
                diff = position - center
                dist = float(np.linalg.norm(diff))
                safe_dist = obstacle["radius"] + ball_radius + self.obstacle_safety_margin
                if dist < safe_dist:
                    direction = diff / max(dist, 1e-6)
                    safety_push += 1.35 * direction * ((safe_dist - dist) / max(safe_dist, 1e-6))
            candidate = projected[agent_index] + safety_push / max(dt, 1e-6)
            norm = float(np.linalg.norm(candidate))
            if norm > self.action_scale:
                candidate = candidate * (self.action_scale / norm)
            projected[agent_index] = candidate.astype(np.float32)
        return projected

    def select_action(self, snapshot, temperature=0.0):
        self._reset_inference_cache_if_needed(snapshot)
        sample = snapshot_to_sample(
            snapshot,
            max_steps=self.max_steps,
            action_targets=np.zeros((len(snapshot["agents"]), 2), dtype=np.float32),
        )
        batch = {key: value[None, ...] for key, value in sample.items() if key != "action_targets"}
        batch, original_agent_count = self._pad_snapshot_batch_to_teacher_shape(batch)
        initial_chunk = self._build_initial_chunk(batch["active_mask"])
        guidance_state = self._build_cost_guidance_state(snapshot, self._to_tensor(batch["active_mask"]))
        self.model.eval()
        with torch.no_grad():
            action_chunk = self.sample_action_chunk_batch(
                batch,
                temperature=temperature,
                initial_chunk=initial_chunk,
                guidance_state=guidance_state,
            )
        action_chunk_np = action_chunk[0].detach().cpu().numpy().astype(np.float32)
        self._cached_action_chunk = action_chunk_np
        sampled_action = action_chunk_np[0, :original_agent_count]
        guided_action = self._blend_with_guidance(snapshot, sampled_action)
        return self._apply_safety_projection(snapshot, guided_action)

    def save(self, path, epoch=0, best_score=-np.inf, optimizer=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            "model_state": self.model.state_dict(),
            "epoch": int(epoch),
            "best_score": float(best_score),
            "num_balls": self.num_balls,
            "num_tasks": self.num_tasks,
            "max_steps": self.max_steps,
            "horizon": self.horizon,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "topk_robot": self.topk_robot,
            "topk_task": self.topk_task,
            "topk_obstacle": self.topk_obstacle,
            "action_scale": self.action_scale,
            "diffusion_steps": self.diffusion_steps,
            "disable_temporal_bias": self.disable_temporal_bias,
            "disable_geometric_bias": self.disable_geometric_bias,
            "shared_relation_attention": self.shared_relation_attention,
            "use_dense_residual": self.use_dense_residual,
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


def load_diffusion_poilcy_from_checkpoint(path, device=None):
    checkpoint = torch.load(path, map_location=device or ("cuda" if torch.cuda.is_available() else "cpu"))
    agent = HRSGADiffusionPoilcyAgent(
        num_balls=int(checkpoint["num_balls"]),
        num_tasks=checkpoint.get("num_tasks"),
        max_steps=int(checkpoint["max_steps"]),
        horizon=int(checkpoint.get("horizon", 4)),
        hidden_dim=int(checkpoint.get("hidden_dim", 128)),
        num_heads=int(checkpoint.get("num_heads", 4)),
        topk_robot=int(checkpoint.get("topk_robot", 2)),
        topk_task=int(checkpoint.get("topk_task", 2)),
        topk_obstacle=int(checkpoint.get("topk_obstacle", 1)),
        action_scale=float(checkpoint.get("action_scale", 1.4)),
        diffusion_steps=int(checkpoint.get("diffusion_steps", 16)),
        disable_temporal_bias=bool(checkpoint.get("disable_temporal_bias", False)),
        disable_geometric_bias=bool(checkpoint.get("disable_geometric_bias", False)),
        shared_relation_attention=bool(checkpoint.get("shared_relation_attention", False)),
        use_dense_residual=bool(checkpoint.get("use_dense_residual", False)),
        device=device,
    )
    adapted_state = _adapt_state_dict_for_model(agent.model.state_dict(), checkpoint["model_state"])
    agent.model.load_state_dict(adapted_state)
    agent.model.eval()
    return agent


def evaluate_diffusion_poilcy(teacher, num_episodes=10, num_balls=None, num_tasks=None, max_steps=None, base_seed=5300, layout_mode="structured", strict_collision_stop=False, temperature=0.0):
    num_balls = int(teacher.num_balls if num_balls is None else num_balls)
    resolved_num_tasks = teacher.num_tasks if num_tasks is None else num_tasks
    resolved_num_tasks = int(max(num_balls, resolved_num_tasks if resolved_num_tasks is not None else num_balls * 2))
    max_steps = int(teacher.max_steps if max_steps is None else max_steps)
    returns = {
        "success_rate": 0.0,
        "avg_completed_fraction": 0.0,
        "avg_deadline_satisfaction": 0.0,
        "avg_collisions": 0.0,
        "avg_mean_task_distance": 0.0,
        "avg_min_pair_distance": 0.0,
        "avg_missed_deadlines": 0.0,
        "avg_steps": 0.0,
    }
    runs = []
    for episode_index in range(num_episodes):
        env = HRSGABallEnvironment(num_balls=num_balls, num_tasks=resolved_num_tasks, max_steps=max_steps)
        snapshot = env.reset(seed=base_seed + episode_index, layout_mode=layout_mode)
        collision_terminated = False
        while True:
            actions = teacher.select_action(snapshot, temperature=temperature)
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


__all__ = [
    "DiffusionDatasetSplit",
    "HRSGADiffusionPoilcyAgent",
    "HRSGADiffusionPoilcyNetwork",
    "build_diffusion_behavior_dataset",
    "evaluate_diffusion_poilcy",
    "iterate_diffusion_minibatches",
    "load_diffusion_poilcy_from_checkpoint",
    "_split_diffusion_dataset",
]