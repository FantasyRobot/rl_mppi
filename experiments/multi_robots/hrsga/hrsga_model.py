import math
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

try:
    from .env_hrsga import HRSGAEnvironment, pairwise_distance, vec_add, vec_clip_norm, vec_mul, vec_norm, vec_sub, vec_unit
except ImportError:
    from env_hrsga import HRSGAEnvironment, pairwise_distance, vec_add, vec_clip_norm, vec_mul, vec_norm, vec_sub, vec_unit


def softmax_scores(scores):
    if not scores:
        return []
    max_score = max(scores)
    exps = [math.exp(score - max_score) for score in scores]
    total = sum(exps)
    if total < 1e-8:
        return [1.0 / len(scores)] * len(scores)
    return [value / total for value in exps]


def topk_items(items, scores, k):
    ranked = sorted(zip(items, scores), key=lambda pair: pair[1], reverse=True)
    selected = ranked[:k]
    return [item for item, _ in selected], [score for _, score in selected]


class HRSGAExpertController:
    name = "hrsga_expert"

    def __init__(self, robot_radius=3.4, task_radius=9.0, obstacle_radius=2.8, topk_robot=2, topk_task=2, topk_obstacle=1):
        self.robot_radius = robot_radius
        self.task_radius = task_radius
        self.obstacle_radius = obstacle_radius
        self.topk_robot = topk_robot
        self.topk_task = topk_task
        self.topk_obstacle = topk_obstacle
        self.coordination_step_radius = 0.18

    def _task_is_assignable(self, task, agent_index):
        if task["completed"]:
            return False
        servicing_agent = int(task.get("servicing_agent", -1))
        if servicing_agent >= 0 and servicing_agent != agent_index:
            return False
        assigned_agent = int(task.get("assigned_agent", -1))
        is_available = bool(task.get("is_available", True))
        return is_available or servicing_agent == agent_index or assigned_agent == agent_index

    def _rr_score(self, agent, other):
        rel = vec_sub(other["pos"], agent["pos"])
        dist = max(1e-6, vec_norm(rel))
        rel_dir = vec_unit(rel)
        rel_vel = vec_sub(other["vel"], agent["vel"])
        approach_rate = max(0.0, -(rel_vel[0] * rel_dir[0] + rel_vel[1] * rel_dir[1]))
        own_x_time = abs(agent["pos"][0]) / max(0.25, abs(agent["vel"][0]) + 0.25)
        other_x_time = abs(other["pos"][0]) / max(0.25, abs(other["vel"][0]) + 0.25)
        conflict_gap = abs(own_x_time - other_x_time)
        overlap_bias = 1.0 / (0.5 + conflict_gap)
        return 2.8 / (dist + 0.2) + 1.2 * approach_rate + 0.9 * overlap_bias

    def _rr_message(self, agent, other):
        rel = vec_sub(agent["pos"], other["pos"])
        dist = max(1e-6, vec_norm(rel))
        away = vec_unit(rel)
        tangent = (-away[1], away[0])
        if tangent[1] * away[0] - tangent[0] * away[1] < 0.0:
            tangent = (-tangent[0], -tangent[1])
        rel_vel = vec_sub(other["vel"], agent["vel"])
        closing = max(0.0, -(rel_vel[0] * away[0] + rel_vel[1] * away[1]))
        strength = max(0.0, self.robot_radius - dist) / self.robot_radius
        return vec_add(vec_mul(away, strength * (1.1 + 0.95 * closing)), vec_mul(tangent, 0.55 * strength))

    def _tr_score(self, snapshot, agent_index, agent, task):
        if int(task.get("servicing_agent", -1)) == agent_index:
            return 100.0
        if not self._task_is_assignable(task, agent_index):
            return -1e6
        rel = vec_sub(task["pos"], agent["pos"])
        dist = max(1e-6, vec_norm(rel))
        release_slack = task["release_step"] - snapshot["step"]
        deadline_slack = task["deadline_step"] - snapshot["step"]
        assigned = 1.0 if task.get("assigned_agent", -1) == agent_index else 0.0
        rank_bonus = 1.0 - float(task.get("visit_rank", len(snapshot["tasks"]))) / max(1.0, float(len(snapshot["tasks"])))
        predecessor_pending = float(task.get("predecessor_pending", 0.0))
        is_available = float(task.get("is_available", True))
        release_bias = 1.1 if release_slack <= 0 and is_available > 0.5 else 0.35 / (1.0 + 0.10 * max(0, release_slack) + 0.85 * predecessor_pending)
        deadline_bias = 1.6 / max(6.0, deadline_slack + 6.0)
        feasibility = 1.2 / (dist + 0.35)
        availability_bonus = 0.6 * is_available
        return 2.4 * assigned + release_bias + 10.0 * deadline_bias + feasibility + availability_bonus + 0.35 * rank_bonus + 0.15 * task["priority"]

    def _tr_message(self, snapshot, agent_index, agent, task):
        if int(task.get("servicing_agent", -1)) == agent_index:
            return (0.0, 0.0)
        if not self._task_is_assignable(task, agent_index):
            return (0.0, 0.0)
        rel = vec_sub(task["pos"], agent["pos"])
        dist = max(1e-6, vec_norm(rel))
        direction = vec_unit(rel)
        assigned = 1.0 if task.get("assigned_agent", -1) == agent_index else 0.35
        release_wait = max(0, task["release_step"] - snapshot["step"])
        predecessor_pending = float(task.get("predecessor_pending", 0.0))
        is_available = bool(task.get("is_available", True))
        release_gate = 1.0 if is_available and release_wait == 0 else 0.42 / (1.0 + 0.08 * release_wait + 0.65 * predecessor_pending)
        deadline_slack = max(1, task["deadline_step"] - snapshot["step"])
        urgency = min(1.0, 18.0 / deadline_slack)
        rank_gate = 1.15 - float(task.get("visit_rank", len(snapshot["tasks"]))) / max(1.0, 2.0 * float(len(snapshot["tasks"])))
        strength = assigned * release_gate * task["priority"] * rank_gate * (1.8 + 0.85 * urgency)
        return vec_mul(direction, strength / (0.2 + 0.45 * dist))

    def _self_task_force(self, snapshot, agent_index, agent):
        if agent.get("is_servicing", False):
            return (0.0, 0.0)
        task_index = int(agent.get("task_id", -1))
        if task_index < 0 or task_index >= len(snapshot["tasks"]):
            return (0.0, 0.0)
        task = snapshot["tasks"][task_index]
        if task["completed"] or not self._task_is_assignable(task, agent_index):
            return (0.0, 0.0)
        rel = vec_sub(task["pos"], agent["pos"])
        direction = vec_unit(rel)
        release_wait = max(0, task["release_step"] - snapshot["step"])
        predecessor_pending = float(task.get("predecessor_pending", 0.0))
        is_available = bool(task.get("is_available", True))
        release_gate = 1.0 if is_available and release_wait == 0 else 0.55 / (1.0 + 0.08 * release_wait + 0.70 * predecessor_pending)
        urgency = min(1.0, 22.0 / max(8.0, task["deadline_step"] - snapshot["step"] + 8.0))
        return vec_mul(direction, release_gate * (0.95 + 0.65 * urgency))

    def _or_score(self, snapshot, agent, obstacle):
        rel = vec_sub(agent["pos"], obstacle["center"])
        dist = max(1e-6, vec_norm(rel))
        clearance = dist - (obstacle["radius"] + snapshot["robot_radius"])
        return 2.5 / (clearance + 0.35)

    def _or_message(self, snapshot, agent, obstacle):
        rel = vec_sub(agent["pos"], obstacle["center"])
        dist = max(1e-6, vec_norm(rel))
        clearance = obstacle["radius"] + snapshot["robot_radius"] + 0.9
        direction = vec_unit(rel)
        tangent = (-direction[1], direction[0])
        risk = max(0.0, clearance - dist) / clearance
        return vec_add(vec_mul(direction, 1.75 * risk), vec_mul(tangent, 0.32 * risk))

    def _pool(self, items, scores, message_fn, k):
        if not items:
            return (0.0, 0.0), 0.0
        selected_items, selected_scores = topk_items(items, scores, k)
        weights = softmax_scores(selected_scores)
        pooled = (0.0, 0.0)
        for weight, item in zip(weights, selected_items):
            pooled = vec_add(pooled, vec_mul(message_fn(item), weight))
        risk = sum(weight * score for weight, score in zip(weights, selected_scores))
        return pooled, risk

    def act(self, snapshot):
        targets = []
        for agent_index, agent in enumerate(snapshot["agents"]):
            if agent["reached"] or agent.get("is_servicing", False):
                targets.append(_reference_target_point(snapshot, agent_index))
                continue

            robot_neighbors = []
            robot_scores = []
            for other_index, other in enumerate(snapshot["agents"]):
                if other_index == agent_index or other["reached"]:
                    continue
                if pairwise_distance(agent["pos"], other["pos"]) <= self.robot_radius:
                    robot_neighbors.append(other)
                    robot_scores.append(self._rr_score(agent, other))

            task_neighbors = []
            task_scores = []
            for task in snapshot["tasks"]:
                if task["completed"] or not self._task_is_assignable(task, agent_index):
                    continue
                if pairwise_distance(agent["pos"], task["pos"]) <= self.task_radius:
                    task_neighbors.append(task)
                    task_scores.append(self._tr_score(snapshot, agent_index, agent, task))

            obstacle_neighbors = []
            obstacle_scores = []
            for obstacle in snapshot["obstacles"]:
                if pairwise_distance(agent["pos"], obstacle["center"]) <= self.obstacle_radius + obstacle["radius"]:
                    obstacle_neighbors.append(obstacle)
                    obstacle_scores.append(self._or_score(snapshot, agent, obstacle))

            robot_force, robot_risk = self._pool(robot_neighbors, robot_scores, lambda other: self._rr_message(agent, other), self.topk_robot)
            task_force, task_risk = self._pool(task_neighbors, task_scores, lambda task: self._tr_message(snapshot, agent_index, agent, task), self.topk_task)
            obstacle_force, obstacle_risk = self._pool(obstacle_neighbors, obstacle_scores, lambda obstacle: self._or_message(snapshot, agent, obstacle), self.topk_obstacle)

            risk_gate = max(0.35, min(1.0, (0.4 * robot_risk + 0.35 * obstacle_risk + 0.25 * task_risk) / 5.5))
            self_task_force = self._self_task_force(snapshot, agent_index, agent)
            damping = vec_mul(agent["vel"], -0.38)
            action = vec_add(
                vec_mul(vec_add(task_force, self_task_force), 1.15 + 0.15 * risk_gate),
                vec_add(
                    vec_mul(robot_force, 1.15 + 1.15 * risk_gate),
                    vec_add(vec_mul(obstacle_force, 1.30 + 0.45 * risk_gate), damping),
                ),
            )
            if not math.isfinite(action[0]) or not math.isfinite(action[1]):
                raise ValueError("Expert controller produced a non-finite coordination action.")
            reference_target = _reference_target_point(snapshot, agent_index)
            coordination_delta = vec_clip_norm(action, self.coordination_step_radius)
            target_xy = vec_add(
                (float(reference_target[0]), float(reference_target[1])),
                coordination_delta,
            )
            targets.append((float(target_xy[0]), float(target_xy[1]), float(reference_target[2])))
        return targets


@dataclass
class DatasetSplit:
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


class StructuredReplayBuffer:
    def __init__(self, max_size):
        self.max_size = int(max_size)
        self.storage = []
        self.ptr = 0

    @property
    def size(self):
        return len(self.storage)

    def add(self, transition):
        if self.size < self.max_size:
            self.storage.append(transition)
        else:
            self.storage[self.ptr] = transition
        self.ptr = (self.ptr + 1) % self.max_size

    def sample(self, batch_size):
        if self.size < batch_size:
            raise ValueError("Not enough samples in replay buffer.")
        indices = np.random.choice(self.size, batch_size, replace=False)
        keys = self.storage[0].keys()
        batch = {}
        for key in keys:
            batch[key] = np.stack([self.storage[index][key] for index in indices], axis=0)
        return batch


def _safe_div(value, denom):
    return float(value) / float(max(denom, 1e-6))


def _signed_clip(value, scale):
    return float(np.clip(_safe_div(value, scale), -1.5, 1.5))


def _positive_clip(value, scale):
    return float(np.clip(_safe_div(value, scale), -1.0, 2.0))


def _resolve_reference_task(tasks, agents, agent_index, step):
    assigned_index = int(agents[agent_index].get("task_id", -1))
    if 0 <= assigned_index < len(tasks):
        assigned_task = tasks[assigned_index]
        if not assigned_task["completed"]:
            return assigned_index, assigned_task

    agent_pos = agents[agent_index]["pos"]

    def _select_best_task(*, allow_predecessor_pending: bool, allow_other_servicing: bool, allow_other_assigned: bool):
        fallback_index = -1
        fallback_task = None
        fallback_score = -np.inf
        for task_index, task in enumerate(tasks):
            if task["completed"]:
                continue
            predecessor_pending = float(task.get("predecessor_pending", 0.0)) > 0.5
            if predecessor_pending and not allow_predecessor_pending:
                continue
            servicing_agent = int(task.get("servicing_agent", -1))
            if servicing_agent >= 0 and servicing_agent != agent_index and not allow_other_servicing:
                continue
            assigned_agent = int(task.get("assigned_agent", -1))
            if assigned_agent >= 0 and assigned_agent != agent_index and not allow_other_assigned:
                continue
            dist = max(1e-6, vec_norm(vec_sub(task["pos"], agent_pos)))
            release_wait = max(0, task["release_step"] - step)
            deadline_slack = max(1, task["deadline_step"] - step)
            remaining_dwell = float(task.get("remaining_dwell_steps", task.get("dwell_steps", 1)))
            visit_rank = float(task.get("visit_rank", task_index + 1))
            predecessor_penalty = 0.0 if not predecessor_pending else 0.55 + 0.08 * visit_rank
            service_penalty = 0.0 if servicing_agent < 0 or servicing_agent == agent_index else 0.9
            assignment_penalty = 0.0 if assigned_agent < 0 or assigned_agent == agent_index else 0.65
            score = (
                (1.8 / (0.35 + dist))
                + (0.9 if release_wait == 0 else 0.35 / (1.0 + 0.12 * release_wait))
                + 2.0 / (6.0 + deadline_slack)
                + 0.18 / max(1.0, visit_rank)
                - 0.08 * remaining_dwell
                - predecessor_penalty
                - service_penalty
                - assignment_penalty
            )
            if score > fallback_score:
                fallback_index = task_index
                fallback_task = task
                fallback_score = score
        return fallback_index, fallback_task

    reference_index, reference_task = _select_best_task(
        allow_predecessor_pending=False,
        allow_other_servicing=False,
        allow_other_assigned=False,
    )
    if reference_task is not None:
        return reference_index, reference_task

    reference_index, reference_task = _select_best_task(
        allow_predecessor_pending=True,
        allow_other_servicing=False,
        allow_other_assigned=False,
    )
    if reference_task is not None:
        return reference_index, reference_task

    reference_index, reference_task = _select_best_task(
        allow_predecessor_pending=True,
        allow_other_servicing=False,
        allow_other_assigned=True,
    )
    if reference_task is not None:
        return reference_index, reference_task

    return _select_best_task(
        allow_predecessor_pending=True,
        allow_other_servicing=True,
        allow_other_assigned=True,
    )


def _reference_target_point(snapshot, agent_index):
    agents = snapshot["agents"]
    tasks = snapshot["tasks"]
    agent = agents[agent_index]
    goal_height = float(snapshot.get("goal_height", 0.30))

    service_task_index = int(agent.get("service_task_id", -1))
    if 0 <= service_task_index < len(tasks):
        service_task = tasks[service_task_index]
        if not service_task["completed"]:
            return (float(service_task["pos"][0]), float(service_task["pos"][1]), goal_height)

    task_index, task = _resolve_reference_task(tasks, agents, agent_index, snapshot["step"])
    if task_index >= 0 and task is not None:
        return (float(task["pos"][0]), float(task["pos"][1]), goal_height)

    if bool(agent.get("is_returning_home", False)):
        home_pos = agent.get("start_pos", agent["pos"])
        return (float(home_pos[0]), float(home_pos[1]), goal_height)
    return (float(agent["pos"][0]), float(agent["pos"][1]), goal_height)


TEMPORAL_FEATURE_SLICES = {
    "robot_features": [(6, 9)],
    "task_features": [(2, 4), (11, 12)],
    "rr_edges": [(6, 8)],
    "tr_edges": [(3, 5), (11, 12)],
}


GEOMETRIC_FEATURE_SLICES = {
    "robot_features": [(0, 6)],
    "task_features": [(0, 2)],
    "obstacle_features": [(0, 3)],
    "rr_edges": [(0, 6)],
    "tr_edges": [(0, 3)],
    "or_edges": [(0, 5)],
}


def _zero_feature_slices(array, slices):
    result = array.clone() if isinstance(array, torch.Tensor) else np.array(array, copy=True)
    for start, end in slices:
        result[..., start:end] = 0
    return result


def apply_feature_ablation(batch_or_sample, disable_temporal_bias=False, disable_geometric_bias=False):
    if not disable_temporal_bias and not disable_geometric_bias:
        return batch_or_sample

    transformed = {}
    for key, value in batch_or_sample.items():
        slices = []
        if disable_temporal_bias and key in TEMPORAL_FEATURE_SLICES:
            slices.extend(TEMPORAL_FEATURE_SLICES[key])
        if disable_geometric_bias and key in GEOMETRIC_FEATURE_SLICES:
            slices.extend(GEOMETRIC_FEATURE_SLICES[key])
        transformed[key] = _zero_feature_slices(value, slices) if slices else value
    return transformed


def snapshot_to_sample(snapshot, max_steps, action_targets):
    step = snapshot["step"]
    world_size = float(snapshot["world_size"])
    robot_radius = float(snapshot["robot_radius"])
    agents = snapshot["agents"]
    tasks = snapshot["tasks"]
    obstacles = snapshot["obstacles"]
    num_agents = len(agents)
    num_tasks = len(tasks)
    num_obstacles = len(obstacles)
    speed_scale = 1.5

    robot_features = np.zeros((num_agents, 19), dtype=np.float32)
    task_features = np.zeros((num_tasks, 14), dtype=np.float32)
    obstacle_features = np.zeros((num_obstacles, 3), dtype=np.float32)
    rr_edges = np.zeros((num_agents, num_agents, 8), dtype=np.float32)
    tr_edges = np.zeros((num_agents, num_tasks, 14), dtype=np.float32)
    or_edges = np.zeros((num_agents, num_obstacles, 5), dtype=np.float32)
    rr_mask = np.zeros((num_agents, num_agents), dtype=bool)
    tr_mask = np.zeros((num_agents, num_tasks), dtype=bool)
    or_mask = np.zeros((num_agents, num_obstacles), dtype=bool)
    active_mask = np.zeros((num_agents,), dtype=bool)

    for task_index, task in enumerate(tasks):
        release_slack = task["release_step"] - step
        deadline_slack = task["deadline_step"] - step
        nearest_agent_distance = min(pairwise_distance(agent["pos"], task["pos"]) for agent in agents) if agents else 0.0
        visit_rank = float(task.get("visit_rank", task_index + 1))
        predecessor_pending = float(task.get("predecessor_pending", 0.0))
        dwell_steps = float(task.get("dwell_steps", 1))
        remaining_dwell = float(task.get("remaining_dwell_steps", dwell_steps))
        is_servicing = float(task.get("servicing_agent", -1) >= 0)
        is_available = float(task.get("is_available", not task["completed"]))
        task_features[task_index] = np.array(
            [
                _signed_clip(task["pos"][0], world_size),
                _signed_clip(task["pos"][1], world_size),
                _signed_clip(release_slack, max_steps),
                _signed_clip(deadline_slack, max_steps),
                float(task["priority"]),
                float(task["completed"]),
                float(release_slack <= 0),
                _positive_clip(nearest_agent_distance, world_size),
                _positive_clip(visit_rank, max(1, num_tasks)),
                predecessor_pending,
                _positive_clip(dwell_steps, max_steps),
                _positive_clip(remaining_dwell, max_steps),
                is_servicing,
                is_available,
            ],
            dtype=np.float32,
        )

    for obstacle_index, obstacle in enumerate(obstacles):
        obstacle_features[obstacle_index] = np.array(
            [
                _signed_clip(obstacle["center"][0], world_size),
                _signed_clip(obstacle["center"][1], world_size),
                _positive_clip(obstacle["radius"], world_size),
            ],
            dtype=np.float32,
        )

    for agent_index, agent in enumerate(agents):
        reference_task_index, reference_task = _resolve_reference_task(tasks, agents, agent_index, step)
        if reference_task is not None:
            release_slack = reference_task["release_step"] - step
            deadline_slack = reference_task["deadline_step"] - step
            remaining_dwell = float(reference_task.get("remaining_dwell_steps", reference_task.get("dwell_steps", 1)))
            predecessor_pending = float(reference_task.get("predecessor_pending", 0.0))
            task_rel = vec_sub(reference_task["pos"], agent["pos"])
            task_distance = vec_norm(task_rel)
        else:
            release_slack = max_steps
            deadline_slack = max_steps
            remaining_dwell = 0.0
            predecessor_pending = 0.0
            task_rel = (0.0, 0.0)
            task_distance = world_size
        remaining_fraction = sum(float(not task["completed"]) for task in tasks) / max(1, num_tasks)
        active_mask[agent_index] = int(agent.get("task_id", -1)) >= 0 or int(agent.get("service_task_id", -1)) >= 0
        joint_values = list(agent.get("joint_values", [0.0] * 6))
        joint_values = [float(value) for value in joint_values[:6]]
        while len(joint_values) < 6:
            joint_values.append(0.0)
        robot_features[agent_index] = np.array(
            [
                _signed_clip(agent["pos"][0], world_size),
                _signed_clip(agent["pos"][1], world_size),
                _signed_clip(agent["vel"][0], speed_scale),
                _signed_clip(agent["vel"][1], speed_scale),
                _signed_clip(task_rel[0], world_size),
                _signed_clip(task_rel[1], world_size),
                _signed_clip(release_slack, max_steps),
                _signed_clip(deadline_slack, max_steps),
                _positive_clip(remaining_dwell, max_steps),
                predecessor_pending,
                float(remaining_fraction),
                _positive_clip(task_distance, world_size),
                1.0,
                _signed_clip(joint_values[0], math.pi),
                _signed_clip(joint_values[1], math.pi),
                _signed_clip(joint_values[2], math.pi),
                _signed_clip(joint_values[3], math.pi),
                _signed_clip(joint_values[4], math.pi),
                _signed_clip(joint_values[5], math.pi),
            ],
            dtype=np.float32,
        )

        for other_index, other in enumerate(agents):
            if agent_index == other_index or other["reached"]:
                continue
            rel = vec_sub(other["pos"], agent["pos"])
            dist = vec_norm(rel)
            rel_dir = vec_unit(rel)
            rel_vel = vec_sub(other["vel"], agent["vel"])
            approach_rate = max(0.0, -(rel_vel[0] * rel_dir[0] + rel_vel[1] * rel_dir[1]))
            own_x_time = abs(agent["pos"][0]) / max(0.25, abs(agent["vel"][0]) + 0.25)
            other_x_time = abs(other["pos"][0]) / max(0.25, abs(other["vel"][0]) + 0.25)
            conflict_gap = abs(own_x_time - other_x_time)
            other_task_index, other_task = _resolve_reference_task(tasks, agents, other_index, step)
            yield_hint = 0.0
            if reference_task_index >= 0 and other_task_index >= 0 and other_task is not None and reference_task is not None:
                yield_hint = 1.0 if other_task["deadline_step"] < reference_task["deadline_step"] else 0.0
            rr_mask[agent_index, other_index] = True
            rr_edges[agent_index, other_index] = np.array(
                [
                    _signed_clip(rel[0], world_size),
                    _signed_clip(rel[1], world_size),
                    _signed_clip(rel_vel[0], speed_scale),
                    _signed_clip(rel_vel[1], speed_scale),
                    _positive_clip(dist, world_size),
                    _positive_clip(approach_rate, speed_scale),
                    _positive_clip(conflict_gap, max_steps),
                    yield_hint,
                ],
                dtype=np.float32,
            )

        for task_index, task in enumerate(tasks):
            if task["completed"]:
                continue
            rel = vec_sub(task["pos"], agent["pos"])
            dist = vec_norm(rel)
            release_slack = task["release_step"] - step
            deadline_slack = task["deadline_step"] - step
            predecessor_pending = float(task.get("predecessor_pending", 0.0))
            remaining_dwell = float(task.get("remaining_dwell_steps", task.get("dwell_steps", 1)))
            tr_mask[agent_index, task_index] = True
            tr_edges[agent_index, task_index] = np.array(
                [
                    _signed_clip(rel[0], world_size),
                    _signed_clip(rel[1], world_size),
                    _positive_clip(dist, world_size),
                    _signed_clip(release_slack, max_steps),
                    _signed_clip(deadline_slack, max_steps),
                    float(task.get("assigned_agent", -1) == agent_index),
                    float(task["priority"]),
                    float(release_slack <= 0),
                    _positive_clip(float(task.get("visit_rank", task_index + 1)), max(1, num_tasks)),
                    predecessor_pending,
                    _positive_clip(float(task.get("dwell_steps", 1)), max_steps),
                    _positive_clip(remaining_dwell, max_steps),
                    float(task.get("servicing_agent", -1) == agent_index),
                    float(task.get("is_available", True)),
                ],
                dtype=np.float32,
            )

        for obstacle_index, obstacle in enumerate(obstacles):
            rel = vec_sub(obstacle["center"], agent["pos"])
            dist = vec_norm(rel)
            clearance = dist - obstacle["radius"] - robot_radius
            or_mask[agent_index, obstacle_index] = True
            or_edges[agent_index, obstacle_index] = np.array(
                [
                    _signed_clip(rel[0], world_size),
                    _signed_clip(rel[1], world_size),
                    _positive_clip(dist, world_size),
                    _signed_clip(clearance, world_size),
                    _positive_clip(obstacle["radius"], world_size),
                ],
                dtype=np.float32,
            )

    return {
        "robot_features": robot_features,
        "task_features": task_features,
        "obstacle_features": obstacle_features,
        "rr_edges": rr_edges,
        "tr_edges": tr_edges,
        "or_edges": or_edges,
        "rr_mask": rr_mask,
        "tr_mask": tr_mask,
        "or_mask": or_mask,
        "active_mask": active_mask,
        "action_targets": np.asarray(action_targets, dtype=np.float32),
    }


def snapshot_to_model_inputs(snapshot, max_steps):
    sample = snapshot_to_sample(
        snapshot,
        max_steps=max_steps,
        action_targets=np.zeros((len(snapshot["agents"]), 6), dtype=np.float32),
    )
    sample.pop("action_targets")
    return sample


def stack_samples(samples):
    if not samples:
        raise ValueError("No samples collected.")
    stacked = {}
    for key in samples[0]:
        stacked[key] = np.stack([sample[key] for sample in samples], axis=0)
    return DatasetSplit(**stacked)


def _normalize_layout_pattern(layout_pattern):
    if isinstance(layout_pattern, str):
        tokens = [token.strip() for token in layout_pattern.split(",") if token.strip()]
    else:
        tokens = [str(token).strip() for token in layout_pattern if str(token).strip()]
    if not tokens:
        return ["structured"]
    valid = {"structured", "random", "representative"}
    invalid = [token for token in tokens if token not in valid]
    if invalid:
        raise ValueError(f"Unsupported layout modes in pattern: {invalid}")
    return tokens


def build_behavior_dataset(
    num_episodes,
    max_steps,
    num_robots=None,
    num_tasks=None,
    seed_start=4300,
    layout_pattern="structured",
    expert_max_collisions=None,
    return_metadata=False,
    xml_path=None,
    expert_ik_iterations=20,
    progress_interval=8,
    enforce_visit_order=None,
):
    resolved_pattern = _normalize_layout_pattern(layout_pattern)
    samples = []
    episode_stats = []
    dropped_episode_stats = []
    target_episode_count = int(num_episodes)
    kept_episode_count = 0
    attempt_index = 0
    max_attempts = max(target_episode_count, 1) * 20 if expert_max_collisions is not None else target_episode_count
    expert_ik_iterations = int(max(1, expert_ik_iterations))
    progress_interval = int(max(1, progress_interval))
    build_start_time = time.perf_counter()

    while kept_episode_count < target_episode_count:
        if attempt_index >= max_attempts:
            raise ValueError(
                "Unable to collect enough expert episodes under the collision filter. "
                f"requested={target_episode_count} kept={kept_episode_count} attempted={attempt_index} "
                f"expert_max_collisions={expert_max_collisions}"
            )
        env = HRSGAEnvironment(
            num_robots=num_robots,
            num_tasks=num_tasks,
            max_steps=max_steps,
            xml_path=xml_path,
            enforce_visit_order=enforce_visit_order,
        )
        layout_mode = resolved_pattern[attempt_index % len(resolved_pattern)]
        snapshot = env.reset(seed=seed_start + attempt_index, layout_mode=layout_mode)
        episode_samples = []
        while True:
            expert_target_points = [
                _reference_target_point(snapshot, agent_index)
                for agent_index in range(len(snapshot["agents"]))
            ]
            expert_joint_targets = np.asarray(
                env.solve_joint_targets_for_points(
                    expert_target_points,
                    ik_iterations=expert_ik_iterations,
                ),
                dtype=np.float32,
            )
            episode_samples.append(snapshot_to_sample(snapshot, max_steps=max_steps, action_targets=expert_joint_targets))
            snapshot, done, info = env.step(expert_joint_targets)
            if done:
                info = dict(info)
                info["layout_mode"] = layout_mode
                total_collisions = int(info.get("total_collisions", 0))
                keep_episode = expert_max_collisions is None or total_collisions <= int(expert_max_collisions)
                if keep_episode:
                    samples.extend(episode_samples)
                    episode_stats.append(info)
                    kept_episode_count += 1
                else:
                    dropped_episode_stats.append(info)
                break
        attempt_index += 1
        if (
            kept_episode_count == target_episode_count
            or attempt_index % progress_interval == 0
            or kept_episode_count % progress_interval == 0
        ):
            elapsed = time.perf_counter() - build_start_time
            last_completed = int(info.get("completed_tasks", 0))
            last_collisions = int(info.get("total_collisions", 0))
            print(
                "[DATASET] "
                f"kept={kept_episode_count}/{target_episode_count} "
                f"attempted={attempt_index}/{max_attempts} "
                f"layout={layout_mode} "
                f"last_completed={last_completed} "
                f"last_collisions={last_collisions} "
                f"ik_iters={expert_ik_iterations} "
                f"elapsed_s={elapsed:.1f}"
            )

    dataset = stack_samples(samples)
    if not return_metadata:
        return dataset, episode_stats

    metadata = {
        "requested_episode_count": target_episode_count,
        "attempted_episode_count": int(attempt_index),
        "kept_episode_count": int(len(episode_stats)),
        "dropped_episode_count": int(len(dropped_episode_stats)),
        "expert_max_collisions": None if expert_max_collisions is None else int(expert_max_collisions),
        "kept_expert_stats": episode_stats,
        "dropped_expert_stats": dropped_episode_stats,
    }
    return dataset, episode_stats, metadata


def _remap_legacy_policy_state_dict(state_dict):
    if any(key.startswith("state_encoder.") for key in state_dict.keys()):
        return state_dict

    remapped = {}
    encoder_prefixes = (
        "robot_encoder.",
        "task_encoder.",
        "obstacle_encoder.",
        "rr_attention.",
        "tr_attention.",
        "or_attention.",
        "fusion.",
    )
    for key, value in state_dict.items():
        if key.startswith(encoder_prefixes):
            remapped[f"state_encoder.{key}"] = value
        else:
            remapped[key] = value
    return remapped


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


class SparseRelationAttention(nn.Module):
    def __init__(self, target_dim, source_dim, edge_dim, hidden_dim, num_heads, topk, use_edge_bias=True):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.topk = topk
        self.query_proj = nn.Linear(target_dim, hidden_dim)
        self.key_proj = nn.Linear(source_dim, hidden_dim)
        self.value_proj = nn.Linear(source_dim, hidden_dim)
        self.use_edge_bias = bool(use_edge_bias)
        self.edge_bias = MLP(edge_dim, [hidden_dim, hidden_dim // 2], num_heads) if self.use_edge_bias else None
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, target_hidden, source_hidden, edge_features, valid_mask, topk=None):
        batch_size, target_count, _ = target_hidden.shape
        source_count = source_hidden.shape[1]
        if source_count == 0:
            return torch.zeros(batch_size, target_count, self.hidden_dim, device=target_hidden.device, dtype=target_hidden.dtype)

        query = self.query_proj(target_hidden).view(batch_size, target_count, self.num_heads, self.head_dim)
        key = self.key_proj(source_hidden).view(batch_size, source_count, self.num_heads, self.head_dim)
        value = self.value_proj(source_hidden).view(batch_size, source_count, self.num_heads, self.head_dim)
        logits = torch.einsum("bnhd,bmhd->bnmh", query, key) / math.sqrt(self.head_dim)
        if self.use_edge_bias:
            logits = logits + self.edge_bias(edge_features)

        expanded_mask = valid_mask.unsqueeze(-1).expand_as(logits)
        masked_logits = logits.masked_fill(~expanded_mask, -1e9)
        selected_topk = self.topk if topk is None else int(topk)
        k_select = max(1, min(selected_topk, source_count))
        top_values, top_indices = torch.topk(masked_logits, k=k_select, dim=2)
        top_mask = torch.zeros_like(logits, dtype=torch.bool)
        top_mask.scatter_(2, top_indices, top_values > -1e8)
        final_mask = expanded_mask & top_mask
        final_logits = logits.masked_fill(~final_mask, -1e9)
        attention = torch.softmax(final_logits, dim=2)
        attention = torch.where(final_mask, attention, torch.zeros_like(attention))
        empty_relations = (~valid_mask).all(dim=2, keepdim=True).unsqueeze(-1)
        attention = torch.where(empty_relations, torch.zeros_like(attention), attention)
        pooled = torch.einsum("bnmh,bmhd->bnhd", attention, value).reshape(batch_size, target_count, self.hidden_dim)
        return self.output_proj(pooled)


class DenseRelationResidual(nn.Module):
    def __init__(self, hidden_dim, rr_edge_dim, tr_edge_dim, or_edge_dim):
        super().__init__()
        self.rr_edge = MLP(rr_edge_dim, [hidden_dim], hidden_dim)
        self.tr_edge = MLP(tr_edge_dim, [hidden_dim], hidden_dim)
        self.or_edge = MLP(or_edge_dim, [hidden_dim], hidden_dim)
        self.message_mlp = MLP(hidden_dim * 3, [hidden_dim, hidden_dim], hidden_dim)
        self.local_update = MLP(hidden_dim * 3, [hidden_dim * 2, hidden_dim], hidden_dim)
        self.coord_update = MLP(hidden_dim * 3, [hidden_dim * 2, hidden_dim], hidden_dim)
        self.local_gate = nn.Linear(hidden_dim * 3, hidden_dim)
        self.coord_gate = nn.Linear(hidden_dim * 3, hidden_dim)

    def _aggregate(self, target_hidden, source_hidden, edge_features, edge_mask, edge_encoder):
        batch_size, target_count, _ = target_hidden.shape
        source_count = source_hidden.shape[1]
        if source_count == 0:
            return torch.zeros_like(target_hidden)

        edge_hidden = edge_encoder(edge_features)
        target_expand = target_hidden.unsqueeze(2).expand(-1, -1, source_count, -1)
        source_expand = source_hidden.unsqueeze(1).expand(-1, target_count, -1, -1)
        messages = self.message_mlp(torch.cat([target_expand, source_expand, edge_hidden], dim=-1))
        masked_messages = messages * edge_mask.unsqueeze(-1).float()
        denom = edge_mask.sum(dim=2, keepdim=True).clamp_min(1.0).float()
        return masked_messages.sum(dim=2) / denom

    def forward(self, robot_hidden, task_hidden, obstacle_hidden, rr_edges, tr_edges, or_edges, rr_mask, tr_mask, or_mask):
        tr_context = self._aggregate(robot_hidden, task_hidden, tr_edges, tr_mask, self.tr_edge)
        or_context = self._aggregate(robot_hidden, obstacle_hidden, or_edges, or_mask, self.or_edge)
        local_input = torch.cat([robot_hidden, tr_context, or_context], dim=-1)
        local_gate = torch.sigmoid(self.local_gate(local_input))
        local_hidden = robot_hidden + local_gate * self.local_update(local_input)

        rr_context = self._aggregate(local_hidden, local_hidden, rr_edges, rr_mask, self.rr_edge)
        global_context = local_hidden.mean(dim=1, keepdim=True).expand_as(local_hidden)
        coord_input = torch.cat([local_hidden, rr_context, global_context], dim=-1)
        coord_gate = torch.sigmoid(self.coord_gate(coord_input))
        return local_hidden + coord_gate * self.coord_update(coord_input)


class HRSGAStateEncoder(nn.Module):
    def __init__(
        self,
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
        disable_temporal_bias=False,
        disable_geometric_bias=False,
        shared_relation_attention=False,
        use_dense_residual=False,
    ):
        super().__init__()
        self.disable_temporal_bias = bool(disable_temporal_bias)
        self.disable_geometric_bias = bool(disable_geometric_bias)
        self.shared_relation_attention = bool(shared_relation_attention)
        self.use_dense_residual = bool(use_dense_residual)
        self.topk_robot = int(topk_robot)
        self.topk_task = int(topk_task)
        self.topk_obstacle = int(topk_obstacle)
        self.robot_encoder = MLP(robot_dim, [hidden_dim, hidden_dim], hidden_dim)
        self.task_encoder = MLP(task_dim, [hidden_dim, hidden_dim], hidden_dim)
        self.obstacle_encoder = MLP(obstacle_dim, [hidden_dim, hidden_dim], hidden_dim)
        if self.shared_relation_attention:
            self.rr_edge_adapter = MLP(rr_edge_dim, [hidden_dim], hidden_dim)
            self.tr_edge_adapter = MLP(tr_edge_dim, [hidden_dim], hidden_dim)
            self.or_edge_adapter = MLP(or_edge_dim, [hidden_dim], hidden_dim)
            self.shared_attention = SparseRelationAttention(hidden_dim, hidden_dim, hidden_dim, hidden_dim, num_heads, max(topk_robot, topk_task, topk_obstacle), use_edge_bias=not self.disable_geometric_bias)
        else:
            self.rr_attention = SparseRelationAttention(hidden_dim, hidden_dim, rr_edge_dim, hidden_dim, num_heads, topk_robot, use_edge_bias=not self.disable_geometric_bias)
            self.tr_attention = SparseRelationAttention(hidden_dim, hidden_dim, tr_edge_dim, hidden_dim, num_heads, topk_task, use_edge_bias=not self.disable_geometric_bias)
            self.or_attention = SparseRelationAttention(hidden_dim, hidden_dim, or_edge_dim, hidden_dim, num_heads, topk_obstacle, use_edge_bias=not self.disable_geometric_bias)
        self.local_fusion = MLP(hidden_dim * 3, [hidden_dim * 2, hidden_dim], hidden_dim)
        self.coord_fusion = MLP(hidden_dim * 3, [hidden_dim * 2, hidden_dim], hidden_dim)
        self.local_gate = nn.Linear(hidden_dim * 3, hidden_dim)
        self.coord_gate = nn.Linear(hidden_dim * 3, hidden_dim)
        if self.use_dense_residual:
            self.dense_residual = DenseRelationResidual(hidden_dim, rr_edge_dim, tr_edge_dim, or_edge_dim)
            self.dense_gate = nn.Linear(hidden_dim * 3, hidden_dim)

    def forward(self, robot_features, task_features, obstacle_features, rr_edges, tr_edges, or_edges, rr_mask, tr_mask, or_mask):
        if self.disable_temporal_bias or self.disable_geometric_bias:
            transformed = apply_feature_ablation(
                {
                    "robot_features": robot_features,
                    "task_features": task_features,
                    "obstacle_features": obstacle_features,
                    "rr_edges": rr_edges,
                    "tr_edges": tr_edges,
                    "or_edges": or_edges,
                },
                disable_temporal_bias=self.disable_temporal_bias,
                disable_geometric_bias=self.disable_geometric_bias,
            )
            robot_features = transformed["robot_features"]
            task_features = transformed["task_features"]
            obstacle_features = transformed["obstacle_features"]
            rr_edges = transformed["rr_edges"]
            tr_edges = transformed["tr_edges"]
            or_edges = transformed["or_edges"]

        robot_hidden = self.robot_encoder(robot_features)
        task_hidden = self.task_encoder(task_features)
        obstacle_hidden = self.obstacle_encoder(obstacle_features)
        if self.shared_relation_attention:
            tr_context = self.shared_attention(robot_hidden, task_hidden, self.tr_edge_adapter(tr_edges), tr_mask, topk=self.topk_task)
            or_context = self.shared_attention(robot_hidden, obstacle_hidden, self.or_edge_adapter(or_edges), or_mask, topk=self.topk_obstacle)
        else:
            tr_context = self.tr_attention(robot_hidden, task_hidden, tr_edges, tr_mask)
            or_context = self.or_attention(robot_hidden, obstacle_hidden, or_edges, or_mask)
        local_input = torch.cat([robot_hidden, tr_context, or_context], dim=-1)
        local_gate = torch.sigmoid(self.local_gate(local_input))
        local_hidden = robot_hidden + local_gate * self.local_fusion(local_input)

        if self.shared_relation_attention:
            rr_context = self.shared_attention(local_hidden, local_hidden, self.rr_edge_adapter(rr_edges), rr_mask, topk=self.topk_robot)
        else:
            rr_context = self.rr_attention(local_hidden, local_hidden, rr_edges, rr_mask)

        global_context = local_hidden.mean(dim=1, keepdim=True).expand_as(local_hidden)
        coord_input = torch.cat([local_hidden, rr_context, global_context], dim=-1)
        coord_gate = torch.sigmoid(self.coord_gate(coord_input))
        sparse_context = local_hidden + coord_gate * self.coord_fusion(coord_input)
        if not self.use_dense_residual:
            return sparse_context

        dense_context = self.dense_residual(
            robot_hidden,
            task_hidden,
            obstacle_hidden,
            rr_edges,
            tr_edges,
            or_edges,
            rr_mask,
            tr_mask,
            or_mask,
        )
        dense_gate = torch.sigmoid(self.dense_gate(torch.cat([robot_hidden, sparse_context, dense_context], dim=-1)))
        return sparse_context + dense_gate * dense_context


class HRSGAPolicyNetwork(nn.Module):
    def __init__(self, robot_dim, task_dim, obstacle_dim, rr_edge_dim, tr_edge_dim, or_edge_dim, hidden_dim=128, num_heads=4, topk_robot=2, topk_task=2, topk_obstacle=1, action_scale=3.0, disable_temporal_bias=False, disable_geometric_bias=False, shared_relation_attention=False, use_dense_residual=False):
        super().__init__()
        self.action_scale = action_scale
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
        self.action_head = nn.Linear(hidden_dim, 6)

    def forward(self, robot_features, task_features, obstacle_features, rr_edges, tr_edges, or_edges, rr_mask, tr_mask, or_mask, active_mask):
        fused = self.state_encoder(robot_features, task_features, obstacle_features, rr_edges, tr_edges, or_edges, rr_mask, tr_mask, or_mask)
        actions = torch.tanh(self.action_head(fused)) * self.action_scale
        return actions * active_mask.unsqueeze(-1)


class HRSGAQNetwork(nn.Module):
    def __init__(self, robot_dim, task_dim, obstacle_dim, rr_edge_dim, tr_edge_dim, or_edge_dim, hidden_dim=128, num_heads=4, topk_robot=2, topk_task=2, topk_obstacle=1, disable_temporal_bias=False, disable_geometric_bias=False, shared_relation_attention=False, use_dense_residual=False):
        super().__init__()
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
        self.action_encoder = MLP(hidden_dim + 6, [hidden_dim, hidden_dim], hidden_dim)
        self.q_head = MLP(hidden_dim * 2, [hidden_dim * 2, hidden_dim], 1)

    def forward(self, robot_features, task_features, obstacle_features, rr_edges, tr_edges, or_edges, rr_mask, tr_mask, or_mask, active_mask, actions):
        fused = self.state_encoder(robot_features, task_features, obstacle_features, rr_edges, tr_edges, or_edges, rr_mask, tr_mask, or_mask)
        active = active_mask.unsqueeze(-1)
        acted = self.action_encoder(torch.cat([fused, actions], dim=-1)) * active
        denom = active.sum(dim=1).clamp_min(1.0)
        pooled_mean = acted.sum(dim=1) / denom
        pooled_max = acted.masked_fill(~active.bool(), -1e9).max(dim=1).values
        pooled_max = torch.where(torch.isfinite(pooled_max), pooled_max, torch.zeros_like(pooled_mean))
        return self.q_head(torch.cat([pooled_mean, pooled_max], dim=-1)).squeeze(-1)


class HRSGAAgent:
    def __init__(self, num_robots, max_steps, num_tasks=None, hidden_dim=128, num_heads=4, topk_robot=2, topk_task=2, topk_obstacle=1, disable_temporal_bias=False, disable_geometric_bias=False, shared_relation_attention=False, use_dense_residual=False, device=None, xml_path=None):
        self.num_robots = int(num_robots)
        self.max_steps = int(max_steps)
        self.num_tasks = None if num_tasks is None else int(num_tasks)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.topk_robot = int(topk_robot)
        self.topk_task = int(topk_task)
        self.topk_obstacle = int(topk_obstacle)
        self.disable_temporal_bias = bool(disable_temporal_bias)
        self.disable_geometric_bias = bool(disable_geometric_bias)
        self.shared_relation_attention = bool(shared_relation_attention)
        self.use_dense_residual = bool(use_dense_residual)
        self.xml_path = xml_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = HRSGAPolicyNetwork(
            robot_dim=19,
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
        sample = snapshot_to_sample(snapshot, max_steps=self.max_steps, action_targets=np.zeros((len(snapshot["agents"]), 6), dtype=np.float32))
        batch = {key: value[None, ...] for key, value in sample.items() if key != "action_targets"}
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
            "num_heads": self.num_heads,
            "topk_robot": self.topk_robot,
            "topk_task": self.topk_task,
            "topk_obstacle": self.topk_obstacle,
            "disable_temporal_bias": self.disable_temporal_bias,
            "disable_geometric_bias": self.disable_geometric_bias,
            "shared_relation_attention": self.shared_relation_attention,
            "use_dense_residual": self.use_dense_residual,
            "xml_path": self.xml_path,
        }
        if optimizer is not None:
            checkpoint["optimizer_state"] = optimizer.state_dict()
        _save_checkpoint_with_retry(checkpoint, path)

    def load(self, path, optimizer=None):
        checkpoint = torch.load(path, map_location=self.device)
        model_state = _remap_legacy_policy_state_dict(checkpoint["model_state"])
        adapted_state = _adapt_state_dict_for_model(self.model.state_dict(), model_state)
        self.model.load_state_dict(adapted_state)
        if optimizer is not None and "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        return {
            "epoch": int(checkpoint.get("epoch", 0)),
            "best_score": float(checkpoint.get("best_score", -np.inf)),
        }


def load_agent_from_checkpoint(path, device=None):
    checkpoint = torch.load(path, map_location=device or ("cuda" if torch.cuda.is_available() else "cpu"))
    agent = HRSGAAgent(
        num_robots=int(checkpoint["num_robots"]),
        num_tasks=checkpoint.get("num_tasks"),
        max_steps=int(checkpoint["max_steps"]),
        hidden_dim=int(checkpoint.get("hidden_dim", 128)),
        num_heads=int(checkpoint.get("num_heads", 4)),
        topk_robot=int(checkpoint.get("topk_robot", 2)),
        topk_task=int(checkpoint.get("topk_task", 2)),
        topk_obstacle=int(checkpoint.get("topk_obstacle", 1)),
        disable_temporal_bias=bool(checkpoint.get("disable_temporal_bias", False)),
        disable_geometric_bias=bool(checkpoint.get("disable_geometric_bias", False)),
        shared_relation_attention=bool(checkpoint.get("shared_relation_attention", False)),
        use_dense_residual=bool(checkpoint.get("use_dense_residual", False)),
        device=device,
        xml_path=checkpoint.get("xml_path"),
    )
    model_state = _remap_legacy_policy_state_dict(checkpoint["model_state"])
    adapted_state = _adapt_state_dict_for_model(agent.model.state_dict(), model_state)
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


def iterate_minibatches(dataset, batch_size, shuffle=True, seed=0):
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
        }


def compute_action_loss(pred_actions, target_actions, active_mask):
    squared_error = (pred_actions - target_actions) ** 2
    active = active_mask.unsqueeze(-1)
    denom = active.sum().clamp_min(1.0) * pred_actions.shape[-1]
    return (squared_error * active).sum() / denom


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