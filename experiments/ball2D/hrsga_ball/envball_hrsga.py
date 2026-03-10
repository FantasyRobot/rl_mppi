import math
import random
from dataclasses import dataclass


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def vec_add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def vec_sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def vec_mul(a, scalar: float):
    return (a[0] * scalar, a[1] * scalar)


def vec_norm(a) -> float:
    return math.hypot(a[0], a[1])


def vec_unit(a):
    norm = vec_norm(a)
    if norm < 1e-8:
        return (0.0, 0.0)
    return (a[0] / norm, a[1] / norm)


def vec_clip_norm(a, max_norm: float):
    norm = vec_norm(a)
    if norm <= max_norm or norm < 1e-8:
        return a
    scale = max_norm / norm
    return (a[0] * scale, a[1] * scale)


def pairwise_distance(a, b) -> float:
    return vec_norm(vec_sub(a, b))


@dataclass
class BallAgent:
    pos: tuple[float, float]
    vel: tuple[float, float]
    task_id: int
    reached: bool = False


@dataclass
class TaskNode:
    pos: tuple[float, float]
    assigned_agent: int
    release_step: int
    deadline_step: int
    priority: float
    completed: bool = False
    completed_step: int = -1


@dataclass
class CircleObstacle:
    center: tuple[float, float]
    radius: float


class HRSGABallEnvironment:
    def __init__(
        self,
        num_balls: int = 4,
        world_size: float = 4.5,
        max_steps: int = 180,
        dt: float = 0.12,
        ball_radius: float = 0.18,
        task_radius: float = 0.30,
        max_accel: float = 1.4,
        max_speed: float = 1.25,
    ):
        self.num_balls = num_balls
        self.world_size = world_size
        self.max_steps = max_steps
        self.dt = dt
        self.ball_radius = ball_radius
        self.task_radius = task_radius
        self.max_accel = max_accel
        self.max_speed = max_speed
        self.step_count = 0
        self.total_collisions = 0
        self.min_pair_distance = float("inf")
        self.obstacles = [
            CircleObstacle(center=(0.0, 0.95), radius=0.72),
            CircleObstacle(center=(0.0, -0.95), radius=0.72),
        ]
        self.agents: list[BallAgent] = []
        self.tasks: list[TaskNode] = []

    def reset(self, seed: int | None = None, random_layout: bool = False, layout_mode: str | None = None):
        rng = random.Random(seed)
        self.step_count = 0
        self.total_collisions = 0
        self.min_pair_distance = float("inf")
        self.agents = []
        self.tasks = []

        resolved_layout_mode = self._resolve_layout_mode(random_layout=random_layout, layout_mode=layout_mode)
        if resolved_layout_mode == "random":
            self._reset_random_layout(rng)
        elif resolved_layout_mode == "representative":
            self._reset_representative_layout(seed)
        elif resolved_layout_mode == "structured":
            self._reset_structured_layout(rng)
        else:
            raise ValueError(f"Unsupported layout mode: {resolved_layout_mode}")

        return self.snapshot()

    def _resolve_layout_mode(self, random_layout: bool, layout_mode: str | None):
        if layout_mode is not None:
            return layout_mode
        return "random" if random_layout else "structured"

    def _reset_structured_layout(self, rng: random.Random):
        left_count = self.num_balls // 2
        right_count = self.num_balls - left_count
        y_slots = self._build_y_slots(max(left_count, right_count))

        for idx in range(left_count):
            y = y_slots[idx] + rng.uniform(-0.08, 0.08)
            agent_id = len(self.agents)
            task_pos = (self.world_size - 0.8, -y)
            release_step = 4 + idx * 10
            deadline_step = self.max_steps - 20 - idx * 4
            self.agents.append(BallAgent(pos=(-self.world_size + 0.8, y), vel=(0.0, 0.0), task_id=agent_id))
            self.tasks.append(
                TaskNode(
                    pos=task_pos,
                    assigned_agent=agent_id,
                    release_step=release_step,
                    deadline_step=deadline_step,
                    priority=1.0 + 0.08 * idx,
                )
            )

        for idx in range(right_count):
            y = y_slots[idx] + rng.uniform(-0.08, 0.08)
            agent_id = len(self.agents)
            task_pos = (-self.world_size + 0.8, y)
            release_step = 7 + idx * 10
            deadline_step = self.max_steps - 18 - idx * 4
            self.agents.append(BallAgent(pos=(self.world_size - 0.8, -y), vel=(0.0, 0.0), task_id=agent_id))
            self.tasks.append(
                TaskNode(
                    pos=task_pos,
                    assigned_agent=agent_id,
                    release_step=release_step,
                    deadline_step=deadline_step,
                    priority=1.0 + 0.08 * idx,
                )
            )

    def _reset_random_layout(self, rng: random.Random):
        occupied_agent_positions = []
        occupied_task_positions = []
        min_agent_gap = 2.4 * self.ball_radius
        min_task_gap = 1.8 * self.task_radius
        min_start_goal_distance = 1.6

        for agent_id in range(self.num_balls):
            agent_pos = self._sample_free_position(
                rng,
                occupied_positions=occupied_agent_positions,
                min_gap=min_agent_gap,
                clearance=max(self.ball_radius, 0.18),
            )
            task_pos = self._sample_task_position(
                rng,
                occupied_positions=occupied_task_positions,
                agent_pos=agent_pos,
                min_gap=min_task_gap,
                min_agent_distance=min_start_goal_distance,
            )
            release_step = 4 + agent_id * 7
            deadline_step = self.max_steps - 20 - agent_id * 4
            self.agents.append(BallAgent(pos=agent_pos, vel=(0.0, 0.0), task_id=agent_id))
            self.tasks.append(
                TaskNode(
                    pos=task_pos,
                    assigned_agent=agent_id,
                    release_step=release_step,
                    deadline_step=deadline_step,
                    priority=1.0 + 0.08 * agent_id,
                )
            )
            occupied_agent_positions.append(agent_pos)
            occupied_task_positions.append(task_pos)

    def _reset_representative_layout(self, seed: int | None):
        left_count = self.num_balls // 2
        right_count = self.num_balls - left_count
        y_slots = self._build_y_slots(max(left_count, right_count))
        scenario_index = 0 if seed is None else abs(int(seed)) % 4

        left_target_ys = self._representative_permutation([-y_slots[idx] for idx in range(left_count)], scenario_index)
        right_target_ys = self._representative_permutation([y_slots[idx] for idx in range(right_count)], scenario_index + 1)

        for idx in range(left_count):
            agent_id = len(self.agents)
            agent_pos = (-self.world_size + 0.8, y_slots[idx])
            task_pos = (self.world_size - 0.8, left_target_ys[idx])
            release_step = 4 + scenario_index * 2 + idx * 8
            deadline_step = self.max_steps - 22 - idx * 4 - scenario_index
            self.agents.append(BallAgent(pos=agent_pos, vel=(0.0, 0.0), task_id=agent_id))
            self.tasks.append(
                TaskNode(
                    pos=task_pos,
                    assigned_agent=agent_id,
                    release_step=release_step,
                    deadline_step=deadline_step,
                    priority=1.0 + 0.08 * idx,
                )
            )

        for idx in range(right_count):
            agent_id = len(self.agents)
            agent_pos = (self.world_size - 0.8, -y_slots[idx])
            task_pos = (-self.world_size + 0.8, right_target_ys[idx])
            release_step = 6 + scenario_index * 2 + idx * 8
            deadline_step = self.max_steps - 20 - idx * 4 - scenario_index
            self.agents.append(BallAgent(pos=agent_pos, vel=(0.0, 0.0), task_id=agent_id))
            self.tasks.append(
                TaskNode(
                    pos=task_pos,
                    assigned_agent=agent_id,
                    release_step=release_step,
                    deadline_step=deadline_step,
                    priority=1.0 + 0.08 * idx,
                )
            )

    def _representative_permutation(self, values, scenario_index: int):
        if len(values) <= 1:
            return list(values)

        pattern = scenario_index % 4
        values = list(values)
        if pattern == 0:
            return values
        if pattern == 1:
            return list(reversed(values))
        if pattern == 2:
            shift = 1 % len(values)
            return values[shift:] + values[:shift]

        ordered = []
        left = 0
        right = len(values) - 1
        while left <= right:
            ordered.append(values[left])
            left += 1
            if left <= right:
                ordered.append(values[right])
                right -= 1
        return ordered

    def _sample_task_position(self, rng: random.Random, occupied_positions, agent_pos, min_gap: float, min_agent_distance: float):
        for _ in range(256):
            candidate = self._sample_free_position(
                rng,
                occupied_positions=occupied_positions,
                min_gap=min_gap,
                clearance=max(self.task_radius, 0.22),
            )
            if pairwise_distance(candidate, agent_pos) >= min_agent_distance:
                return candidate
        raise RuntimeError("Failed to sample a valid random task position.")

    def _sample_free_position(self, rng: random.Random, occupied_positions, min_gap: float, clearance: float):
        margin = max(0.55, clearance + 0.25)
        for _ in range(256):
            candidate = (
                rng.uniform(-self.world_size + margin, self.world_size - margin),
                rng.uniform(-self.world_size + margin, self.world_size - margin),
            )
            if not self._is_position_valid(candidate, occupied_positions, min_gap, clearance):
                continue
            return candidate
        raise RuntimeError("Failed to sample a valid random position.")

    def _is_position_valid(self, candidate, occupied_positions, min_gap: float, clearance: float):
        for obstacle in self.obstacles:
            obstacle_clearance = obstacle.radius + clearance + 0.18
            if pairwise_distance(candidate, obstacle.center) <= obstacle_clearance:
                return False
        for occupied in occupied_positions:
            if pairwise_distance(candidate, occupied) <= min_gap:
                return False
        return True

    def _build_y_slots(self, count: int):
        if count <= 1:
            return [0.0]
        span = 2.1
        return [(-span / 2.0) + span * index / (count - 1) for index in range(count)]

    def snapshot(self):
        return {
            "step": self.step_count,
            "world_size": self.world_size,
            "ball_radius": self.ball_radius,
            "task_radius": self.task_radius,
            "agents": [
                {
                    "pos": agent.pos,
                    "vel": agent.vel,
                    "task_id": agent.task_id,
                    "reached": agent.reached,
                }
                for agent in self.agents
            ],
            "tasks": [
                {
                    "pos": task.pos,
                    "assigned_agent": task.assigned_agent,
                    "release_step": task.release_step,
                    "deadline_step": task.deadline_step,
                    "priority": task.priority,
                    "completed": task.completed,
                    "completed_step": task.completed_step,
                }
                for task in self.tasks
            ],
            "obstacles": [
                {"center": obstacle.center, "radius": obstacle.radius}
                for obstacle in self.obstacles
            ],
        }

    def step(self, actions):
        if len(actions) != len(self.agents):
            raise ValueError("Action count must match agent count.")

        self.step_count += 1

        for agent, action in zip(self.agents, actions):
            if agent.reached:
                agent.vel = (0.0, 0.0)
                continue

            accel = vec_clip_norm((float(action[0]), float(action[1])), self.max_accel)
            new_vel = vec_add(agent.vel, vec_mul(accel, self.dt))
            new_vel = vec_clip_norm(new_vel, self.max_speed)
            new_vel = vec_mul(new_vel, 0.98)
            tentative_pos = vec_add(agent.pos, vec_mul(new_vel, self.dt))
            bounded_pos, bounded_vel = self._apply_bounds(tentative_pos, new_vel)
            obstacle_pos, obstacle_vel = self._apply_obstacles(bounded_pos, bounded_vel)
            agent.pos = obstacle_pos
            agent.vel = obstacle_vel

        pair_collisions = self._resolve_agent_collisions()
        self.total_collisions += pair_collisions

        completed_tasks = 0
        completed_before_deadline = 0
        missed_deadlines = 0
        mean_task_distance = 0.0
        for agent in self.agents:
            task = self.tasks[agent.task_id]
            distance = pairwise_distance(agent.pos, task.pos)
            mean_task_distance += distance
            if not task.completed and self.step_count >= task.release_step and distance <= self.task_radius:
                task.completed = True
                task.completed_step = self.step_count
                agent.reached = True
                agent.vel = (0.0, 0.0)

            if task.completed:
                completed_tasks += 1
                if task.completed_step <= task.deadline_step:
                    completed_before_deadline += 1
            elif self.step_count > task.deadline_step:
                missed_deadlines += 1

        success = completed_tasks == len(self.tasks)
        deadline_satisfaction = completed_before_deadline / max(1, len(self.tasks))
        mean_task_distance /= max(1, len(self.agents))
        done = success or self.step_count >= self.max_steps
        info = {
            "collisions": pair_collisions,
            "total_collisions": self.total_collisions,
            "completed_tasks": completed_tasks,
            "success": success,
            "deadline_satisfaction": deadline_satisfaction,
            "missed_deadlines": missed_deadlines,
            "mean_task_distance": mean_task_distance,
            "min_pair_distance": self.min_pair_distance,
        }
        return self.snapshot(), done, info

    def _apply_bounds(self, pos, vel):
        x = clamp(pos[0], -self.world_size, self.world_size)
        y = clamp(pos[1], -self.world_size, self.world_size)
        vx, vy = vel
        if abs(x - pos[0]) > 1e-8:
            vx *= -0.2
        if abs(y - pos[1]) > 1e-8:
            vy *= -0.2
        return (x, y), (vx, vy)

    def _apply_obstacles(self, pos, vel):
        adjusted_pos = pos
        adjusted_vel = vel
        for obstacle in self.obstacles:
            diff = vec_sub(adjusted_pos, obstacle.center)
            dist = vec_norm(diff)
            clearance = obstacle.radius + self.ball_radius
            if dist < clearance:
                normal = (1.0, 0.0) if dist < 1e-8 else vec_unit(diff)
                adjusted_pos = vec_add(obstacle.center, vec_mul(normal, clearance + 1e-4))
                radial_speed = adjusted_vel[0] * normal[0] + adjusted_vel[1] * normal[1]
                tangent_vel = vec_sub(adjusted_vel, vec_mul(normal, radial_speed))
                adjusted_vel = vec_mul(tangent_vel, 0.5)
        return adjusted_pos, adjusted_vel

    def _resolve_agent_collisions(self):
        collisions = 0
        for first_index in range(len(self.agents)):
            for second_index in range(first_index + 1, len(self.agents)):
                first = self.agents[first_index]
                second = self.agents[second_index]
                dist = pairwise_distance(first.pos, second.pos)
                self.min_pair_distance = min(self.min_pair_distance, dist)
                clearance = 2.0 * self.ball_radius
                if dist < clearance:
                    collisions += 1
                    normal = (1.0, 0.0) if dist < 1e-8 else vec_unit(vec_sub(first.pos, second.pos))
                    correction = vec_mul(normal, 0.5 * (clearance - dist + 1e-4))
                    first.pos = vec_add(first.pos, correction)
                    second.pos = vec_sub(second.pos, correction)
                    first.vel = vec_mul(first.vel, 0.55)
                    second.vel = vec_mul(second.vel, 0.55)
        if self.min_pair_distance == float("inf"):
            self.min_pair_distance = 2.0 * self.world_size
        return collisions

__all__ = [
    "BallAgent",
    "CircleObstacle",
    "HRSGABallEnvironment",
    "TaskNode",
    "clamp",
    "pairwise_distance",
    "vec_add",
    "vec_clip_norm",
    "vec_mul",
    "vec_norm",
    "vec_sub",
    "vec_unit",
]