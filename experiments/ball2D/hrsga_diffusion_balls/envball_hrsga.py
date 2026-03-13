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
    start_pos: tuple[float, float]
    task_id: int = -1
    reached: bool = False
    is_servicing: bool = False
    service_task_id: int = -1
    is_returning_home: bool = False


@dataclass
class TaskNode:
    pos: tuple[float, float]
    assigned_agent: int
    release_step: int
    deadline_step: int
    priority: float
    visit_rank: int
    dwell_steps: int
    completed: bool = False
    completed_step: int = -1
    servicing_agent: int = -1
    service_progress: int = 0
    arrival_step: int = -1


@dataclass
class CircleObstacle:
    center: tuple[float, float]
    radius: float


class HRSGABallEnvironment:
    def __init__(
        self,
        num_balls: int = 4,
        num_tasks: int | None = None,
        world_size: float = 4.5,
        max_steps: int = 220,
        dt: float = 0.12,
        ball_radius: float = 0.18,
        task_radius: float = 0.30,
        max_accel: float = 1.4,
        max_speed: float = 1.25,
        enforce_visit_order: bool = True,
        min_dwell_steps: int = 1,
        max_dwell_steps: int = 3,
    ):
        self.num_balls = int(num_balls)
        self.num_tasks = int(max(self.num_balls, num_tasks if num_tasks is not None else self.num_balls * 2))
        self.world_size = float(world_size)
        self.max_steps = int(max_steps)
        self.dt = float(dt)
        self.ball_radius = float(ball_radius)
        self.task_radius = float(task_radius)
        self.max_accel = float(max_accel)
        self.max_speed = float(max_speed)
        self.home_return_kp = 1.15
        self.home_return_kd = 0.55
        self.home_idle_tolerance = max(0.10, self.ball_radius * 0.8)
        self.enforce_visit_order = bool(enforce_visit_order)
        self.min_dwell_steps = int(max(1, min_dwell_steps))
        self.max_dwell_steps = int(max(self.min_dwell_steps, max_dwell_steps))
        self.step_count = 0
        self.total_collisions = 0
        self.min_pair_distance = float("inf")
        self.obstacles = [
            CircleObstacle(center=(0.0, 0.95), radius=0.72),
            CircleObstacle(center=(0.0, -0.95), radius=0.72),
        ]
        self.agents: list[BallAgent] = []
        self.tasks: list[TaskNode] = []

    def task_rule_config(self):
        return {
            "enforce_visit_order": bool(self.enforce_visit_order),
            "min_dwell_steps": int(self.min_dwell_steps),
            "max_dwell_steps": int(self.max_dwell_steps),
        }

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

        self._update_assignments()
        return self.snapshot()

    def _resolve_layout_mode(self, random_layout: bool, layout_mode: str | None):
        if layout_mode is not None:
            return layout_mode
        return "random" if random_layout else "structured"

    def _reset_structured_layout(self, rng: random.Random):
        self._spawn_side_agents(rng=rng, add_jitter=True)
        left_count = self.num_tasks // 2
        right_count = self.num_tasks - left_count
        left_slots = self._build_y_slots(max(left_count, 1))
        right_slots = self._build_y_slots(max(right_count, 1))

        for idx in range(left_count):
            y = left_slots[idx] + rng.uniform(-0.10, 0.10)
            release_step = 4 + idx * 4
            deadline_step = self.max_steps - 26 - idx * 2
            self.tasks.append(
                TaskNode(
                    pos=(-self.world_size + 0.8, y),
                    assigned_agent=-1,
                    release_step=release_step,
                    deadline_step=deadline_step,
                    priority=1.0 + 0.03 * idx,
                    visit_rank=len(self.tasks) + 1,
                    dwell_steps=self._task_dwell_steps(len(self.tasks)),
                )
            )

        for idx in range(right_count):
            y = right_slots[idx] + rng.uniform(-0.10, 0.10)
            release_step = 6 + idx * 4
            deadline_step = self.max_steps - 24 - idx * 2
            self.tasks.append(
                TaskNode(
                    pos=(self.world_size - 0.8, -y),
                    assigned_agent=-1,
                    release_step=release_step,
                    deadline_step=deadline_step,
                    priority=1.0 + 0.03 * (left_count + idx),
                    visit_rank=len(self.tasks) + 1,
                    dwell_steps=self._task_dwell_steps(len(self.tasks)),
                )
            )

    def _reset_random_layout(self, rng: random.Random):
        occupied_agent_positions = []
        occupied_task_positions = []
        min_agent_gap = 2.4 * self.ball_radius
        min_task_gap = 1.8 * self.task_radius
        min_start_goal_distance = 1.3

        for _ in range(self.num_balls):
            agent_pos = self._sample_free_position(
                rng,
                occupied_positions=occupied_agent_positions,
                min_gap=min_agent_gap,
                clearance=max(self.ball_radius, 0.18),
            )
            self.agents.append(BallAgent(pos=agent_pos, vel=(0.0, 0.0), start_pos=agent_pos))
            occupied_agent_positions.append(agent_pos)

        for task_idx in range(self.num_tasks):
            anchor_agent_pos = occupied_agent_positions[task_idx % len(occupied_agent_positions)]
            task_pos = self._sample_task_position(
                rng,
                occupied_positions=occupied_task_positions,
                anchor_pos=anchor_agent_pos,
                min_gap=min_task_gap,
                min_anchor_distance=min_start_goal_distance,
            )
            release_step = 4 + task_idx * 3
            deadline_step = self.max_steps - 24 - task_idx * 2
            self.tasks.append(
                TaskNode(
                    pos=task_pos,
                    assigned_agent=-1,
                    release_step=release_step,
                    deadline_step=deadline_step,
                    priority=1.0 + 0.03 * task_idx,
                    visit_rank=task_idx + 1,
                    dwell_steps=self._task_dwell_steps(task_idx),
                )
            )
            occupied_task_positions.append(task_pos)

    def _reset_representative_layout(self, seed: int | None):
        scenario_index = 0 if seed is None else abs(int(seed)) % 4
        self._spawn_side_agents(rng=random.Random(seed), add_jitter=False)

        candidate_positions = [
            (-3.15, -1.60),
            (-3.15, 1.60),
            (-1.55, -2.15),
            (-1.55, 2.15),
            (-0.60, -0.20),
            (-0.60, 0.20),
            (0.60, -0.20),
            (0.60, 0.20),
            (1.55, -2.15),
            (1.55, 2.15),
            (3.15, -1.60),
            (3.15, 1.60),
        ]
        ordered_positions = self._representative_permutation(candidate_positions, scenario_index)

        for task_idx in range(self.num_tasks):
            base_pos = ordered_positions[task_idx % len(ordered_positions)]
            if task_idx >= len(ordered_positions):
                jitter_scale = 0.18 + 0.04 * (task_idx // len(ordered_positions))
                jitter_rng = random.Random((seed or 0) + 101 * task_idx)
                task_pos = (
                    clamp(base_pos[0] + jitter_rng.uniform(-jitter_scale, jitter_scale), -self.world_size + 0.5, self.world_size - 0.5),
                    clamp(base_pos[1] + jitter_rng.uniform(-jitter_scale, jitter_scale), -self.world_size + 0.5, self.world_size - 0.5),
                )
            else:
                task_pos = base_pos
            release_step = 4 + scenario_index + task_idx * 3
            deadline_step = self.max_steps - 26 - task_idx * 2 - scenario_index
            self.tasks.append(
                TaskNode(
                    pos=task_pos,
                    assigned_agent=-1,
                    release_step=release_step,
                    deadline_step=deadline_step,
                    priority=1.0 + 0.04 * task_idx,
                    visit_rank=task_idx + 1,
                    dwell_steps=self._task_dwell_steps(task_idx + scenario_index),
                )
            )

    def _spawn_side_agents(self, rng: random.Random, add_jitter: bool):
        left_count = self.num_balls // 2
        right_count = self.num_balls - left_count
        y_slots = self._build_y_slots(max(left_count, right_count, 1))

        for idx in range(left_count):
            jitter = rng.uniform(-0.08, 0.08) if add_jitter else 0.0
            start_pos = (-self.world_size + 0.8, y_slots[idx] + jitter)
            self.agents.append(BallAgent(pos=start_pos, vel=(0.0, 0.0), start_pos=start_pos))

        for idx in range(right_count):
            jitter = rng.uniform(-0.08, 0.08) if add_jitter else 0.0
            start_pos = (self.world_size - 0.8, -y_slots[idx] + jitter)
            self.agents.append(BallAgent(pos=start_pos, vel=(0.0, 0.0), start_pos=start_pos))

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
            shift = 2 % len(values)
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

    def _sample_task_position(self, rng: random.Random, occupied_positions, anchor_pos, min_gap: float, min_anchor_distance: float):
        for _ in range(256):
            candidate = self._sample_free_position(
                rng,
                occupied_positions=occupied_positions,
                min_gap=min_gap,
                clearance=max(self.task_radius, 0.22),
            )
            if pairwise_distance(candidate, anchor_pos) >= min_anchor_distance:
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
        span = min(3.4, 1.2 + 0.45 * (count - 1))
        return [(-span / 2.0) + span * index / (count - 1) for index in range(count)]

    def _task_dwell_steps(self, task_index: int):
        span = self.max_dwell_steps - self.min_dwell_steps + 1
        return self.min_dwell_steps + (int(task_index) % max(1, span))

    def _predecessors_completed(self, task: TaskNode):
        if not self.enforce_visit_order:
            return True
        for other_task in self.tasks:
            if other_task.visit_rank < task.visit_rank and not other_task.completed:
                return False
        return True

    def _task_is_available(self, task: TaskNode):
        if task.completed:
            return False
        if self.step_count < task.release_step:
            return False
        return self._predecessors_completed(task)

    def _remaining_dwell_steps(self, task: TaskNode):
        return max(0, int(task.dwell_steps) - int(task.service_progress))

    def _available_task_indices(self):
        return [
            task_index
            for task_index, task in enumerate(self.tasks)
            if self._task_is_available(task) and task.servicing_agent < 0
        ]

    def _assignable_task_indices(self):
        return [
            task_index
            for task_index, task in enumerate(self.tasks)
            if not task.completed and task.servicing_agent < 0
        ]

    def _assignment_score(self, agent: BallAgent, task: TaskNode):
        distance = pairwise_distance(agent.pos, task.pos)
        release_wait = max(0, task.release_step - self.step_count)
        deadline_slack = max(1, task.deadline_step - self.step_count)
        release_bias = 0.9 if release_wait == 0 else 0.35 / (1.0 + 0.12 * release_wait)
        urgency = 2.0 / (6.0 + deadline_slack)
        return (1.8 / (0.35 + distance)) + release_bias + urgency + 0.12 * task.priority

    def _update_assignments(self):
        all_tasks_completed = all(task.completed for task in self.tasks)
        for agent in self.agents:
            agent.reached = False
            if not agent.is_servicing:
                agent.task_id = -1
                agent.service_task_id = -1
                agent.is_returning_home = all_tasks_completed and not self._home_status(agent)[0]
        for task in self.tasks:
            if task.completed:
                task.assigned_agent = -1
                task.servicing_agent = -1
                task.service_progress = max(task.service_progress, task.dwell_steps)
            elif task.servicing_agent < 0:
                task.assigned_agent = -1

        assigned_agents = {agent_index for agent_index, agent in enumerate(self.agents) if agent.is_servicing}
        assigned_tasks = set()
        for task_index, task in enumerate(self.tasks):
            if task.completed or task.servicing_agent < 0:
                continue
            task.assigned_agent = task.servicing_agent
            assigned_tasks.add(task_index)
            service_agent = self.agents[task.servicing_agent]
            service_agent.task_id = task_index
            service_agent.service_task_id = task_index

        candidate_pairs = []
        for agent_index, agent in enumerate(self.agents):
            if agent_index in assigned_agents:
                continue
            for task_index in self._assignable_task_indices():
                candidate_pairs.append((self._assignment_score(agent, self.tasks[task_index]), agent_index, task_index))

        for _, agent_index, task_index in sorted(candidate_pairs, key=lambda item: item[0], reverse=True):
            if agent_index in assigned_agents or task_index in assigned_tasks:
                continue
            self.agents[agent_index].task_id = task_index
            self.tasks[task_index].assigned_agent = agent_index
            assigned_agents.add(agent_index)
            assigned_tasks.add(task_index)

    def _home_status(self, agent: BallAgent):
        distance = pairwise_distance(agent.pos, agent.start_pos)
        at_home = distance <= self.home_idle_tolerance
        return at_home, distance

    def _return_home_step(self, agent: BallAgent):
        at_home, _ = self._home_status(agent)
        if at_home:
            agent.pos = agent.start_pos
            agent.vel = (0.0, 0.0)
            agent.is_returning_home = False
            return

        rel_home = vec_sub(agent.start_pos, agent.pos)
        desired_accel = vec_sub(vec_mul(rel_home, self.home_return_kp), vec_mul(agent.vel, self.home_return_kd))
        accel = vec_clip_norm(desired_accel, self.max_accel)
        new_vel = vec_add(agent.vel, vec_mul(accel, self.dt))
        new_vel = vec_clip_norm(new_vel, self.max_speed)
        new_vel = vec_mul(new_vel, 0.98)
        tentative_pos = vec_add(agent.pos, vec_mul(new_vel, self.dt))
        bounded_pos, bounded_vel = self._apply_bounds(tentative_pos, new_vel)
        obstacle_pos, obstacle_vel = self._apply_obstacles(bounded_pos, bounded_vel)
        agent.pos = obstacle_pos
        agent.vel = obstacle_vel
        at_home, _ = self._home_status(agent)
        agent.is_returning_home = not at_home
        if at_home:
            agent.pos = agent.start_pos
            agent.vel = (0.0, 0.0)

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
                    "start_pos": agent.start_pos,
                    "task_id": agent.task_id,
                    "reached": agent.reached,
                    "is_servicing": agent.is_servicing,
                    "service_task_id": agent.service_task_id,
                    "is_returning_home": agent.is_returning_home,
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
                    "visit_rank": task.visit_rank,
                    "dwell_steps": task.dwell_steps,
                    "completed": task.completed,
                    "completed_step": task.completed_step,
                    "servicing_agent": task.servicing_agent,
                    "service_progress": task.service_progress,
                    "arrival_step": task.arrival_step,
                    "remaining_dwell_steps": self._remaining_dwell_steps(task),
                    "predecessor_pending": float(not self._predecessors_completed(task)),
                    "is_available": self._task_is_available(task) and task.servicing_agent < 0,
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
        all_tasks_completed = all(task.completed for task in self.tasks)

        for agent, action in zip(self.agents, actions):
            if agent.is_servicing:
                agent.vel = (0.0, 0.0)
                agent.is_returning_home = False
                continue
            if agent.task_id < 0:
                if all_tasks_completed:
                    self._return_home_step(agent)
                else:
                    agent.vel = (0.0, 0.0)
                    agent.is_returning_home = False
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
            agent.is_returning_home = False

        pair_collisions = self._resolve_agent_collisions()
        self.total_collisions += pair_collisions

        self._update_task_service()
        self._update_assignments()

        completed_tasks = sum(int(task.completed) for task in self.tasks)
        completed_before_deadline = sum(
            int(task.completed and task.completed_step <= task.deadline_step)
            for task in self.tasks
        )
        missed_deadlines = sum(
            int((not task.completed) and self.step_count > task.deadline_step)
            for task in self.tasks
        )
        mean_task_distance = self._mean_unfinished_task_distance()

        success = completed_tasks == len(self.tasks)
        deadline_satisfaction = completed_before_deadline / max(1, len(self.tasks))
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
            "active_service_tasks": sum(int(task.servicing_agent >= 0 and not task.completed) for task in self.tasks),
        }
        return self.snapshot(), done, info

    def _finish_task(self, task_index: int):
        task = self.tasks[task_index]
        task.completed = True
        task.completed_step = self.step_count
        servicing_agent = task.servicing_agent
        if 0 <= servicing_agent < len(self.agents):
            self.agents[servicing_agent].is_servicing = False
            self.agents[servicing_agent].service_task_id = -1
            self.agents[servicing_agent].task_id = -1
            self.agents[servicing_agent].vel = (0.0, 0.0)
            self.agents[servicing_agent].is_returning_home = all(item.completed for item in self.tasks)
        task.assigned_agent = -1
        task.servicing_agent = -1
        task.service_progress = task.dwell_steps

    def _update_task_service(self):
        for task in self.tasks:
            if task.completed or not self._task_is_available(task):
                continue

            if task.servicing_agent >= 0:
                agent = self.agents[task.servicing_agent]
                if pairwise_distance(agent.pos, task.pos) <= self.task_radius:
                    task.service_progress += 1
                    agent.vel = (0.0, 0.0)
                    if task.service_progress >= task.dwell_steps:
                        self._finish_task(self.tasks.index(task))
                else:
                    agent.is_servicing = False
                    agent.service_task_id = -1
                    agent.is_returning_home = False
                    task.servicing_agent = -1
                    task.assigned_agent = -1
                    task.service_progress = 0
                    task.arrival_step = -1
                continue

            for agent_index, agent in enumerate(self.agents):
                if agent.is_servicing:
                    continue
                if pairwise_distance(agent.pos, task.pos) <= self.task_radius:
                    agent.is_servicing = True
                    agent.service_task_id = self.tasks.index(task)
                    agent.task_id = self.tasks.index(task)
                    agent.vel = (0.0, 0.0)
                    agent.is_returning_home = False
                    task.assigned_agent = agent_index
                    task.servicing_agent = agent_index
                    task.service_progress = 1
                    task.arrival_step = self.step_count
                    if task.service_progress >= task.dwell_steps:
                        self._finish_task(self.tasks.index(task))
                    break

    def _mean_unfinished_task_distance(self):
        unfinished = [task for task in self.tasks if not task.completed]
        if not unfinished:
            return 0.0
        total = 0.0
        for task in unfinished:
            total += min(pairwise_distance(agent.pos, task.pos) for agent in self.agents)
        return total / max(1, len(unfinished))

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