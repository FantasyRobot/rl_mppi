import math
import os
import random
from dataclasses import dataclass

import numpy as np

try:
    from .mujoco_interface import apply_joint_targets, get_robot_states, load_layout_templates_from_mujoco, load_model_and_data, solve_joint_targets_for_ee_points, _build_goal_handles, _build_robot_handles, _build_robot_initial_joint_targets
except ImportError:
    from mujoco_interface import apply_joint_targets, get_robot_states, load_layout_templates_from_mujoco, load_model_and_data, solve_joint_targets_for_ee_points, _build_goal_handles, _build_robot_handles, _build_robot_initial_joint_targets


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
class RobotAgent:
    pos: tuple[float, float]
    vel: tuple[float, float]
    start_pos: tuple[float, float]
    joint_values: tuple[float, ...]
    start_joint_values: tuple[float, ...]
    ee_pos: tuple[float, float, float]
    base_pos: tuple[float, float] | None = None
    start_base_pos: tuple[float, float] | None = None
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


DEFAULT_LAYOUT_XML_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "urdf", "multi_robots.xml")
)


def load_layout_templates(xml_path: str):
    layout = load_layout_templates_from_mujoco(xml_path)
    return {
        "robot_positions": list(layout["robot_positions"]),
        "robot_initial_joints": list(layout.get("robot_initial_joints", [])),
        "goal_positions": list(layout["goal_positions"]),
        "goal_positions_3d": list(layout.get("goal_positions_3d", [])),
        "obstacles": [CircleObstacle(center=item["center"], radius=item["radius"]) for item in layout["obstacles"]],
        "goal_radius": float(layout.get("goal_radius", 0.0)),
        "goal_height": float(layout.get("goal_height", 0.30)),
        "max_joint_speed": None if layout.get("max_joint_speed") is None else float(layout["max_joint_speed"]),
        "enforce_visit_order": None if layout.get("enforce_visit_order") is None else bool(layout["enforce_visit_order"]),
    }


class HRSGAEnvironment:
    def __init__(
        self,
        num_robots: int | None = None,
        num_tasks: int | None = None,
        world_size: float = 4.5,
        max_steps: int = 280,
        dt: float = 0.12,
        robot_radius: float = 0.18,
        task_radius: float = 0.30,
        max_accel: float = 1.4,
        max_speed: float = 1.25,
        max_joint_speed: float | None = None,
        control_substeps: int = 8,
        enforce_visit_order: bool | None = None,
        min_dwell_steps: int = 1,
        max_dwell_steps: int = 3,
        xml_path: str | None = None,
    ):
        self.xml_path = os.path.abspath(xml_path or DEFAULT_LAYOUT_XML_PATH)
        self.layout_templates = load_layout_templates(self.xml_path)
        self.mujoco, self.mj_model, self.mj_data, self.resolved_xml_path = load_model_and_data(self.xml_path)
        self.robot_handles = _build_robot_handles(self.mujoco, self.mj_model)
        self.robot_initial_joint_targets = _build_robot_initial_joint_targets(self.mujoco, self.mj_model, self.robot_handles)
        self.goal_handles = _build_goal_handles(self.mujoco, self.mj_model, self.mj_data)
        parsed_num_robots = int(len(self.robot_handles))
        parsed_num_tasks = int(len(self.goal_handles))
        if num_robots is not None and int(num_robots) != parsed_num_robots:
            raise ValueError(
                f"Robot count is defined by XML: requested={int(num_robots)} parsed={parsed_num_robots} xml={self.xml_path}"
            )
        if num_tasks is not None and int(num_tasks) != parsed_num_tasks:
            raise ValueError(
                f"Task count is defined by XML goals: requested={int(num_tasks)} parsed={parsed_num_tasks} xml={self.xml_path}"
            )

        self.num_robots = int(max(1, parsed_num_robots))
        self.num_tasks = int(max(self.num_robots, parsed_num_tasks if parsed_num_tasks > 0 else self.num_robots * 2))
        self.world_size = float(world_size)
        self.max_steps = int(max_steps)
        self.dt = float(dt)
        self.robot_radius = float(robot_radius)
        parsed_goal_radius = float(self.layout_templates.get("goal_radius", 0.0))
        self.task_radius = float(parsed_goal_radius * 1.1 if parsed_goal_radius > 0.0 else task_radius)
        self.max_accel = float(max_accel)
        self.max_speed = float(max_speed)
        xml_max_joint_speed = self.layout_templates.get("max_joint_speed")
        resolved_max_joint_speed = xml_max_joint_speed if max_joint_speed is None else max_joint_speed
        if resolved_max_joint_speed is None:
            resolved_max_joint_speed = 2.5
        self.max_joint_speed = float(resolved_max_joint_speed)
        xml_enforce_visit_order = self.layout_templates.get("enforce_visit_order")
        if enforce_visit_order is None:
            enforce_visit_order = True if xml_enforce_visit_order is None else bool(xml_enforce_visit_order)
        self.control_substeps = int(max(1, control_substeps))
        self.target_track_kp = 1.35
        self.target_track_kd = 0.62
        self.home_return_kp = 1.15
        self.home_return_kd = 0.55
        self.home_idle_tolerance = max(0.10, self.robot_radius * 0.8)
        self.enforce_visit_order = bool(enforce_visit_order)
        self.min_dwell_steps = int(max(1, min_dwell_steps))
        self.max_dwell_steps = int(max(self.min_dwell_steps, max_dwell_steps))
        self.step_count = 0
        self.total_collisions = 0
        self.min_pair_distance = float("inf")
        self.robot_start_templates = list(self.layout_templates["robot_positions"])
        self.robot_initial_joint_templates = list(self.layout_templates.get("robot_initial_joints", []))
        self.goal_position_templates = list(self.layout_templates["goal_positions"])
        self.goal_position_templates_3d = list(self.layout_templates.get("goal_positions_3d", []))
        self.goal_height = float(self.layout_templates.get("goal_height", 0.30))
        self.obstacles = list(self.layout_templates["obstacles"])
        self.action_dim = int(len(self.robot_handles[0]["qpos_indices"])) if self.robot_handles else 0
        self.agents: list[RobotAgent] = []
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
        self.mujoco.mj_resetData(self.mj_model, self.mj_data)
        self._apply_initial_joint_configuration()
        self.mujoco.mj_forward(self.mj_model, self.mj_data)

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

    def _apply_initial_joint_configuration(self):
        for handle, initial_joint_values in zip(self.robot_handles, self.robot_initial_joint_targets):
            qpos_indices = handle["qpos_indices"]
            actuator_indices = handle["actuator_indices"]
            q_init = np.asarray(initial_joint_values, dtype=np.float64).reshape(-1)
            self.mj_data.qpos[qpos_indices] = q_init[: len(qpos_indices)]
            self.mj_data.ctrl[actuator_indices] = q_init[: len(actuator_indices)]

    def _reset_structured_layout(self, rng: random.Random):
        self._spawn_template_agents(rng=rng, add_jitter=False)
        task_positions = self._expanded_template_positions(
            self.goal_position_templates,
            target_count=self.num_tasks,
            rng=rng,
            jitter_scale=0.0,
        )
        self._create_tasks_from_positions(task_positions, release_offset=4, deadline_margin=24)

    def _reset_random_layout(self, rng: random.Random):
        self._spawn_template_agents(rng=rng, add_jitter=False, jitter_scale=0.0)
        task_positions = self._expanded_template_positions(
            self.goal_position_templates,
            target_count=self.num_tasks,
            rng=rng,
            jitter_scale=0.0,
            permute=True,
        )
        self._create_tasks_from_positions(task_positions, release_offset=3, deadline_margin=22)

    def _reset_representative_layout(self, seed: int | None):
        scenario_index = 0 if seed is None else abs(int(seed)) % 4
        rng = random.Random(seed)
        self._spawn_template_agents(rng=rng, add_jitter=False)
        ordered_positions = self._representative_permutation(self.goal_position_templates, scenario_index)
        task_positions = self._expanded_template_positions(
            ordered_positions,
            target_count=self.num_tasks,
            rng=rng,
            jitter_scale=0.0,
        )
        self._create_tasks_from_positions(
            task_positions,
            release_offset=4 + scenario_index,
            deadline_margin=24 + scenario_index,
            scenario_index=scenario_index,
        )

    def _spawn_template_agents(self, rng: random.Random, add_jitter: bool, jitter_scale: float = 0.0):
        robot_states = get_robot_states(self.mj_data, self.robot_handles)
        self.agents = []
        template_positions = list(self.robot_start_templates)
        for agent_index, state in enumerate(robot_states[: self.num_robots]):
            ee_pos = tuple(float(value) for value in state["ee_pos"])
            joint_values = tuple(float(value) for value in state["joint_values"])
            ee_xy = (float(ee_pos[0]), float(ee_pos[1]))
            if agent_index < len(template_positions):
                template_pos = template_positions[agent_index]
                jitter = rng.uniform(-jitter_scale, jitter_scale) if add_jitter and jitter_scale > 1e-8 else 0.0
                base_xy = (
                    float(template_pos[0] + jitter),
                    float(template_pos[1] + jitter),
                )
            else:
                base_xy = ee_xy
            self.agents.append(
                RobotAgent(
                    pos=ee_xy,
                    vel=(0.0, 0.0),
                    start_pos=ee_xy,
                    joint_values=joint_values,
                    start_joint_values=joint_values,
                    ee_pos=ee_pos,
                    base_pos=base_xy,
                    start_base_pos=base_xy,
                )
            )

    def _sync_agents_from_mujoco(self, previous_positions=None):
        robot_states = get_robot_states(self.mj_data, self.robot_handles)
        previous_positions = list(previous_positions or [])
        for agent_index, state in enumerate(robot_states[: len(self.agents)]):
            ee_pos = tuple(float(value) for value in state["ee_pos"])
            pos_xy = (float(ee_pos[0]), float(ee_pos[1]))
            if agent_index < len(previous_positions):
                prev_xy = previous_positions[agent_index]
                velocity = (
                    (pos_xy[0] - float(prev_xy[0])) / max(self.dt, 1e-6),
                    (pos_xy[1] - float(prev_xy[1])) / max(self.dt, 1e-6),
                )
            else:
                velocity = (0.0, 0.0)
            self.agents[agent_index].pos = pos_xy
            self.agents[agent_index].vel = velocity
            self.agents[agent_index].joint_values = tuple(float(value) for value in state["joint_values"])
            self.agents[agent_index].ee_pos = ee_pos

    def solve_joint_targets_for_points(self, target_points, *, ik_iterations: int = 60, damping: float = 1e-5, gain: float = 1.0):
        return solve_joint_targets_for_ee_points(
            self.mujoco,
            self.mj_model,
            self.mj_data,
            self.robot_handles[: len(target_points)],
            target_points,
            ik_iterations=ik_iterations,
            damping=damping,
            gain=gain,
            frame_substeps=self.control_substeps,
            max_joint_speed=self.max_joint_speed,
        )

    def _expanded_template_positions(self, base_positions, target_count: int, rng: random.Random, jitter_scale: float, permute: bool = False):
        if not base_positions:
            raise ValueError("No goal positions were parsed from the XML layout.")
        ordered = list(base_positions)
        if permute:
            rng.shuffle(ordered)

        # If the XML already defines enough goal positions and no jitter is requested,
        # keep those positions exactly as authored instead of re-sampling around
        # obstacle/task clearance rules. This keeps structured XML layouts authoritative.
        if target_count <= len(ordered) and jitter_scale <= 1e-8:
            return [
                (float(position[0]), float(position[1]))
                for position in ordered[:target_count]
            ]

        occupied_positions = []
        results = []
        for task_index in range(target_count):
            template = ordered[task_index % len(ordered)]
            extra_scale = jitter_scale + 0.04 * (task_index // max(1, len(ordered)))
            pos = self._jitter_template_position(
                template,
                rng=rng,
                occupied_positions=occupied_positions,
                min_gap=1.8 * self.task_radius,
                clearance=max(self.task_radius, 0.22),
                jitter_scale=extra_scale,
            )
            results.append(pos)
            occupied_positions.append(pos)
        return results

    def _create_tasks_from_positions(self, positions, release_offset: int, deadline_margin: int, scenario_index: int = 0):
        for task_index, task_pos in enumerate(positions):
            release_step = release_offset + task_index * 3
            deadline_step = self.max_steps - deadline_margin - task_index * 2
            self.tasks.append(
                TaskNode(
                    pos=task_pos,
                    assigned_agent=-1,
                    release_step=release_step,
                    deadline_step=deadline_step,
                    priority=1.0 + 0.04 * task_index,
                    visit_rank=task_index + 1,
                    dwell_steps=self._task_dwell_steps(task_index + scenario_index),
                )
            )

    def _jitter_template_position(self, template, rng: random.Random, occupied_positions, min_gap: float, clearance: float, jitter_scale: float):
        if jitter_scale <= 1e-8 and self._is_position_valid(template, occupied_positions, min_gap, clearance):
            return template
        for _ in range(128):
            candidate = (
                clamp(template[0] + rng.uniform(-jitter_scale, jitter_scale), -self.world_size + 0.5, self.world_size - 0.5),
                clamp(template[1] + rng.uniform(-jitter_scale, jitter_scale), -self.world_size + 0.5, self.world_size - 0.5),
            )
            if self._is_position_valid(candidate, occupied_positions, min_gap, clearance):
                return candidate
        if self._is_position_valid(template, occupied_positions, min_gap, clearance):
            return template
        return self._sample_free_position(rng, occupied_positions=occupied_positions, min_gap=min_gap, clearance=clearance)

    def _spawn_side_agents(self, rng: random.Random, add_jitter: bool):
        left_count = self.num_robots // 2
        right_count = self.num_robots - left_count
        y_slots = self._build_y_slots(max(left_count, right_count, 1))

        for idx in range(left_count):
            jitter = rng.uniform(-0.08, 0.08) if add_jitter else 0.0
            start_pos = (-self.world_size + 0.8, y_slots[idx] + jitter)
            self.agents.append(RobotAgent(pos=start_pos, vel=(0.0, 0.0), start_pos=start_pos))

        for idx in range(right_count):
            jitter = rng.uniform(-0.08, 0.08) if add_jitter else 0.0
            start_pos = (self.world_size - 0.8, -y_slots[idx] + jitter)
            self.agents.append(RobotAgent(pos=start_pos, vel=(0.0, 0.0), start_pos=start_pos))

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

    def _assignment_score(self, agent: RobotAgent, task: TaskNode):
        distance = self._agent_task_distance(agent, task)
        release_wait = max(0, task.release_step - self.step_count)
        deadline_slack = max(1, task.deadline_step - self.step_count)
        is_available = self._task_is_available(task)
        predecessor_pending = 0.0 if self._predecessors_completed(task) else 1.0
        release_bias = 0.9 if is_available else 0.45 / (1.0 + 0.08 * release_wait + 0.75 * predecessor_pending)
        urgency = 2.0 / (6.0 + deadline_slack)
        availability_bonus = 0.6 if is_available else 0.0
        staging_penalty = 0.08 * float(task.visit_rank - 1)
        return (1.8 / (0.35 + distance)) + release_bias + urgency + availability_bonus + 0.12 * task.priority - staging_penalty

    def _agent_task_distance(self, agent: RobotAgent, task: TaskNode):
        target = np.array([float(task.pos[0]), float(task.pos[1]), float(self.goal_height)], dtype=np.float64)
        current = np.array(agent.ee_pos, dtype=np.float64)
        return float(np.linalg.norm(target - current))

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
                task = self.tasks[task_index]
                candidate_pairs.append(
                    (
                        int(self._task_is_available(task)),
                        self._assignment_score(agent, task),
                        agent_index,
                        task_index,
                    )
                )

        for _, _, agent_index, task_index in sorted(candidate_pairs, key=lambda item: (item[0], item[1]), reverse=True):
            if agent_index in assigned_agents or task_index in assigned_tasks:
                continue
            self.agents[agent_index].task_id = task_index
            self.tasks[task_index].assigned_agent = agent_index
            assigned_agents.add(agent_index)
            assigned_tasks.add(task_index)

    def _home_status(self, agent: RobotAgent):
        distance = pairwise_distance(agent.pos, agent.start_pos)
        at_home = distance <= self.home_idle_tolerance
        return at_home, distance

    def _return_home_step(self, agent: RobotAgent):
        agent.is_returning_home = not self._home_status(agent)[0]

    def _decode_joint_target(self, agent: RobotAgent, action):
        current = np.asarray(agent.joint_values, dtype=np.float64)
        try:
            return np.asarray(action, dtype=np.float64).reshape(-1)[: current.shape[0]] if action is not None else current
        except Exception:
            return current

    def snapshot(self):
        return {
            "step": self.step_count,
            "world_size": self.world_size,
            "robot_radius": self.robot_radius,
            "task_radius": self.task_radius,
            "goal_height": self.goal_height,
            "agents": [
                {
                    "pos": agent.pos,
                    "vel": agent.vel,
                    "start_pos": agent.start_pos,
                    "base_pos": agent.base_pos,
                    "start_base_pos": agent.start_base_pos,
                    "joint_values": list(agent.joint_values),
                    "start_joint_values": list(agent.start_joint_values),
                    "ee_pos": list(agent.ee_pos),
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
        previous_positions = [agent.pos for agent in self.agents]
        joint_targets = []

        for agent, action in zip(self.agents, actions):
            if agent.is_servicing:
                agent.vel = (0.0, 0.0)
                agent.is_returning_home = False
                joint_targets.append(np.asarray(agent.joint_values, dtype=np.float64))
                continue

            current_task = self.tasks[agent.task_id] if 0 <= agent.task_id < len(self.tasks) else None
            if current_task is not None and self._task_is_available(current_task):
                if self._agent_task_distance(agent, current_task) <= self.task_radius:
                    agent.reached = True
                    agent.vel = (0.0, 0.0)
                    agent.is_returning_home = False
                    joint_targets.append(np.asarray(agent.joint_values, dtype=np.float64))
                    continue

            if agent.task_id < 0:
                if all_tasks_completed:
                    self._return_home_step(agent)
                    joint_targets.append(np.asarray(agent.start_joint_values, dtype=np.float64))
                else:
                    agent.is_returning_home = False
                    agent.vel = (0.0, 0.0)
                    joint_targets.append(np.asarray(agent.joint_values, dtype=np.float64))
                continue
            joint_targets.append(self._decode_joint_target(agent, action))
            agent.is_returning_home = False

        apply_joint_targets(
            self.mujoco,
            self.mj_model,
            self.mj_data,
            self.robot_handles[: len(joint_targets)],
            joint_targets,
            frame_substeps=self.control_substeps,
            max_joint_speed=self.max_joint_speed,
        )
        self._sync_agents_from_mujoco(previous_positions)

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
                if self._agent_task_distance(agent, task) <= self.task_radius:
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
                if self._agent_task_distance(agent, task) <= self.task_radius:
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
            total += min(self._agent_task_distance(agent, task) for agent in self.agents)
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
            clearance = obstacle.radius + self.robot_radius
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
                clearance = 2.0 * self.robot_radius
                if dist < clearance:
                    collisions += 1
        if self.min_pair_distance == float("inf"):
            self.min_pair_distance = 2.0 * self.world_size
        return collisions


__all__ = [
    "RobotAgent",
    "CircleObstacle",
    "HRSGAEnvironment",
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