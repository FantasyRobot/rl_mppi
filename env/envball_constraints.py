#!/usr/bin/env python3

from __future__ import annotations

import numpy as np


class BallEnvironmentConstraints:
    """Ball environment with explicit acceleration + velocity constraints.

    This environment is compatible with the existing SAC/MPPI codebase:
    - State: [x, y, vx, vy]
    - Action: normalized acceleration command in [-1, 1]^2

        Constraints (by default):
        - Acceleration component bound: |ax| <= acceleration_bound and |ay| <= acceleration_bound
            (implemented by clipping normalized action to [-1,1]^2 then scaling each component)
        - Velocity component bound: |vx| <= vel_bound and |vy| <= vel_bound
            (violation is reported via `info`, but does not terminate the episode)
    """

    def __init__(
        self,
        target_pos: list[float] | None = None,
        *,
        max_steps: int = 100,
        reward_scale: float = 1.0,
        reset_span: float = 5.0,
        reach_threshold: float = 0.5,
        pos_bound: float = 10.0,
        vel_bound: float = 2.0,
        acceleration_bound: float = 1.0,
        acc_bound: float | None = None,
    ):
        self.max_steps = int(max_steps)
        self.dt = 0.01

        self.pos_bound = float(pos_bound)
        self.vel_bound = float(vel_bound)
        self.acceleration_bound = float(acceleration_bound)
        # Optional tighter acceleration constraint (effective bound). If None, use physical limit.
        self.acc_bound = float(self.acceleration_bound if acc_bound is None else acc_bound)

        self.reward_scale = float(reward_scale)
        self.reset_span = float(reset_span)
        self.reach_threshold = float(reach_threshold)

        if target_pos is None:
            self.target_pos = np.array([0.0, 0.0], dtype=np.float32)
        else:
            self.target_pos = np.asarray(target_pos, dtype=np.float32).reshape(2)

        self.reset()

    def reset(self, initial_state=None, *, reset_span: float | None = None):
        self.current_step = 0

        if initial_state is None:
            self.state = np.zeros(4, dtype=np.float32)
            span = float(self.reset_span if reset_span is None else reset_span)
            self.state[0] = np.random.uniform(float(self.target_pos[0]) - span, float(self.target_pos[0]) + span)
            self.state[1] = np.random.uniform(float(self.target_pos[1]) - span, float(self.target_pos[1]) + span)
            self.state[0] = np.clip(self.state[0], -self.pos_bound, self.pos_bound)
            self.state[1] = np.clip(self.state[1], -self.pos_bound, self.pos_bound)
            self.state[2] = 0.0
            self.state[3] = 0.0
        else:
            s = np.asarray(initial_state, dtype=np.float32).reshape(-1)
            out = np.zeros(4, dtype=np.float32)
            out[: min(4, s.shape[0])] = s[:4]
            self.state = out

        self.prev_distance = float(np.linalg.norm(self.state[:2] - self.target_pos))
        self.last_action = np.zeros(2, dtype=np.float32)
        return self.state

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(2)
        action = np.clip(action, -1.0, 1.0)
        self.last_action = action.copy()

        # Physical command (scaled by physical limit), then clip by the effective acceleration constraint.
        ax_raw = float(action[0] * self.acceleration_bound)
        ay_raw = float(action[1] * self.acceleration_bound)
        ax = float(np.clip(ax_raw, -self.acc_bound, self.acc_bound))
        ay = float(np.clip(ay_raw, -self.acc_bound, self.acc_bound))

        x, y, vx, vy = [float(v) for v in self.state]

        new_vx_raw = vx + ax * self.dt
        new_vy_raw = vy + ay * self.dt

        vx_violation = max(0.0, abs(new_vx_raw) - float(self.vel_bound))
        vy_violation = max(0.0, abs(new_vy_raw) - float(self.vel_bound))
        vel_violation_amount = float(vx_violation + vy_violation)
        constraint_violation = (vx_violation > 1e-12) or (vy_violation > 1e-12)

        # Apply hard component-wise velocity limits to keep the state bounded.
        new_vx = float(np.clip(new_vx_raw, -self.vel_bound, self.vel_bound))
        new_vy = float(np.clip(new_vy_raw, -self.vel_bound, self.vel_bound))

        vel_norm_raw = float(np.hypot(float(new_vx_raw), float(new_vy_raw)))
        vel_norm = float(np.hypot(float(new_vx), float(new_vy)))

        new_x = x + new_vx * self.dt + 0.5 * ax * self.dt**2
        new_y = y + new_vy * self.dt + 0.5 * ay * self.dt**2

        clipped_x = float(np.clip(new_x, -self.pos_bound, self.pos_bound))
        clipped_y = float(np.clip(new_y, -self.pos_bound, self.pos_bound))
        hit_boundary = (abs(clipped_x - new_x) > 1e-9) or (abs(clipped_y - new_y) > 1e-9)

        self.state = np.asarray([clipped_x, clipped_y, float(new_vx), float(new_vy)], dtype=np.float32)
        self.current_step += 1

        reward = self._calculate_reward()

        distance = float(np.linalg.norm(self.state[:2] - self.target_pos))
        reached = distance < float(self.reach_threshold)

        time_limit = False
        done = False
        if reached:
            done = True
        elif self.current_step >= self.max_steps:
            done = True
            time_limit = True

        info = {
            "distance": distance,
            "step": int(self.current_step),
            "time_limit": bool(time_limit),
            "hit_boundary": bool(hit_boundary),
            "constraint_violation": bool(constraint_violation),
            "vel_norm": float(vel_norm),
            "vel_norm_raw": float(vel_norm_raw),
            "vel_violation_amount": float(vel_violation_amount),
            "vx": float(new_vx),
            "vy": float(new_vy),
            "vx_raw": float(new_vx_raw),
            "vy_raw": float(new_vy_raw),
            "acc_norm": float(np.hypot(ax, ay)),
            "applied_acceleration": float(np.hypot(ax, ay)),
            "ax": float(ax),
            "ay": float(ay),
            "ax_raw": float(ax_raw),
            "ay_raw": float(ay_raw),
            "acc_limit": float(self.acceleration_bound),
            "acc_bound": float(self.acc_bound),
        }

        return self.state, float(reward), bool(done), info

    def _calculate_reward(self) -> float:
        current_pos = self.state[:2]
        distance = float(np.linalg.norm(current_pos - self.target_pos))
        prev = float(getattr(self, "prev_distance", distance))
        reward = (prev - distance) * float(self.reward_scale)
        self.prev_distance = distance
        return float(reward)

    @property
    def state_dim(self) -> int:
        return 4

    @property
    def action_dim(self) -> int:
        return 2
