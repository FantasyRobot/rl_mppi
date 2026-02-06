#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CircleObstacle:
    x: float
    y: float
    r: float

    @property
    def center(self) -> np.ndarray:
        return np.asarray([self.x, self.y], dtype=np.float32)


def _point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance from point p to segment [a,b] in 2D."""
    p = np.asarray(p, dtype=np.float32).reshape(2)
    a = np.asarray(a, dtype=np.float32).reshape(2)
    b = np.asarray(b, dtype=np.float32).reshape(2)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        return float(np.linalg.norm(p - a))
    t = float(np.dot(p - a, ab) / denom)
    t = float(np.clip(t, 0.0, 1.0))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


class Robot2DEnvironmentObstacles:
    """Planar N-link arm reaching a workspace goal with circular obstacles.

    State: [q_1..q_N, qd_1..qd_N]
    Action: normalized joint acceleration command in [-1,1]^N

    Dynamics:
      qd <- clip(qd + qdd*dt, -qd_max, qd_max)
      q  <- wrap(q + qd*dt + 0.5*qdd*dt^2)   (wrap to [-pi,pi])

    Goal: minimize end-effector distance to goal position.

    Collision: if any link segment intersects obstacle (clearance < 0).
    """

    def __init__(
        self,
        *,
        link_lengths: list[float] | np.ndarray = (2.0, 2.0, 1.0),
        target_pos: list[float] | np.ndarray = (3.0, 3.0),
        max_steps: int = 400,
        dt: float = 0.02,
        qd_max: float = 4.0,
        qdd_max: float = 8.0,
        reset_noise: float = 0.1,
        reach_threshold: float = 0.15,
        reward_scale: float = 100.0,
        obstacles: list[CircleObstacle] | None = None,
        obstacle_margin: float = 0.20,
        obstacle_penalty: float = 200.0,
        terminate_on_collision: bool = True,
        safety_distance: float = 0.03,
        collision_check: str = "arm",
    ):
        self.link_lengths = np.asarray(link_lengths, dtype=np.float32).reshape(-1)
        self.n = int(self.link_lengths.shape[0])
        if self.n < 1:
            raise ValueError("link_lengths must be non-empty")

        self.target_pos = np.asarray(target_pos, dtype=np.float32).reshape(2)
        self.max_steps = int(max_steps)
        self.dt = float(dt)

        self.qd_max = float(qd_max)
        self.qdd_max = float(qdd_max)
        self.reset_noise = float(reset_noise)
        self.reach_threshold = float(reach_threshold)
        self.reward_scale = float(reward_scale)

        self.obstacles = list(obstacles) if obstacles is not None else []
        self.obstacle_margin = float(obstacle_margin)
        self.obstacle_penalty = float(obstacle_penalty)
        self.terminate_on_collision = bool(terminate_on_collision)
        self.safety_distance = float(safety_distance)
        self.collision_check = str(collision_check).lower().strip()
        if self.collision_check not in {"arm", "eef"}:
            raise ValueError("collision_check must be 'arm' or 'eef'")

        self.current_step = 0
        self.state = np.zeros((2 * self.n,), dtype=np.float32)
        self.prev_distance = 0.0

        self.reset()

    @property
    def state_dim(self) -> int:
        return int(self.state.shape[0])

    @property
    def action_dim(self) -> int:
        return int(self.n)

    @property
    def action_bound(self) -> float:
        return 1.0

    def reset(self, initial_state: np.ndarray | None = None) -> np.ndarray:
        self.current_step = 0
        if initial_state is None:
            q = np.zeros((self.n,), dtype=np.float32)
            q += np.random.uniform(-self.reset_noise, self.reset_noise, size=(self.n,)).astype(np.float32)
            qd = np.zeros((self.n,), dtype=np.float32)
        else:
            s = np.asarray(initial_state, dtype=np.float32).reshape(-1)
            if s.shape[0] < 2 * self.n:
                raise ValueError(f"initial_state must have {2*self.n} elements")
            q = s[: self.n].copy()
            qd = s[self.n : 2 * self.n].copy()

        q = self._wrap_angles(q)
        qd = np.clip(qd, -self.qd_max, self.qd_max)
        self.state = np.concatenate([q, qd], axis=0).astype(np.float32)

        self.prev_distance = float(np.linalg.norm(self.forward_kinematics_eef(q) - self.target_pos))
        return self.state

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(self.n)
        action = np.clip(action, -1.0, 1.0)

        q = self.state[: self.n]
        qd = self.state[self.n : 2 * self.n]

        qdd = np.clip(action * self.qdd_max, -self.qdd_max, self.qdd_max)
        new_qd = np.clip(qd + qdd * self.dt, -self.qd_max, self.qd_max)
        new_q = q + new_qd * self.dt + 0.5 * qdd * (self.dt**2)
        new_q = self._wrap_angles(new_q)

        self.state = np.concatenate([new_q, new_qd], axis=0).astype(np.float32)
        self.current_step += 1

        eef = self.forward_kinematics_eef(new_q)
        dist = float(np.linalg.norm(eef - self.target_pos))

        # Progress reward: positive if distance decreases
        reward = (self.prev_distance - dist) * self.reward_scale
        self.prev_distance = dist

        clearance, collided = self.min_obstacle_clearance(new_q)

        # Soft penalty inside inflated obstacle margin.
        if self.obstacles:
            if clearance < float(self.obstacle_margin):
                depth = float(self.obstacle_margin - clearance)
                reward -= float(self.obstacle_penalty) * depth

        done = False
        reached = dist < float(self.reach_threshold)
        if reached:
            done = True
        if self.current_step >= self.max_steps:
            done = True
        if collided and self.terminate_on_collision:
            done = True

        info = {
            "step": int(self.current_step),
            "distance": float(dist),
            "reached": bool(reached),
            "time_limit": bool(self.current_step >= self.max_steps and (not reached)),
            "eef": np.asarray(eef, dtype=np.float32),
            "min_obstacle_clearance": float(clearance),
            "hit_obstacle": bool(collided),
        }

        return self.state, float(reward), bool(done), info

    def _wrap_angles(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float32)
        return (q + np.pi) % (2.0 * np.pi) - np.pi

    def forward_kinematics_all_joints(self, q: np.ndarray) -> np.ndarray:
        """Return joint positions including base and end-effector: shape (N+1,2)."""
        q = np.asarray(q, dtype=np.float32).reshape(self.n)
        pts = np.zeros((self.n + 1, 2), dtype=np.float32)
        ang = 0.0
        for i in range(self.n):
            ang += float(q[i])
            dx = float(self.link_lengths[i] * np.cos(ang))
            dy = float(self.link_lengths[i] * np.sin(ang))
            pts[i + 1, 0] = pts[i, 0] + dx
            pts[i + 1, 1] = pts[i, 1] + dy
        return pts

    def forward_kinematics_eef(self, q: np.ndarray) -> np.ndarray:
        pts = self.forward_kinematics_all_joints(q)
        return pts[-1]

    def jacobian_eef(self, q: np.ndarray) -> np.ndarray:
        """Analytical Jacobian of end-effector position wrt joint angles: shape (2,N)."""
        q = np.asarray(q, dtype=np.float32).reshape(self.n)
        J = np.zeros((2, self.n), dtype=np.float32)
        ang = 0.0
        # Precompute cumulative angles
        cum = np.cumsum(q).astype(np.float32)
        for k in range(self.n):
            sx = 0.0
            sy = 0.0
            for i in range(k, self.n):
                a = float(cum[i])
                Li = float(self.link_lengths[i])
                sx += -Li * np.sin(a)
                sy += Li * np.cos(a)
            J[0, k] = sx
            J[1, k] = sy
        return J

    def min_obstacle_clearance(self, q: np.ndarray) -> tuple[float, bool]:
        if not self.obstacles:
            return float("inf"), False

        if self.collision_check == "eef":
            eef = self.forward_kinematics_eef(q)
            ex = float(eef[0])
            ey = float(eef[1])
            min_clear = float("inf")
            collided = False
            rr_extra = float(self.safety_distance)
            for obs in self.obstacles:
                dx = ex - float(obs.x)
                dy = ey - float(obs.y)
                dist = float(np.sqrt(dx * dx + dy * dy))
                clear = dist - (float(obs.r) + rr_extra)
                if clear < min_clear:
                    min_clear = clear
                if clear < 0.0:
                    collided = True
            return float(min_clear), bool(collided)

        # Full-arm collision check: minimum clearance across all link segments.
        pts = self.forward_kinematics_all_joints(q)
        min_clear = float("inf")
        collided = False
        for obs in self.obstacles:
            c = obs.center
            rr = float(obs.r) + float(self.safety_distance)
            for i in range(self.n):
                d = _point_segment_distance(c, pts[i], pts[i + 1])
                clear = float(d - rr)
                if clear < min_clear:
                    min_clear = clear
                if clear < 0.0:
                    collided = True
        return float(min_clear), bool(collided)
