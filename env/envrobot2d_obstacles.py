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
        # Obstacle shaping source for reward (collision checking always uses clearance).
        # - 'clearance': penalize low task-space min-clearance (default; best for online planning)
        # - 'cdf': penalize low C-space distance field value (best for SAC shaping)
        # - 'both': apply both penalties
        # - 'none': no soft obstacle penalty (still can terminate on collision)
        obstacle_shaping: str = "clearance",
        # CDF shaping parameters (only used when obstacle_shaping includes 'cdf')
        cdf_method: str = "offline_grid",
        cdf_obstacle_inflate: float = 0.0,
        cdf_margin: float | None = None,
        cdf_penalty: float | None = None,
        cdf_penalty_power: float = 2.0,
        cdf_bonus_gain: float = 0.0,
        cdf_bonus_cap: float = 2.0,
        collision_penalty: float = 0.0,
        terminate_on_collision: bool = True,
        safety_distance: float = 0.03,
        collision_check: str = "arm",
        # Optional safety shaping: when near obstacles, modify joint velocity to
        # move tangentially to the clearance level set (i.e., "along contour")
        # to bias trajectories away from collision.
        contour_avoidance: bool = False,
        contour_mode: str = "clearance",
        contour_cdf_method: str = "offline_grid",
        contour_clearance_start: float = 0.25,
        contour_clearance_full: float = 0.08,
        contour_fd_eps: float = 1e-3,
        contour_repulse_gain: float = 0.0,
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
        self.obstacle_shaping = str(obstacle_shaping).lower().strip()
        if self.obstacle_shaping not in {"clearance", "cdf", "both", "none"}:
            raise ValueError("obstacle_shaping must be one of: 'clearance', 'cdf', 'both', 'none'")

        self.cdf_method = str(cdf_method).lower().strip()
        self.cdf_obstacle_inflate = float(cdf_obstacle_inflate)
        self.cdf_margin = float(self.obstacle_margin if cdf_margin is None else cdf_margin)
        self.cdf_penalty = float(self.obstacle_penalty if cdf_penalty is None else cdf_penalty)
        self.cdf_penalty_power = float(cdf_penalty_power)
        self.cdf_bonus_gain = float(cdf_bonus_gain)
        self.cdf_bonus_cap = float(cdf_bonus_cap)
        self.collision_penalty = float(collision_penalty)
        if self.cdf_penalty_power <= 0.0:
            raise ValueError("cdf_penalty_power must be positive")
        if self.cdf_bonus_cap <= 0.0:
            raise ValueError("cdf_bonus_cap must be positive")
        self.terminate_on_collision = bool(terminate_on_collision)
        self.safety_distance = float(safety_distance)
        self.collision_check = str(collision_check).lower().strip()
        if self.collision_check not in {"arm", "eef"}:
            raise ValueError("collision_check must be 'arm' or 'eef'")

        self.current_step = 0
        self.state = np.zeros((2 * self.n,), dtype=np.float32)
        self.prev_distance = 0.0

        self.contour_avoidance = bool(contour_avoidance)
        self.contour_mode = str(contour_mode).lower().strip()
        self.contour_cdf_method = str(contour_cdf_method).lower().strip()
        self.contour_clearance_start = float(contour_clearance_start)
        self.contour_clearance_full = float(contour_clearance_full)
        self.contour_fd_eps = float(contour_fd_eps)
        self.contour_repulse_gain = float(contour_repulse_gain)
        if self.contour_mode not in {"clearance", "cdf"}:
            raise ValueError("contour_mode must be 'clearance' or 'cdf'")
        if self.contour_mode == "cdf" and self.n != 2:
            raise ValueError("contour_mode='cdf' currently requires n=2 joints")
        if self.contour_clearance_full > self.contour_clearance_start:
            raise ValueError("contour_clearance_full must be <= contour_clearance_start")
        if self.contour_fd_eps <= 0.0:
            raise ValueError("contour_fd_eps must be positive")

        self._cdf2d = None
        # Cache converted obstacle scene(s) for CDF, keyed by (inflate_total, obstacles tuple).
        self._cdf_scene_cache: dict[tuple[float, tuple[tuple[float, float, float], ...]], object] = {}

        self.reset()

    def _lazy_init_cdf(self):
        if self._cdf2d is not None:
            return
        import torch

        from algorithms.cdf_rl_mppi.cdf_2d.cdf import CDF2D

        self._cdf2d = CDF2D(device=torch.device("cpu"))

    def _cdf_scene(self, *, extra_inflate: float) -> object:
        """Return cached CDF obstacle scene list (torch primitives) for current env obstacles."""
        try:
            import torch
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError("PyTorch is required for CDF-based shaping") from e

        self._lazy_init_cdf()
        from algorithms.cdf_rl_mppi.cdf_2d.primitives2D_torch import Circle

        inflate_total = float(self.safety_distance) + float(extra_inflate)
        obs_sig = tuple((float(o.x), float(o.y), float(o.r)) for o in self.obstacles)
        key = (float(inflate_total), obs_sig)
        if key in self._cdf_scene_cache:
            return self._cdf_scene_cache[key]

        device = torch.device("cpu")
        scene = []
        for o in self.obstacles:
            center = torch.tensor([float(o.x), float(o.y)], dtype=torch.float32, device=device)
            radius = float(o.r) + float(inflate_total)
            scene.append(Circle(center=center, radius=radius, device=device))
        self._cdf_scene_cache[key] = scene
        return scene

    def cdf_value(self, q: np.ndarray) -> float:
        """Return CDF value at q (n=2 only). Uses configured cdf_method and inflation."""
        if int(self.n) != 2:
            raise ValueError("cdf_value currently supports only n=2")
        if not self.obstacles:
            return float("inf")
        try:
            import torch
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError("PyTorch is required for CDF-based shaping") from e

        self._lazy_init_cdf()
        scene = self._cdf_scene(extra_inflate=float(self.cdf_obstacle_inflate))
        q_t = torch.tensor(np.asarray(q, dtype=np.float32).reshape(1, 2), requires_grad=False)
        d = self._cdf2d.calculate_cdf(q_t, scene, method=str(self.cdf_method), return_grad=False)
        return float(np.asarray(d.detach().cpu().numpy()).reshape(-1)[0])

    def _contour_weight(self, value: float) -> float:
        if (not np.isfinite(value)) or (not self.obstacles):
            return 0.0
        c0 = float(self.contour_clearance_start)
        c1 = float(self.contour_clearance_full)
        if value >= c0:
            return 0.0
        if value <= c1:
            return 1.0
        # Linear ramp: 0 at start, 1 at full.
        return float((c0 - value) / max(1e-12, (c0 - c1)))

    def _clearance_value(self, q: np.ndarray) -> float:
        c, _ = self.min_obstacle_clearance(q)
        return float(c)

    def _cdf_value_and_grad(self, q: np.ndarray) -> tuple[float, np.ndarray]:
        """Return (cdf_value, cdf_grad) for n=2 using CDF2D.calculate_cdf."""
        if self.n != 2:
            raise RuntimeError("CDF mode only supported for n=2")
        try:
            import torch
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError("PyTorch is required for contour_mode='cdf'") from e

        self._lazy_init_cdf()
        scene = self._cdf_scene(extra_inflate=0.0)

        q_t = torch.tensor(np.asarray(q, dtype=np.float32).reshape(1, 2), requires_grad=True)
        d, g = self._cdf2d.calculate_cdf(q_t, scene, method=self.contour_cdf_method, return_grad=True)
        return float(d.detach().cpu().numpy().reshape(-1)[0]), g.detach().cpu().numpy().reshape(2).astype(np.float32)

    def _clearance_grad_fd(self, q: np.ndarray) -> np.ndarray:
        """Finite-difference gradient of min obstacle clearance wrt q."""
        q = np.asarray(q, dtype=np.float32).reshape(self.n)
        eps = float(self.contour_fd_eps)
        grad = np.zeros((self.n,), dtype=np.float32)
        for i in range(self.n):
            dq = np.zeros((self.n,), dtype=np.float32)
            dq[i] = eps
            qp = self._wrap_angles(q + dq)
            qm = self._wrap_angles(q - dq)
            cp = self._clearance_value(qp)
            cm = self._clearance_value(qm)
            # If either side is non-finite, fall back to one-sided.
            if np.isfinite(cp) and np.isfinite(cm):
                grad[i] = (cp - cm) / (2.0 * eps)
            elif np.isfinite(cp):
                c0 = self._clearance_value(q)
                grad[i] = (cp - c0) / eps
            elif np.isfinite(cm):
                c0 = self._clearance_value(q)
                grad[i] = (c0 - cm) / eps
            else:
                grad[i] = 0.0
        return grad

    def _shape_velocity_along_contour(self, q: np.ndarray, qd_des: np.ndarray) -> tuple[np.ndarray, dict]:
        """Project velocity onto clearance level set when approaching obstacles."""
        q = np.asarray(q, dtype=np.float32).reshape(self.n)
        qd_des = np.asarray(qd_des, dtype=np.float32).reshape(self.n)

        source = str(self.contour_mode)
        if source == "cdf":
            value, grad = self._cdf_value_and_grad(q)
        else:
            value = self._clearance_value(q)
            grad = self._clearance_grad_fd(q)

        w = self._contour_weight(value)
        if w <= 0.0:
            return qd_des, {"contour_w": 0.0, "contour_applied": False, "contour_source": source}

        g2 = float(np.dot(grad, grad))
        if g2 < 1e-10 or (not np.isfinite(g2)):
            return qd_des, {
                "contour_w": float(w),
                "contour_applied": False,
                "contour_source": source,
                "contour_reason": "degenerate_grad",
            }

        # If we're moving toward the obstacle (clearance decreasing), cancel that component.
        d_dot = float(np.dot(grad, qd_des))
        qd_tangent = qd_des
        applied = False
        if d_dot < 0.0:
            qd_tangent = qd_des - (d_dot / (g2 + 1e-12)) * grad
            applied = True

        # Optional repulsive normal component (increase clearance).
        rep = float(self.contour_repulse_gain)
        if rep != 0.0:
            qd_tangent = qd_tangent + rep * w * (grad / (np.sqrt(g2) + 1e-12))
            applied = True

        # Blend to keep continuity.
        qd_shaped = (1.0 - w) * qd_des + w * qd_tangent
        qd_shaped = np.clip(qd_shaped, -self.qd_max, self.qd_max)

        info = {
            "contour_w": float(w),
            "contour_applied": bool(applied),
            "contour_source": source,
            "contour_value": float(value),
            "contour_ddot": float(d_dot),
        }
        return qd_shaped.astype(np.float32), info

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

        contour_info: dict = {"contour_w": 0.0, "contour_applied": False}
        if self.contour_avoidance and self.obstacles:
            new_qd, contour_info = self._shape_velocity_along_contour(q, new_qd)

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

        # Soft obstacle penalty for shaping (collision checking remains clearance-based).
        cdf_val: float | None = None
        if self.obstacles and self.obstacle_shaping != "none":
            if self.obstacle_shaping in {"clearance", "both"}:
                if clearance < float(self.obstacle_margin):
                    depth = float(self.obstacle_margin - clearance)
                    reward -= float(self.obstacle_penalty) * depth

            if self.obstacle_shaping in {"cdf", "both"}:
                # Penalize low CDF (distance-to-contact in configuration space).
                # NOTE: we do not use CDF for collision detection; only shaping.
                try:
                    cdf_val = float(self.cdf_value(new_q))
                except ModuleNotFoundError:
                    # If torch is unavailable, fall back to clearance shaping.
                    cdf_val = None
                if cdf_val is not None and np.isfinite(cdf_val):
                    if cdf_val < float(self.cdf_margin):
                        depth = float(self.cdf_margin - cdf_val)
                        reward -= float(self.cdf_penalty) * float(depth ** float(self.cdf_penalty_power))
                    # Optional positive shaping: reward larger CDF (kept bounded).
                    if float(self.cdf_bonus_gain) != 0.0:
                        reward += float(self.cdf_bonus_gain) * float(min(float(cdf_val), float(self.cdf_bonus_cap)))

        # Optional collision penalty (in addition to termination), still based on clearance collision.
        if bool(collided) and float(self.collision_penalty) != 0.0:
            reward -= float(self.collision_penalty)

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
        if cdf_val is not None:
            info["cdf_value"] = float(cdf_val)
        info.update(contour_info)

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
