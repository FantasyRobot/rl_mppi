#!/usr/bin/env python3

from __future__ import annotations

"""RL-MPPI controller for the 2D ball environment.

This implements an MPPI-style path integral controller where the trajectory
probability density is proportional to exp(-S/λ), and uses a pre-trained RL
policy (SAC) as the *proposal / prior mean* for the control sequence.

High level:
- RL provides a nominal control sequence u_nom (by rolling the policy forward
  through predicted dynamics).
- MPPI samples noisy sequences around u_nom.
- Each sequence gets a weight w_i ∝ exp(-(S_i - S_min)/λ).
- The final control is the weighted average of sampled sequences.
"""

import os
import sys
from dataclasses import dataclass

import numpy as np

# Ensure repo root is importable regardless of CWD.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from env.envball_utils import BallEnvironment


@dataclass(frozen=True)
class SACPolicyWrapper:
    """Wraps SACAgent + checkpoint metadata for deterministic inference."""

    agent: object
    checkpoint: dict
    state_norm: str | None

    def norm_state(self, s: np.ndarray, *, env: BallEnvironment) -> np.ndarray:
        mode = self.state_norm
        if mode is None:
            return np.asarray(s, dtype=np.float32)

        out = np.asarray(s, dtype=np.float32).copy()
        if mode == "fixed_bounds_relative":
            out[0] = (out[0] - float(env.target_pos[0])) / float(env.pos_bound)
            out[1] = (out[1] - float(env.target_pos[1])) / float(env.pos_bound)
        elif mode == "fixed_bounds":
            out[0] = out[0] / float(env.pos_bound)
            out[1] = out[1] / float(env.pos_bound)
        else:
            return np.asarray(s, dtype=np.float32)

        out[2] = out[2] / float(env.vel_bound)
        out[3] = out[3] / float(env.vel_bound)
        return out

    def act(self, s: np.ndarray, *, env: BallEnvironment) -> np.ndarray:
        s_in = self.norm_state(s, env=env)
        a = self.agent.select_action(s_in, evaluate=True)
        return np.clip(np.asarray(a, dtype=np.float32), -1.0, 1.0)


def load_sac_policy(model_path: str, *, state_dim: int, action_dim: int) -> SACPolicyWrapper:
    """Load SAC policy from the checkpoint format used by sac_ball."""

    raw_path = os.path.expandvars(os.path.expanduser(str(model_path)))
    candidate_paths: list[str] = []
    candidate_paths.append(raw_path)

    # If given a relative path, also try resolving it from the repo root.
    if not os.path.isabs(raw_path):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../algorithms
        repo_root = os.path.dirname(repo_root)  # .../<repo>
        candidate_paths.append(os.path.join(repo_root, raw_path))

    resolved_path = None
    for p in candidate_paths:
        if os.path.exists(p):
            resolved_path = p
            break

    if resolved_path is None:
        tried = "\n".join([f"  - {p}" for p in candidate_paths])
        raise FileNotFoundError(
            "Model file not found. Tried:\n"
            f"{tried}\n"
            "Train first (experiments/sac_ball/sac_ball_cli.py train) or pass a valid --model_path."
        )

    import torch

    checkpoint = torch.load(resolved_path, map_location="cpu")
    auto_entropy_tuning = bool(checkpoint.get("auto_entropy_tuning", False))

    # Import here to avoid importing torch on module import when not needed.
    from algorithms.sac.sac_utils import SACAgent

    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        learning_rate=3e-4,
        alpha=float(checkpoint.get("alpha", 0.2)),
        gamma=0.99,
        tau=0.005,
        auto_entropy_tuning=auto_entropy_tuning,
    )

    agent.policy_net.load_state_dict(checkpoint["policy_state_dict"])
    agent.q_net1.load_state_dict(checkpoint["q1_state_dict"])
    agent.q_net2.load_state_dict(checkpoint["q2_state_dict"])
    agent.target_q_net1.load_state_dict(checkpoint["target_q1_state_dict"])
    agent.target_q_net2.load_state_dict(checkpoint["target_q2_state_dict"])

    ckpt_norm = checkpoint.get("state_norm", None)
    state_norm = (ckpt_norm if ckpt_norm in ("fixed_bounds", "fixed_bounds_relative") else None)

    return SACPolicyWrapper(agent=agent, checkpoint=checkpoint, state_norm=state_norm)


class RLMppiController:
    """RL-guided MPPI controller for BallEnvironment."""

    def __init__(
        self,
        env: BallEnvironment,
        policy: SACPolicyWrapper,
        *,
        horizon: int = 20,
        num_samples: int = 100,
        lambda_coeff: float = 0.5,
        noise_std: float = 0.5,
        vectorized_rollouts: bool = True,
        obstacles: list[tuple[float, float, float]] | None = None,
        obstacle_margin: float = 0.2,
        obstacle_cost_coeff: float = 2000.0,
        obstacle_safety_distance: float = 0.1,
        collision_cost: float = 1e7,
        use_los_obstacle_cost: bool = True,
        los_influence: float = 0.8,
        los_cost_coeff: float = 20000.0,
        dt: float | None = None,
        action_min: float = -1.0,
        action_max: float = 1.0,
        action_cost_coeff: float = 0.001,
        pos_cost_coeff: float = 1000.0,
    ):
        self.env = env
        self.policy = policy
        self.horizon = int(horizon)
        self.num_samples = int(num_samples)
        self.lambda_coeff = float(lambda_coeff)
        self.noise_std = float(noise_std)
        self.vectorized_rollouts = bool(vectorized_rollouts)
        self.dt = float(env.dt if dt is None else dt)
        self.action_min = float(action_min)
        self.action_max = float(action_max)

        self.action_cost_coeff = float(action_cost_coeff)
        self.pos_cost_coeff = float(pos_cost_coeff)

        self.obstacles = list(obstacles) if obstacles is not None else []
        self.obstacle_margin = float(obstacle_margin)
        self.obstacle_cost_coeff = float(obstacle_cost_coeff)
        self.obstacle_safety_distance = float(obstacle_safety_distance)
        self.collision_cost = float(collision_cost)
        self.use_los_obstacle_cost = bool(use_los_obstacle_cost)
        self.los_influence = float(los_influence)
        self.los_cost_coeff = float(los_cost_coeff)

        self.action_dim = env.action_dim
        self.state_dim = env.state_dim

        self.u = np.zeros((self.horizon, self.action_dim), dtype=np.float32)

    def reset(self) -> None:
        self.u[:] = 0.0

    def _step_dynamics(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        x, y, vx, vy = state
        ax = float(action[0]) * float(self.env.acceleration_bound)
        ay = float(action[1]) * float(self.env.acceleration_bound)

        ax = float(np.clip(ax, -self.env.acceleration_bound, self.env.acceleration_bound))
        ay = float(np.clip(ay, -self.env.acceleration_bound, self.env.acceleration_bound))

        new_vx = vx + ax * self.dt
        new_vy = vy + ay * self.dt
        new_x = x + new_vx * self.dt + 0.5 * ax * self.dt**2
        new_y = y + new_vy * self.dt + 0.5 * ay * self.dt**2

        new_x = float(np.clip(new_x, -self.env.pos_bound, self.env.pos_bound))
        new_y = float(np.clip(new_y, -self.env.pos_bound, self.env.pos_bound))
        new_vx = float(np.clip(new_vx, -self.env.vel_bound, self.env.vel_bound))
        new_vy = float(np.clip(new_vy, -self.env.vel_bound, self.env.vel_bound))

        return np.array([new_x, new_y, new_vx, new_vy], dtype=np.float32)

    def compute_cost(self, state: np.ndarray, action: np.ndarray, target_pos: np.ndarray) -> float:
        pos_error = state[:2] - target_pos
        pos_cost = float(np.linalg.norm(pos_error)) * self.pos_cost_coeff
        act_cost = self.action_cost_coeff * float(np.linalg.norm(action))
        total = pos_cost + act_cost

        if self.obstacles:
            x = float(state[0])
            y = float(state[1])
            influence = float(self.obstacle_margin)
            los_infl = float(self.los_influence)
            for (ox, oy, r) in self.obstacles:
                dx = x - float(ox)
                dy = y - float(oy)
                dist = float(np.sqrt(dx * dx + dy * dy))
                clearance = dist - (float(r) + self.obstacle_safety_distance)
                if clearance < 0.0:
                    total += self.collision_cost
                depth = max(0.0, influence - clearance)
                if depth > 0.0:
                    total += self.obstacle_cost_coeff * (depth * depth)

                if self.use_los_obstacle_cost and los_infl > 0.0:
                    px, py = float(state[0]), float(state[1])
                    gx, gy = float(target_pos[0]), float(target_pos[1])
                    sx = gx - px
                    sy = gy - py
                    seg_len2 = sx * sx + sy * sy
                    if seg_len2 > 1e-9:
                        t = ((float(ox) - px) * sx + (float(oy) - py) * sy) / seg_len2
                        t = max(0.0, min(1.0, t))
                        cx = px + t * sx
                        cy = py + t * sy
                        ddx = float(ox) - cx
                        ddy = float(oy) - cy
                        dseg = float(np.sqrt(ddx * ddx + ddy * ddy))
                        clearance_seg = dseg - (float(r) + self.obstacle_safety_distance)
                        depth_seg = max(0.0, los_infl - clearance_seg)
                        if depth_seg > 0.0:
                            total += self.los_cost_coeff * (depth_seg * depth_seg)

        return float(total)

    def simulate_trajectory(self, initial_state: np.ndarray, action_sequence: np.ndarray, target_pos: np.ndarray) -> float:
        state = np.asarray(initial_state, dtype=np.float32).copy()
        total_cost = 0.0
        for t in range(self.horizon):
            action = action_sequence[t]
            total_cost += self.compute_cost(state, action, target_pos) * self.dt
            state = self._step_dynamics(state, action)
        return float(total_cost)

    def simulate_trajectories_batch(self, initial_state: np.ndarray, action_sequences: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
        action_sequences = np.asarray(action_sequences, dtype=np.float32)
        if action_sequences.ndim != 3:
            raise ValueError(f"action_sequences must be 3D (N,H,dim), got shape={action_sequences.shape}")
        n, h, ad = action_sequences.shape
        if h != self.horizon or ad != self.action_dim:
            raise ValueError(
                f"action_sequences shape mismatch: expected (N,{self.horizon},{self.action_dim}), got {action_sequences.shape}"
            )

        s0 = np.asarray(initial_state, dtype=np.float32).reshape(-1)
        if s0.shape[0] < 4:
            raise ValueError(f"initial_state must have 4 elements, got shape={s0.shape}")

        target_pos = np.asarray(target_pos, dtype=np.float32).reshape(2)
        target_x = float(target_pos[0])
        target_y = float(target_pos[1])

        x = np.full((n,), float(s0[0]), dtype=np.float32)
        y = np.full((n,), float(s0[1]), dtype=np.float32)
        vx = np.full((n,), float(s0[2]), dtype=np.float32)
        vy = np.full((n,), float(s0[3]), dtype=np.float32)

        dt = np.float32(self.dt)
        dt2 = np.float32(self.dt * self.dt)

        acc_bound = np.float32(self.env.acceleration_bound)
        pos_bound = np.float32(self.env.pos_bound)
        vel_bound = np.float32(self.env.vel_bound)

        pos_cost_coeff = np.float32(self.pos_cost_coeff)
        act_cost_coeff = np.float32(self.action_cost_coeff)

        has_obs = bool(self.obstacles)
        influence = np.float32(self.obstacle_margin)
        obs_cost_coeff = np.float32(self.obstacle_cost_coeff)
        safety = np.float32(self.obstacle_safety_distance)
        collision_cost = np.float32(self.collision_cost)
        use_los = bool(self.use_los_obstacle_cost)
        los_influence = np.float32(self.los_influence)
        los_cost_coeff = np.float32(self.los_cost_coeff)
        if has_obs:
            obs_arr = np.asarray(self.obstacles, dtype=np.float32).reshape(-1, 3)

        costs = np.zeros((n,), dtype=np.float32)
        collided = np.zeros((n,), dtype=bool)

        for t in range(self.horizon):
            a = action_sequences[:, t, :]

            dx = x - target_x
            dy = y - target_y
            pos_norm = np.sqrt(dx * dx + dy * dy)
            act_norm = np.sqrt(a[:, 0] * a[:, 0] + a[:, 1] * a[:, 1])
            step_cost = (pos_norm * pos_cost_coeff + act_norm * act_cost_coeff)

            if has_obs:
                obs_pen = np.zeros((n,), dtype=np.float32)
                for j in range(obs_arr.shape[0]):
                    ox, oy, r = obs_arr[j]
                    odx = x - ox
                    ody = y - oy
                    dist = np.sqrt(odx * odx + ody * ody)
                    clearance = dist - (r + safety)
                    new_collide = (clearance < np.float32(0.0)) & (~collided)
                    if np.any(new_collide):
                        costs[new_collide] += collision_cost
                    collided |= (clearance < np.float32(0.0))
                    depth = np.maximum(np.float32(0.0), influence - clearance)
                    obs_pen += depth * depth

                    if use_los and los_influence > np.float32(0.0):
                        sx = (np.float32(target_pos[0]) - x)
                        sy = (np.float32(target_pos[1]) - y)
                        seg_len2 = sx * sx + sy * sy + np.float32(1e-6)
                        tt = ((ox - x) * sx + (oy - y) * sy) / seg_len2
                        tt = np.clip(tt, np.float32(0.0), np.float32(1.0))
                        cx = x + tt * sx
                        cy = y + tt * sy
                        ddx = ox - cx
                        ddy = oy - cy
                        dseg = np.sqrt(ddx * ddx + ddy * ddy)
                        clearance_seg = dseg - (r + safety)
                        depth_seg = np.maximum(np.float32(0.0), los_influence - clearance_seg)
                        obs_pen += (los_cost_coeff / obs_cost_coeff) * (depth_seg * depth_seg)
                step_cost = step_cost + obs_cost_coeff * obs_pen

            costs += step_cost * dt

            ax = np.clip(a[:, 0] * acc_bound, -acc_bound, acc_bound)
            ay = np.clip(a[:, 1] * acc_bound, -acc_bound, acc_bound)

            new_vx = vx + ax * dt
            new_vy = vy + ay * dt
            new_x = x + new_vx * dt + np.float32(0.5) * ax * dt2
            new_y = y + new_vy * dt + np.float32(0.5) * ay * dt2

            x = np.clip(new_x, -pos_bound, pos_bound)
            y = np.clip(new_y, -pos_bound, pos_bound)
            vx = np.clip(new_vx, -vel_bound, vel_bound)
            vy = np.clip(new_vy, -vel_bound, vel_bound)

        return costs

    def _rollout_policy_nominal(self, current_state: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
        s = np.asarray(current_state, dtype=np.float32).copy()
        u_nom = np.zeros((self.horizon, self.action_dim), dtype=np.float32)
        for t in range(self.horizon):
            a = self.policy.act(s, env=self.env)
            a = 0.7 * a + 0.3 * self.u[t]
            a = np.clip(a, self.action_min, self.action_max)
            u_nom[t] = a
            s = self._step_dynamics(s, a)
        return u_nom

    def get_action(self, current_state: np.ndarray, target_pos: np.ndarray | list[float]) -> np.ndarray:
        target_pos = np.asarray(target_pos, dtype=np.float32).reshape(2)

        u_nom = self._rollout_policy_nominal(current_state, target_pos)

        noise = np.random.randn(self.num_samples, self.horizon, self.action_dim).astype(np.float32) * self.noise_std
        action_sequences = u_nom[None, :, :] + noise
        action_sequences = np.clip(action_sequences, self.action_min, self.action_max)

        if self.vectorized_rollouts:
            costs = self.simulate_trajectories_batch(current_state, action_sequences, target_pos)
        else:
            costs = np.zeros(self.num_samples, dtype=np.float32)
            for i in range(self.num_samples):
                costs[i] = self.simulate_trajectory(current_state, action_sequences[i], target_pos)

        cost_min = float(np.min(costs))
        weights = np.exp(-(costs - cost_min) / max(1e-8, self.lambda_coeff)).astype(np.float32)
        weights_sum = float(np.sum(weights))
        if weights_sum <= 0.0 or not np.isfinite(weights_sum):
            a0 = u_nom[0]
            self.u[:-1] = u_nom[1:]
            self.u[-1] = u_nom[-1]
            return a0
        weights /= weights_sum

        weighted_actions = np.sum(weights[:, None, None] * action_sequences, axis=0)

        self.u[:-1] = weighted_actions[1:]
        self.u[-1] = weighted_actions[-1]

        return np.asarray(weighted_actions[0], dtype=np.float32)
