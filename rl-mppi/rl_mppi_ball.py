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

This design is intentionally consistent with the existing implementation in
[mppi/mppi_ball.py](../mppi/mppi_ball.py) and the SAC checkpoint format in
[sac/sac_ball/test_sac_ball.py](../sac/sac_ball/test_sac_ball.py).
"""

import os
import sys
from dataclasses import dataclass

import numpy as np

# Ensure we can import project modules regardless of CWD.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, _ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from env.envball_utils import BallEnvironment


@dataclass(frozen=True)
class SACPolicyWrapper:
    """Wraps SACAgent + checkpoint metadata for deterministic inference."""

    agent: object
    checkpoint: dict
    normalize_state: bool

    def norm_state(self, s: np.ndarray, *, env: BallEnvironment) -> np.ndarray:
        if not self.normalize_state:
            return np.asarray(s, dtype=np.float32)
        out = np.asarray(s, dtype=np.float32).copy()
        out[0] = out[0] / float(env.pos_bound)
        out[1] = out[1] / float(env.pos_bound)
        out[2] = out[2] / float(env.vel_bound)
        out[3] = out[3] / float(env.vel_bound)
        return out

    def act(self, s: np.ndarray, *, env: BallEnvironment) -> np.ndarray:
        s_in = self.norm_state(s, env=env)
        a = self.agent.select_action(s_in, evaluate=True)
        return np.clip(np.asarray(a, dtype=np.float32), -1.0, 1.0)


def load_sac_policy(model_path: str, *, state_dim: int, action_dim: int) -> SACPolicyWrapper:
    """Load SAC policy from the checkpoint format used by sac_ball."""

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Train first (sac/sac_ball/sac_ball_cli.py train_online) or pass a valid --model_path."
        )

    import torch

    checkpoint = torch.load(model_path, map_location="cpu")
    auto_entropy_tuning = bool(checkpoint.get("auto_entropy_tuning", False))

    # Import here to avoid importing torch on module import when not needed.
    _SAC_DIR = os.path.join(_ROOT_DIR, "sac")
    if _SAC_DIR not in sys.path:
        sys.path.insert(0, _SAC_DIR)

    from sac_utils import SACAgent

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

    # Use checkpoint normalization metadata (same logic as test_sac_ball.py)
    ckpt_norm = checkpoint.get("state_norm", None)
    normalize_state = (ckpt_norm == "fixed_bounds")

    return SACPolicyWrapper(agent=agent, checkpoint=checkpoint, normalize_state=normalize_state)


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
        self.dt = float(env.dt if dt is None else dt)
        self.action_min = float(action_min)
        self.action_max = float(action_max)

        self.action_cost_coeff = float(action_cost_coeff)
        self.pos_cost_coeff = float(pos_cost_coeff)

        self.action_dim = env.action_dim
        self.state_dim = env.state_dim

        # Warm-start buffer (optional): we keep the last optimized sequence.
        self.u = np.zeros((self.horizon, self.action_dim), dtype=np.float32)

    def reset(self) -> None:
        self.u[:] = 0.0

    def _step_dynamics(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Manual one-step dynamics (mirrors env.step without mutating env)."""
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
        return pos_cost + act_cost

    def simulate_trajectory(self, initial_state: np.ndarray, action_sequence: np.ndarray, target_pos: np.ndarray) -> float:
        state = np.asarray(initial_state, dtype=np.float32).copy()
        total_cost = 0.0
        for t in range(self.horizon):
            action = action_sequence[t]
            total_cost += self.compute_cost(state, action, target_pos) * self.dt
            state = self._step_dynamics(state, action)
        return float(total_cost)

    def _rollout_policy_nominal(self, current_state: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
        """Roll the RL policy forward through predicted dynamics to build u_nom."""
        s = np.asarray(current_state, dtype=np.float32).copy()
        u_nom = np.zeros((self.horizon, self.action_dim), dtype=np.float32)
        for t in range(self.horizon):
            a = self.policy.act(s, env=self.env)
            # Optional: mild attraction to current warm-start (keeps smoothness)
            # We blend nominal with previous u to reduce jitter.
            a = 0.7 * a + 0.3 * self.u[t]
            a = np.clip(a, self.action_min, self.action_max)
            u_nom[t] = a
            s = self._step_dynamics(s, a)
        return u_nom

    def get_action(self, current_state: np.ndarray, target_pos: np.ndarray | list[float]) -> np.ndarray:
        target_pos = np.asarray(target_pos, dtype=np.float32).reshape(2)

        u_nom = self._rollout_policy_nominal(current_state, target_pos)

        # Sample noisy action sequences around u_nom.
        noise = np.random.randn(self.num_samples, self.horizon, self.action_dim).astype(np.float32) * self.noise_std
        action_sequences = u_nom[None, :, :] + noise
        action_sequences = np.clip(action_sequences, self.action_min, self.action_max)

        costs = np.zeros(self.num_samples, dtype=np.float32)
        for i in range(self.num_samples):
            costs[i] = self.simulate_trajectory(current_state, action_sequences[i], target_pos)

        cost_min = float(np.min(costs))
        # MPPI pdf weighting: w_i ∝ exp(-(S_i - S_min)/λ)
        weights = np.exp(-(costs - cost_min) / max(1e-8, self.lambda_coeff)).astype(np.float32)
        weights_sum = float(np.sum(weights))
        if weights_sum <= 0.0 or not np.isfinite(weights_sum):
            # Fallback: if weights blow up/underflow, return RL nominal first action.
            a0 = u_nom[0]
            self.u[:-1] = u_nom[1:]
            self.u[-1] = u_nom[-1]
            return a0
        weights /= weights_sum

        weighted_actions = np.sum(weights[:, None, None] * action_sequences, axis=0)

        # Update warm-start buffer by shifting the optimized sequence.
        self.u[:-1] = weighted_actions[1:]
        self.u[-1] = weighted_actions[-1]

        return np.asarray(weighted_actions[0], dtype=np.float32)
