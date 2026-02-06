#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np

from env.envrobot2d_obstacles import Robot2DEnvironmentObstacles


@dataclass(frozen=True)
class PolicyWrapper:
    """Generic deterministic policy wrapper."""

    act_fn: object

    def act(self, s: np.ndarray, *, env: Robot2DEnvironmentObstacles) -> np.ndarray:
        a = self.act_fn(s, env)
        return np.clip(np.asarray(a, dtype=np.float32), -1.0, 1.0)


class IKHeuristicPolicy:
    """Jacobian-transpose task-space PD policy (used as RL prior for RL-MPPI).

    Note: this is not learned; it's a strong nominal that you can later replace with a SAC policy.
    """

    def __init__(self, *, kp: float = 6.0, kd: float = 2.0, repulse_gain: float = 1.0, influence: float | None = None):
        self.kp = float(kp)
        self.kd = float(kd)
        self.repulse_gain = float(repulse_gain)
        self.influence = influence

    def __call__(self, s: np.ndarray, env: Robot2DEnvironmentObstacles) -> np.ndarray:
        n = env.n
        s = np.asarray(s, dtype=np.float32).reshape(2 * n)
        q = s[:n]
        qd = s[n:]

        eef = env.forward_kinematics_eef(q)
        J = env.jacobian_eef(q)
        eef_vel = J @ qd

        pos_err = env.target_pos - eef
        task_acc = self.kp * pos_err - self.kd * eef_vel

        infl = float(env.obstacle_margin if self.influence is None else self.influence)
        if env.obstacles and self.repulse_gain > 0.0:
            rep = np.zeros((2,), dtype=np.float32)
            for obs in env.obstacles:
                c = obs.center
                v = eef - c
                dist = float(np.linalg.norm(v))
                dist = max(dist, 1e-6)
                clearance = dist - float(obs.r) - float(env.safety_distance)
                depth = max(0.0, infl - clearance)
                if depth > 0.0:
                    rep += (v / np.float32(dist)) * (np.float32(depth) * np.float32(self.repulse_gain))
            task_acc = task_acc + rep

        qdd = (J.T @ task_acc).astype(np.float32)
        a = qdd / float(env.qdd_max)
        return np.clip(a, -1.0, 1.0)


class SACRobot2DPolicy:
    """SAC policy for Robot2D.

    Observation convention (default):
      obs = [q/pi, qd/qd_max, target_pos/reach]
    where reach = sum(link_lengths).
    """

    def __init__(
        self,
        agent: object,
        *,
        state_norm: str | None = None,
        include_obstacles_in_obs: bool = False,
        max_obstacles_in_obs: int = 0,
        obstacle_sort: str = "x_y_r",
    ):
        self.agent = agent
        self.state_norm = state_norm
        self.include_obstacles_in_obs = bool(include_obstacles_in_obs)
        self.max_obstacles_in_obs = int(max(0, max_obstacles_in_obs))
        self.obstacle_sort = str(obstacle_sort)

    def _encode_obstacles(self, env: Robot2DEnvironmentObstacles, *, reach: float) -> np.ndarray:
        if (not self.include_obstacles_in_obs) or self.max_obstacles_in_obs <= 0:
            return np.zeros((0,), dtype=np.float32)

        denom_reach = reach if reach != 0.0 else 1.0

        if self.obstacle_sort == "x_y_r":
            obs_sorted = sorted(env.obstacles, key=lambda o: (float(o.x), float(o.y), float(o.r)))
        else:
            # Fallback: deterministic anyway.
            obs_sorted = sorted(env.obstacles, key=lambda o: (float(o.x), float(o.y), float(o.r)))

        vec = np.zeros((3 * self.max_obstacles_in_obs,), dtype=np.float32)
        for i, o in enumerate(obs_sorted[: self.max_obstacles_in_obs]):
            vec[3 * i + 0] = np.float32(float(o.x) / denom_reach)
            vec[3 * i + 1] = np.float32(float(o.y) / denom_reach)
            vec[3 * i + 2] = np.float32(float(o.r) / denom_reach)
        return vec

    def _make_obs(self, s: np.ndarray, env: Robot2DEnvironmentObstacles) -> np.ndarray:
        n = env.n
        s = np.asarray(s, dtype=np.float32).reshape(2 * n)
        q = s[:n]
        qd = s[n:]

        reach = float(np.sum(np.asarray(env.link_lengths, dtype=np.float32)))
        obs_map = self._encode_obstacles(env, reach=reach)

        if self.state_norm == "robot2d_q_pi_qd_max_target_reach":
            qn = q / np.float32(np.pi)
            denom_qd = float(env.qd_max) if float(env.qd_max) != 0.0 else 1.0
            qdn = qd / np.float32(denom_qd)
            denom_reach = reach if reach != 0.0 else 1.0
            tn = np.asarray(env.target_pos, dtype=np.float32) / np.float32(denom_reach)
            base = np.concatenate([qn, qdn, tn], axis=0).astype(np.float32)
            if obs_map.size:
                return np.concatenate([base, obs_map], axis=0).astype(np.float32)
            return base

        # Default: raw state with target appended.
        base = np.concatenate([q, qd, np.asarray(env.target_pos, dtype=np.float32)], axis=0).astype(np.float32)
        if obs_map.size:
            return np.concatenate([base, obs_map], axis=0).astype(np.float32)
        return base

    def __call__(self, s: np.ndarray, env: Robot2DEnvironmentObstacles) -> np.ndarray:
        obs = self._make_obs(s, env)
        a = self.agent.select_action(obs, evaluate=True)
        return np.asarray(a, dtype=np.float32).reshape(env.action_dim)


def load_sac_policy(model_path: str, *, env: Robot2DEnvironmentObstacles) -> PolicyWrapper:
    """Load a Robot2D SAC checkpoint and return a PolicyWrapper.

    Checkpoint format follows experiments/ball2D/sac_ball/train_sac_ball_online.py.
    """

    try:
        import torch
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ModuleNotFoundError("PyTorch is required to load SAC models (pip install torch)") from e

    from algorithms.sac.sac_utils import SACAgent

    model_path = os.path.expanduser(os.path.expandvars(str(model_path)))
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"SAC model not found: {model_path}")

    ckpt = torch.load(model_path, map_location="cpu")
    obs_dim = int(ckpt.get("obs_dim", 2 * int(env.n) + 2))
    action_dim = int(ckpt.get("action_dim", int(env.action_dim)))
    if action_dim != int(env.action_dim):
        raise ValueError(f"Checkpoint action_dim={action_dim} != env.action_dim={env.action_dim}")

    auto_entropy_tuning = bool(ckpt.get("auto_entropy_tuning", True))
    alpha = float(ckpt.get("alpha", 0.2))

    agent = SACAgent(
        state_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=256,
        learning_rate=3e-4,
        alpha=alpha,
        gamma=0.99,
        tau=0.005,
        auto_entropy_tuning=auto_entropy_tuning,
        use_lr_scheduler=False,
    )

    if "policy_state_dict" not in ckpt:
        raise KeyError("Checkpoint missing 'policy_state_dict'")
    agent.policy_net.load_state_dict(ckpt["policy_state_dict"])
    if "q1_state_dict" in ckpt:
        agent.q_net1.load_state_dict(ckpt["q1_state_dict"])
    if "q2_state_dict" in ckpt:
        agent.q_net2.load_state_dict(ckpt["q2_state_dict"])
    if "target_q1_state_dict" in ckpt:
        agent.target_q_net1.load_state_dict(ckpt["target_q1_state_dict"])
    if "target_q2_state_dict" in ckpt:
        agent.target_q_net2.load_state_dict(ckpt["target_q2_state_dict"])

    state_norm = ckpt.get("state_norm", None)
    obs_config = ckpt.get("obs_config", {}) or {}
    include_obstacles_in_obs = bool(obs_config.get("include_obstacles_in_obs", False))
    max_obstacles_in_obs = int(obs_config.get("max_obstacles_in_obs", 0) or 0)
    obstacle_sort = str(obs_config.get("obstacle_sort", "x_y_r"))

    policy = SACRobot2DPolicy(
        agent,
        state_norm=state_norm,
        include_obstacles_in_obs=include_obstacles_in_obs,
        max_obstacles_in_obs=max_obstacles_in_obs,
        obstacle_sort=obstacle_sort,
    )
    return PolicyWrapper(act_fn=policy)


class RLMppiController:
    """RL-guided MPPI controller for Robot2DEnvironmentObstacles.

    - RL policy provides nominal u_nom by rolling out predicted dynamics.
    - MPPI samples around u_nom and reweights by cost.
    """

    def __init__(
        self,
        env: Robot2DEnvironmentObstacles,
        policy: PolicyWrapper,
        *,
        horizon: int = 25,
        num_samples: int = 300,
        lambda_coeff: float = 0.6,
        noise_std: float = 0.6,
        action_cost_coeff: float = 0.002,
        pos_cost_coeff: float = 1200.0,
        obstacle_margin: float | None = None,
        obstacle_cost_coeff: float = 6000.0,
        collision_cost: float = 2e6,
    ):
        self.env = env
        self.policy = policy
        self.horizon = int(horizon)
        self.num_samples = int(num_samples)
        self.lambda_coeff = float(lambda_coeff)
        self.noise_std = float(noise_std)

        self.action_cost_coeff = float(action_cost_coeff)
        self.pos_cost_coeff = float(pos_cost_coeff)

        self.obstacle_margin = float(env.obstacle_margin if obstacle_margin is None else obstacle_margin)
        self.obstacle_cost_coeff = float(obstacle_cost_coeff)
        self.collision_cost = float(collision_cost)

        self.action_dim = int(env.action_dim)
        self.state_dim = int(env.state_dim)
        self.dt = float(env.dt)

        self.u = np.zeros((self.horizon, self.action_dim), dtype=np.float32)

    def reset(self) -> None:
        self.u[:] = 0.0

    def _step_dynamics(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        n = self.env.n
        q = np.asarray(state[:n], dtype=np.float32)
        qd = np.asarray(state[n : 2 * n], dtype=np.float32)

        a = np.asarray(action, dtype=np.float32).reshape(n)
        a = np.clip(a, -1.0, 1.0)
        qdd = np.clip(a * self.env.qdd_max, -self.env.qdd_max, self.env.qdd_max)
        new_qd = np.clip(qd + qdd * self.dt, -self.env.qd_max, self.env.qd_max)
        new_q = q + new_qd * self.dt + 0.5 * qdd * (self.dt**2)
        new_q = (new_q + np.pi) % (2.0 * np.pi) - np.pi
        return np.concatenate([new_q, new_qd], axis=0).astype(np.float32)

    def _cost_step(self, state: np.ndarray, action: np.ndarray, target_pos: np.ndarray) -> float:
        n = self.env.n
        q = state[:n]
        eef = self.env.forward_kinematics_eef(q)
        pos_cost = float(np.linalg.norm(eef - target_pos)) * self.pos_cost_coeff
        act_cost = self.action_cost_coeff * float(np.linalg.norm(action))
        total = pos_cost + act_cost

        # Obstacle penalty: use environment clearance (EEF-only or full-arm depending on env.collision_check).
        if self.env.obstacles:
            min_clear, collided = self.env.min_obstacle_clearance(q)
            if bool(collided):
                total += float(self.collision_cost)
            depth = max(0.0, float(self.obstacle_margin) - float(min_clear))
            if depth > 0.0:
                total += float(self.obstacle_cost_coeff) * float(depth * depth)

        return float(total)

    def _simulate_cost(self, s0: np.ndarray, u_seq: np.ndarray, target_pos: np.ndarray) -> float:
        s = np.asarray(s0, dtype=np.float32).copy()
        total = 0.0
        for t in range(self.horizon):
            a = u_seq[t]
            total += self._cost_step(s, a, target_pos) * self.dt
            s = self._step_dynamics(s, a)
        return float(total)

    def _simulate_costs_batch(self, s0: np.ndarray, u_seqs: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
        u_seqs = np.asarray(u_seqs, dtype=np.float32)
        costs = np.zeros((u_seqs.shape[0],), dtype=np.float32)
        for i in range(u_seqs.shape[0]):
            costs[i] = np.float32(self._simulate_cost(s0, u_seqs[i], target_pos))
        return costs

    def _rollout_policy_nominal(self, current_state: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
        s = np.asarray(current_state, dtype=np.float32).copy()
        u_nom = np.zeros((self.horizon, self.action_dim), dtype=np.float32)
        for t in range(self.horizon):
            a = self.policy.act(s, env=self.env)
            u_nom[t] = a
            s = self._step_dynamics(s, a)
        return u_nom

    def get_action(self, state: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32).reshape(self.state_dim)
        target_pos = np.asarray(target_pos, dtype=np.float32).reshape(2)

        u_nom = self._rollout_policy_nominal(state, target_pos)

        noise = np.random.normal(0.0, self.noise_std, size=(self.num_samples, self.horizon, self.action_dim)).astype(np.float32)
        u_samples = np.clip(u_nom[None, :, :] + noise, -1.0, 1.0)

        costs = self._simulate_costs_batch(state, u_samples, target_pos)
        beta = float(np.min(costs))
        w = np.exp(-(costs - beta) / float(self.lambda_coeff)).astype(np.float32)
        w_sum = float(np.sum(w))
        if not np.isfinite(w_sum) or w_sum <= 1e-12:
            w = np.ones_like(w) / float(len(w))
        else:
            w = w / w_sum

        u_new = np.sum(u_samples * w[:, None, None], axis=0)
        u_new = np.clip(u_new, -1.0, 1.0)

        self.u = u_new.astype(np.float32)
        action = self.u[0].copy()
        self.u[:-1] = self.u[1:]
        self.u[-1] = 0.0
        return np.asarray(action, dtype=np.float32)
