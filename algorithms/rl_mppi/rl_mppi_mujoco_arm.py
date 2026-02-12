#!/usr/bin/env python3

from __future__ import annotations

"""RL-MPPI for MuJoCo arm.

This mirrors the existing RL-MPPI idea used in `algorithms/rl_mppi/rl_mppi_ball.py`:
- A learned SAC policy provides a proposal / prior mean control sequence u_nom.
- MPPI samples noisy control sequences around u_nom.
- Weights are w_i ∝ exp(-(S_i - S_min)/λ).
- Use the weighted average control (receding horizon) as the applied action.

For t12a_14 we treat controls as position actuator targets (data.ctrl), clipped to ctrlrange.
"""

import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SACArmPolicyWrapper:
    agent: object
    checkpoint: dict

    def act(self, obs: np.ndarray) -> np.ndarray:
        a = self.agent.select_action(np.asarray(obs, dtype=np.float32), evaluate=True)
        return np.clip(np.asarray(a, dtype=np.float64).reshape(-1), -1.0, 1.0)


def load_sac_policy(model_path: str) -> SACArmPolicyWrapper:
    raw_path = os.path.expandvars(os.path.expanduser(str(model_path)))
    if not os.path.isabs(raw_path):
        # best-effort resolve relative to repo root
        here = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(os.path.dirname(here))
        cand = os.path.join(repo_root, raw_path)
        if os.path.exists(cand):
            raw_path = cand

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"SAC model not found: {raw_path}")

    try:
        import torch
    except ModuleNotFoundError as e:
        raise SystemExit(
            "Torch is required for SAC policy loading. Install in your env, e.g. (CPU):\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
        ) from e

    ckpt = torch.load(raw_path, map_location="cpu")

    from algorithms.sac.sac_utils import SACAgent

    agent = SACAgent(
        state_dim=int(ckpt["obs_dim"]),
        action_dim=int(ckpt["action_dim"]),
        hidden_dim=256,
        learning_rate=3e-4,
        alpha=float(ckpt.get("alpha", 0.2)),
        gamma=0.99,
        tau=0.005,
        auto_entropy_tuning=bool(ckpt.get("auto_entropy_tuning", False)),
        use_lr_scheduler=False,
    )

    agent.policy_net.load_state_dict(ckpt["policy_state_dict"])
    agent.q_net1.load_state_dict(ckpt["q1_state_dict"])
    agent.q_net2.load_state_dict(ckpt["q2_state_dict"])
    agent.target_q_net1.load_state_dict(ckpt["target_q1_state_dict"])
    agent.target_q_net2.load_state_dict(ckpt["target_q2_state_dict"])

    return SACArmPolicyWrapper(agent=agent, checkpoint=dict(ckpt))


class RLMuJoCoArmMPPI:
    def __init__(
        self,
        model,
        policy: SACArmPolicyWrapper,
        *,
        eef_site: str = "end_effector",
        goal_site: str = "goal",
        horizon: int = 25,
        num_samples: int = 96,
        lambda_coeff: float = 1.0,
        noise_std: float = 0.06,
        pos_cost_coeff: float = 200.0,
        action_cost_coeff: float = 0.02,
        smooth_cost_coeff: float = 0.2,
        seed: int | None = None,
    ):
        try:
            import mujoco
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError("RLMuJoCoArmMPPI requires 'mujoco'.") from e

        self._mujoco = mujoco
        self.model = model
        self.policy = policy

        self.horizon = int(horizon)
        self.num_samples = int(num_samples)
        self.lambda_coeff = float(lambda_coeff)
        self.noise_std = float(noise_std)

        self.pos_cost_coeff = float(pos_cost_coeff)
        self.action_cost_coeff = float(action_cost_coeff)
        self.smooth_cost_coeff = float(smooth_cost_coeff)

        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        else:
            self._rng = np.random.default_rng()

        self.nu = int(model.nu)
        self.nq = int(model.nq)
        self.nv = int(model.nv)
        if self.nu <= 0:
            raise ValueError("Model has no actuators.")

        self.ctrl_min = np.asarray(model.actuator_ctrlrange[:, 0], dtype=np.float64)
        self.ctrl_max = np.asarray(model.actuator_ctrlrange[:, 1], dtype=np.float64)
        self.ctrl_center = 0.5 * (self.ctrl_min + self.ctrl_max)
        self.ctrl_half = 0.5 * (self.ctrl_max - self.ctrl_min)
        self.ctrl_half = np.where(self.ctrl_half == 0.0, 1.0, self.ctrl_half)

        self.eef_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, str(eef_site))
        self.goal_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, str(goal_site))
        if self.eef_site_id < 0:
            raise ValueError(f"EEF site not found: {eef_site}")
        if self.goal_site_id < 0:
            raise ValueError(f"Goal site not found: {goal_site}")

        self.u = np.zeros((self.horizon, self.nu), dtype=np.float64)

        self._data_nominal = mujoco.MjData(self.model)
        self._data_pool = [mujoco.MjData(self.model) for _ in range(self.num_samples)]

        self._noise = np.empty((self.num_samples, self.horizon, self.nu), dtype=np.float64)
        self._u_samples = np.empty((self.num_samples, self.horizon, self.nu), dtype=np.float64)
        self._costs = np.empty((self.num_samples,), dtype=np.float64)

    def reset(self) -> None:
        self.u[:] = 0.0

    def _clip_ctrl(self, ctrl: np.ndarray) -> np.ndarray:
        return np.clip(ctrl, self.ctrl_min, self.ctrl_max)

    def _eef_pos(self, data) -> np.ndarray:
        return np.asarray(data.site_xpos[self.eef_site_id], dtype=np.float64).copy()

    def _goal_pos(self, data) -> np.ndarray:
        return np.asarray(data.site_xpos[self.goal_site_id], dtype=np.float64).copy()

    def _make_obs(self, data, goal_pos: np.ndarray) -> np.ndarray:
        q = np.asarray(data.qpos[: self.nu], dtype=np.float64)
        qn = (q - self.ctrl_center) / self.ctrl_half
        qn = np.clip(qn, -1.0, 1.0)

        qv = np.asarray(data.qvel, dtype=np.float64)
        qvel_scale = float(self.policy.checkpoint.get("qvel_scale", 5.0))
        qvel_scale = qvel_scale if qvel_scale != 0.0 else 1.0
        qvn = np.clip(qv / qvel_scale, -5.0, 5.0)

        eef = self._eef_pos(data)
        rel = eef - goal_pos

        return np.concatenate([qn, qvn, rel], axis=0).astype(np.float32)

    def _action_to_ctrl(self, a_norm: np.ndarray) -> np.ndarray:
        a = np.asarray(a_norm, dtype=np.float64).reshape(self.nu)
        a = np.clip(a, -1.0, 1.0)
        return self._clip_ctrl(self.ctrl_center + a * self.ctrl_half)

    def _rollout_policy_nominal(self, qpos0: np.ndarray, qvel0: np.ndarray) -> np.ndarray:
        mujoco = self._mujoco
        data = self._data_nominal
        data.qpos[:] = qpos0
        data.qvel[:] = qvel0
        mujoco.mj_forward(self.model, data)

        goal_pos = self._goal_pos(data)

        u_nom = np.zeros((self.horizon, self.nu), dtype=np.float64)
        prev_ctrl = np.clip(np.asarray(qpos0[: self.nu], dtype=np.float64), self.ctrl_min, self.ctrl_max)

        for t in range(self.horizon):
            obs = self._make_obs(data, goal_pos)
            a = self.policy.act(obs)
            ctrl = self._action_to_ctrl(a)

            # Blend with previous to avoid jitter.
            ctrl = self._clip_ctrl(0.8 * ctrl + 0.2 * prev_ctrl)

            u_nom[t] = ctrl
            prev_ctrl = ctrl

            data.ctrl[:] = ctrl
            mujoco.mj_step(self.model, data)

        return u_nom

    def _step_cost(self, data, ctrl: np.ndarray, goal_pos: np.ndarray, prev_ctrl: np.ndarray) -> float:
        eef = self._eef_pos(data)
        dist = float(np.linalg.norm(eef - goal_pos))
        pos_cost = self.pos_cost_coeff * dist

        q = np.asarray(data.qpos[: self.nu], dtype=np.float64)
        act_cost = self.action_cost_coeff * float(np.linalg.norm(ctrl - q))

        smooth_cost = self.smooth_cost_coeff * float(np.linalg.norm(ctrl - prev_ctrl))

        return float(pos_cost + act_cost + smooth_cost)

    def _rollout_cost_inplace(self, data, qpos0: np.ndarray, qvel0: np.ndarray, u_seq: np.ndarray, goal_pos: np.ndarray) -> float:
        mujoco = self._mujoco
        data.qpos[:] = qpos0
        data.qvel[:] = qvel0
        mujoco.mj_forward(self.model, data)

        goal_pos = np.asarray(goal_pos, dtype=np.float64).reshape(3)
        total = 0.0

        prev_ctrl = np.clip(np.asarray(qpos0[: self.nu], dtype=np.float64), self.ctrl_min, self.ctrl_max)
        dt = float(self.model.opt.timestep)

        for t in range(self.horizon):
            ctrl = self._clip_ctrl(np.asarray(u_seq[t], dtype=np.float64).reshape(self.nu))
            total += self._step_cost(data, ctrl, goal_pos, prev_ctrl) * dt
            data.ctrl[:] = ctrl
            mujoco.mj_step(self.model, data)
            prev_ctrl = ctrl

        return float(total)

    def get_action(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        qpos = np.asarray(qpos, dtype=np.float64).reshape(self.nq)
        qvel = np.asarray(qvel, dtype=np.float64).reshape(self.nv)

        # Nominal sequence from SAC policy.
        u_nom = self._rollout_policy_nominal(qpos, qvel)

        self._noise[:] = self._rng.standard_normal(size=self._noise.shape)
        self._noise *= float(self.noise_std)

        self._u_samples[:] = u_nom[None, :, :] + self._noise
        np.clip(self._u_samples, self.ctrl_min[None, None, :], self.ctrl_max[None, None, :], out=self._u_samples)

        # Goal pos from current state.
        data_tmp = self._data_pool[0]
        data_tmp.qpos[:] = qpos
        data_tmp.qvel[:] = qvel
        self._mujoco.mj_forward(self.model, data_tmp)
        goal_pos = self._goal_pos(data_tmp)

        for i in range(self.num_samples):
            self._costs[i] = self._rollout_cost_inplace(self._data_pool[i], qpos, qvel, self._u_samples[i], goal_pos)

        beta = float(np.min(self._costs))
        w = np.exp(-(self._costs - beta) / float(self.lambda_coeff)).astype(np.float64)
        w_sum = float(np.sum(w))
        if not np.isfinite(w_sum) or w_sum <= 1e-12:
            w = np.ones_like(w) / float(len(w))
        else:
            w = w / w_sum

        u_new = np.tensordot(w, self._u_samples, axes=(0, 0))
        u_new = np.clip(u_new, self.ctrl_min[None, :], self.ctrl_max[None, :])

        self.u = u_new
        action = self.u[0].copy()

        self.u[:-1] = self.u[1:]
        self.u[-1] = np.clip(qpos[: self.nu], self.ctrl_min, self.ctrl_max)

        return action
