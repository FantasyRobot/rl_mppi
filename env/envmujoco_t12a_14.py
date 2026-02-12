#!/usr/bin/env python3

from __future__ import annotations

import os
from contextlib import contextmanager

import numpy as np


@contextmanager
def _pushd(path: str):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


class T12A14MuJoCoEnv:
    """Minimal MuJoCo env wrapper for t12a_14.

    - Action: normalized in [-1, 1] for each actuator.
      Mapped to actuator ctrlrange (position targets).
    - Observation (default):
        [qpos_norm (nu), qvel_norm (nv), (eef - goal) (3)]

    This env is intentionally lightweight (no gym dependency) and matches the
    project style used in other SAC scripts.
    """

    def __init__(
        self,
        *,
        xml_path: str,
        goal_site: str = "goal",
        eef_site: str = "end_effector",
        max_steps: int = 2500,
        reach_tol: float = 0.03,
        action_repeat: int = 5,
        qvel_scale: float = 5.0,
        reward_dist_coeff: float = 1.0,
        reward_ctrl_coeff: float = 0.01,
        success_bonus: float = 10.0,
        seed: int = 0,
        reset_noise: float = 0.05,
    ):
        try:
            import mujoco  # noqa: F401
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError("T12A14MuJoCoEnv requires 'mujoco' (pip install mujoco).") from e

        import mujoco

        self.xml_path = os.path.abspath(os.path.expandvars(os.path.expanduser(str(xml_path))))
        if not os.path.exists(self.xml_path):
            raise FileNotFoundError(f"XML not found: {self.xml_path}")

        self.goal_site = str(goal_site)
        self.eef_site = str(eef_site)

        self.max_steps = int(max_steps)
        self.reach_tol = float(reach_tol)
        self.action_repeat = int(max(1, action_repeat))
        self.qvel_scale = float(qvel_scale)

        self.reward_dist_coeff = float(reward_dist_coeff)
        self.reward_ctrl_coeff = float(reward_ctrl_coeff)
        self.success_bonus = float(success_bonus)

        self.reset_noise = float(reset_noise)

        self._rng = np.random.default_rng(int(seed))

        xml_dir = os.path.dirname(self.xml_path)
        with _pushd(xml_dir):
            self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        self._mujoco = mujoco

        self.nu = int(self.model.nu)
        self.nq = int(self.model.nq)
        self.nv = int(self.model.nv)
        if self.nu <= 0:
            raise ValueError("Model has no actuators (model.nu == 0).")

        # ctrlrange for position actuators
        self.ctrl_min = np.asarray(self.model.actuator_ctrlrange[:, 0], dtype=np.float64)
        self.ctrl_max = np.asarray(self.model.actuator_ctrlrange[:, 1], dtype=np.float64)
        self.ctrl_center = 0.5 * (self.ctrl_min + self.ctrl_max)
        self.ctrl_half = 0.5 * (self.ctrl_max - self.ctrl_min)
        self.ctrl_half = np.where(self.ctrl_half == 0.0, 1.0, self.ctrl_half)

        self.goal_sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.goal_site)
        self.eef_sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.eef_site)
        if self.goal_sid < 0:
            raise ValueError(f"Goal site not found: {self.goal_site}")
        if self.eef_sid < 0:
            raise ValueError(f"EEF site not found: {self.eef_site}")

        self._step_count = 0
        self.goal_pos = np.zeros((3,), dtype=np.float64)

        # Exposed dims for SAC.
        self.action_dim = int(self.nu)
        self.obs_dim = int(self.nu + self.nv + 3)

    def _update_goal(self) -> None:
        self._mujoco.mj_forward(self.model, self.data)
        self.goal_pos = np.asarray(self.data.site_xpos[self.goal_sid], dtype=np.float64).copy()

    def _eef_pos(self) -> np.ndarray:
        return np.asarray(self.data.site_xpos[self.eef_sid], dtype=np.float64).copy()

    def action_to_ctrl(self, action_norm: np.ndarray) -> np.ndarray:
        a = np.asarray(action_norm, dtype=np.float64).reshape(self.nu)
        a = np.clip(a, -1.0, 1.0)
        return np.clip(self.ctrl_center + a * self.ctrl_half, self.ctrl_min, self.ctrl_max)

    def ctrl_to_action(self, ctrl: np.ndarray) -> np.ndarray:
        c = np.asarray(ctrl, dtype=np.float64).reshape(self.nu)
        a = (c - self.ctrl_center) / self.ctrl_half
        return np.clip(a, -1.0, 1.0)

    def get_obs(self) -> np.ndarray:
        q = np.asarray(self.data.qpos[: self.nu], dtype=np.float64)
        qn = (q - self.ctrl_center) / self.ctrl_half
        qn = np.clip(qn, -1.0, 1.0)

        qv = np.asarray(self.data.qvel, dtype=np.float64)
        denom = float(self.qvel_scale) if float(self.qvel_scale) != 0.0 else 1.0
        qvn = np.clip(qv / denom, -5.0, 5.0)

        eef = self._eef_pos()
        rel = (eef - self.goal_pos).astype(np.float64)

        return np.concatenate([qn, qvn, rel], axis=0).astype(np.float32)

    def reset(self) -> np.ndarray:
        self._step_count = 0

        q0 = self._rng.normal(0.0, float(self.reset_noise), size=(self.nu,)).astype(np.float64)
        q0 = np.clip(q0, self.ctrl_min, self.ctrl_max)
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.qpos[: self.nu] = q0

        self._mujoco.mj_forward(self.model, self.data)
        self._update_goal()
        return self.get_obs()

    def step(self, action_norm: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        ctrl = self.action_to_ctrl(action_norm)

        for _ in range(self.action_repeat):
            self.data.ctrl[:] = ctrl
            self._mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        eef = self._eef_pos()
        dist = float(np.linalg.norm(eef - self.goal_pos))

        reward = -self.reward_dist_coeff * dist - self.reward_ctrl_coeff * float(np.linalg.norm(action_norm))

        done = False
        success = dist < float(self.reach_tol)
        if success:
            reward += float(self.success_bonus)
            done = True

        if self._step_count >= int(self.max_steps):
            done = True

        info = {
            "dist": float(dist),
            "eef": eef.astype(np.float32),
            "goal": self.goal_pos.astype(np.float32),
            "success": bool(success),
            "step": int(self._step_count),
        }

        return self.get_obs(), float(reward), bool(done), info
