#!/usr/bin/env python3

from __future__ import annotations

import numpy as np


class MuJoCoArmMPPI:
    """MPPI controller for a MuJoCo arm with position actuators.

    This follows the same high-level structure as the existing MPPI controllers in
    `algorithms/mppi/`:
    - keep an internal control sequence `u` of length horizon
    - sample noisy control sequences around a nominal sequence
    - roll out dynamics and compute trajectory costs
    - update `u` with a path-integral weighted average, apply receding horizon

    Actions are interpreted as *actuator position commands* (data.ctrl), clipped
    to actuator ctrlrange.
    """

    def __init__(
        self,
        model,
        *,
        eef_site: str = "end_effector",
        horizon: int = 25,
        num_samples: int = 96,
        lambda_coeff: float = 1.0,
        noise_std: float = 0.06,
        pos_cost_coeff: float = 200.0,
        action_cost_coeff: float = 0.02,
        smooth_cost_coeff: float = 0.2,
        use_jt_nominal: bool = True,
        jt_gain: float = 0.6,
        jt_damping: float = 1e-4,
        seed: int | None = None,
    ):
        try:
            import mujoco  # noqa: F401
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError("MuJoCoArmMPPI requires the 'mujoco' package (pip install mujoco).") from e

        self.model = model
        self.horizon = int(horizon)
        self.num_samples = int(num_samples)
        self.lambda_coeff = float(lambda_coeff)
        self.noise_std = float(noise_std)

        self.pos_cost_coeff = float(pos_cost_coeff)
        self.action_cost_coeff = float(action_cost_coeff)
        self.smooth_cost_coeff = float(smooth_cost_coeff)

        self.use_jt_nominal = bool(use_jt_nominal)
        self.jt_gain = float(jt_gain)
        self.jt_damping = float(jt_damping)

        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        else:
            self._rng = np.random.default_rng()

        self.nu = int(model.nu)
        if self.nu <= 0:
            raise ValueError("Model has no actuators (model.nu == 0).")

        # Use float64 to match MuJoCo state buffers and avoid repeated casts.
        self.ctrl_min = np.asarray(model.actuator_ctrlrange[:, 0], dtype=np.float64)
        self.ctrl_max = np.asarray(model.actuator_ctrlrange[:, 1], dtype=np.float64)

        # We assume position actuators driving the joint positions.
        # `qpos` dimension includes all generalized coordinates, but for this model it is the 6 joints.
        self.nq = int(model.nq)
        self.nv = int(model.nv)

        import mujoco

        self._mujoco = mujoco
        self.eef_site_name = str(eef_site)
        self.eef_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self.eef_site_name)
        if self.eef_site_id < 0:
            raise ValueError(f"EEF site not found: {self.eef_site_name}")

        # Receding-horizon control sequence in ctrl-space.
        self.u = np.zeros((self.horizon, self.nu), dtype=np.float64)

        # Scratch buffers to reduce allocations.
        self._jacp = np.zeros((3, self.nv), dtype=np.float64)
        self._jacr = np.zeros((3, self.nv), dtype=np.float64)

        # Reuse data objects for rollouts (allocation is expensive).
        self._data_nominal = mujoco.MjData(self.model)
        self._data_pool = [mujoco.MjData(self.model) for _ in range(self.num_samples)]

        # Preallocate sampling buffers to reduce per-step allocations.
        self._noise = np.empty((self.num_samples, self.horizon, self.nu), dtype=np.float64)
        self._u_samples = np.empty((self.num_samples, self.horizon, self.nu), dtype=np.float64)
        self._costs = np.empty((self.num_samples,), dtype=np.float64)

    def reset(self) -> None:
        self.u[:] = 0.0

    def _clip_ctrl(self, ctrl: np.ndarray) -> np.ndarray:
        return np.clip(ctrl, self.ctrl_min, self.ctrl_max)

    def _eef_pos(self, data) -> np.ndarray:
        return np.asarray(data.site_xpos[self.eef_site_id], dtype=np.float64).copy()

    def _nominal_jt(self, qpos0: np.ndarray, qvel0: np.ndarray, goal_pos: np.ndarray) -> np.ndarray:
        """Build a nominal ctrl sequence using a Jacobian-transpose heuristic."""
        mujoco = self._mujoco
        data = self._data_nominal
        data.qpos[:] = qpos0
        data.qvel[:] = qvel0
        mujoco.mj_forward(self.model, data)

        goal_pos = np.asarray(goal_pos, dtype=np.float64).reshape(3)
        u_nom = np.zeros((self.horizon, self.nu), dtype=np.float64)

        prev_ctrl = np.clip(np.asarray(qpos0[: self.nu], dtype=np.float64), self.ctrl_min, self.ctrl_max)

        for t in range(self.horizon):
            eef = self._eef_pos(data)
            err = (goal_pos - eef)  # 3D

            mujoco.mj_jacSite(self.model, data, self._jacp, self._jacr, self.eef_site_id)
            J = self._jacp  # (3, nv)

            # Damped JT: dq ~ gain * J^T * err / (||J||^2 + damping)
            denom = float(np.sum(J * J) + self.jt_damping)
            dq = (self.jt_gain / denom) * (J.T @ err)  # (nv,)

            # Map to ctrl targets (position actuators): q_des = q + dq
            q = np.asarray(data.qpos[: self.nu], dtype=np.float64)
            q_des = q + dq[: self.nu]

            ctrl = self._clip_ctrl(np.asarray(q_des, dtype=np.float64))

            # Blend with receding horizon memory to keep it smooth.
            ctrl = self._clip_ctrl(0.7 * ctrl + 0.3 * self.u[t])

            # A small smoothness nudge.
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

        # Penalize deviation from current qpos (keeps commands reasonable).
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

    def get_action(self, qpos: np.ndarray, qvel: np.ndarray, goal_pos: np.ndarray) -> np.ndarray:
        # Accept MuJoCo buffers directly (float64) to avoid copies.
        qpos = np.asarray(qpos, dtype=np.float64).reshape(self.nq)
        qvel = np.asarray(qvel, dtype=np.float64).reshape(self.nv)
        goal_pos = np.asarray(goal_pos, dtype=np.float64).reshape(3)

        if self.use_jt_nominal:
            u_nom = self._nominal_jt(qpos, qvel, goal_pos)
        else:
            u_nom = self.u

        # Fill preallocated noise buffer without extra allocations.
        # (Generator.normal(out=...) is not consistently supported across NumPy builds)
        self._noise[:] = self._rng.standard_normal(size=self._noise.shape)
        self._noise *= float(self.noise_std)
        # u_samples = clip(u_nom + noise)
        self._u_samples[:] = u_nom[None, :, :] + self._noise
        np.clip(self._u_samples, self.ctrl_min[None, None, :], self.ctrl_max[None, None, :], out=self._u_samples)

        for i in range(self.num_samples):
            self._costs[i] = self._rollout_cost_inplace(self._data_pool[i], qpos, qvel, self._u_samples[i], goal_pos)

        costs = self._costs

        beta = float(np.min(costs))
        w = np.exp(-(costs - beta) / float(self.lambda_coeff)).astype(np.float64)
        w_sum = float(np.sum(w))
        if not np.isfinite(w_sum) or w_sum <= 1e-12:
            w = np.ones_like(w) / float(len(w))
        else:
            w = w / w_sum

        u_new = np.tensordot(w, self._u_samples, axes=(0, 0))
        u_new = np.clip(u_new, self.ctrl_min[None, :], self.ctrl_max[None, :])

        self.u = u_new
        action = self.u[0].copy()

        # Receding horizon shift.
        self.u[:-1] = self.u[1:]
        self.u[-1] = np.clip(qpos[: self.nu], self.ctrl_min, self.ctrl_max)

        return action
