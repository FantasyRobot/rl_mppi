#!/usr/bin/env python3

from __future__ import annotations

import numpy as np

from env.envrobot2d_obstacles import Robot2DEnvironmentObstacles


class MPPI:
    """MPPI controller for Robot2DEnvironmentObstacles.

    This is a Robot2D-specific variant to keep the existing ball MPPI unchanged.
    """

    def __init__(
        self,
        env: Robot2DEnvironmentObstacles,
        *,
        horizon: int = 25,
        num_samples: int = 300,
        lambda_coeff: float = 0.6,
        noise_std: float = 0.6,
        vectorized_rollouts: bool = True,
        action_cost_coeff: float = 0.002,
        pos_cost_coeff: float = 1200.0,
        obstacle_margin: float | None = None,
        obstacle_cost_coeff: float = 6000.0,
        collision_cost: float = 2e6,
        use_pd_nominal: bool = True,
        pd_kp: float = 6.0,
        pd_kd: float = 2.0,
        pd_repulse_gain: float = 1.0,
    ):
        self.env = env
        self.horizon = int(horizon)
        self.num_samples = int(num_samples)
        self.lambda_coeff = float(lambda_coeff)
        self.noise_std = float(noise_std)
        self.vectorized_rollouts = bool(vectorized_rollouts)

        self.action_cost_coeff = float(action_cost_coeff)
        self.pos_cost_coeff = float(pos_cost_coeff)

        self.obstacle_margin = float(env.obstacle_margin if obstacle_margin is None else obstacle_margin)
        self.obstacle_cost_coeff = float(obstacle_cost_coeff)
        self.collision_cost = float(collision_cost)

        self.use_pd_nominal = bool(use_pd_nominal)
        self.pd_kp = float(pd_kp)
        self.pd_kd = float(pd_kd)
        self.pd_repulse_gain = float(pd_repulse_gain)

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

    def _nominal_pd(self, state: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
        """Jacobian-transpose task-space PD nominal (in joint acceleration)."""
        n = self.env.n
        s = np.asarray(state, dtype=np.float32).copy()
        target_pos = np.asarray(target_pos, dtype=np.float32).reshape(2)

        u_nom = np.zeros((self.horizon, self.action_dim), dtype=np.float32)
        for t in range(self.horizon):
            q = s[:n]
            qd = s[n : 2 * n]
            eef = self.env.forward_kinematics_eef(q)
            J = self.env.jacobian_eef(q)
            eef_vel = J @ qd

            pos_err = target_pos - eef
            task_acc = self.pd_kp * pos_err - self.pd_kd * eef_vel

            # Repel from obstacles in task space (end-effector).
            if self.env.obstacles and self.pd_repulse_gain > 0.0:
                rep = np.zeros((2,), dtype=np.float32)
                for obs in self.env.obstacles:
                    c = obs.center
                    v = eef - c
                    dist = float(np.linalg.norm(v))
                    dist = max(dist, 1e-6)
                    clearance = dist - float(obs.r) - float(self.env.safety_distance)
                    depth = max(0.0, float(self.obstacle_margin) - clearance)
                    if depth > 0.0:
                        rep += (v / np.float32(dist)) * (np.float32(depth) * np.float32(self.pd_repulse_gain))
                task_acc = task_acc + rep

            qdd_des = (J.T @ task_acc).astype(np.float32)
            a = qdd_des / float(self.env.qdd_max)
            a = np.clip(a, -1.0, 1.0)
            a = 0.7 * a + 0.3 * self.u[t]
            a = np.clip(a, -1.0, 1.0)
            u_nom[t] = a
            s = self._step_dynamics(s, a)

        return u_nom

    def _arm_min_clearance(self, q: np.ndarray) -> tuple[float, bool]:
        """Minimum clearance of any link segment to any obstacle.

        Clearance is computed against obstacle radius inflated by env.safety_distance.
        """

        if not self.env.obstacles:
            return float("inf"), False

        pts = self.env.forward_kinematics_all_joints(q)
        safety = float(self.env.safety_distance)

        min_clear = float("inf")
        collided = False

        for obs in self.env.obstacles:
            cx = float(obs.x)
            cy = float(obs.y)
            rr = float(obs.r) + safety
            for i in range(self.env.n):
                ax = float(pts[i, 0])
                ay = float(pts[i, 1])
                bx = float(pts[i + 1, 0])
                by = float(pts[i + 1, 1])

                abx = bx - ax
                aby = by - ay
                apx = cx - ax
                apy = cy - ay

                denom = abx * abx + aby * aby
                if denom <= 1e-12:
                    dx = apx
                    dy = apy
                    dist = float(np.sqrt(dx * dx + dy * dy))
                else:
                    t = (apx * abx + apy * aby) / denom
                    t = float(np.clip(t, 0.0, 1.0))
                    projx = ax + t * abx
                    projy = ay + t * aby
                    dx = cx - projx
                    dy = cy - projy
                    dist = float(np.sqrt(dx * dx + dy * dy))

                clear = dist - rr
                if clear < min_clear:
                    min_clear = clear
                if clear < 0.0:
                    collided = True

        return float(min_clear), bool(collided)

    def _cost_step(self, state: np.ndarray, action: np.ndarray, target_pos: np.ndarray) -> float:
        n = self.env.n
        q = state[:n]
        eef = self.env.forward_kinematics_eef(q)
        pos_cost = float(np.linalg.norm(eef - target_pos)) * self.pos_cost_coeff
        act_cost = self.action_cost_coeff * float(np.linalg.norm(action))
        total = pos_cost + act_cost

        # Obstacle penalty: full-arm clearance/collision (all link segments).
        if self.env.obstacles:
            min_clear, collided = self._arm_min_clearance(q)
            if collided:
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
        if u_seqs.ndim != 3:
            raise ValueError("u_seqs must be (N,H,dim)")
        N, H, D = u_seqs.shape
        if H != self.horizon or D != self.action_dim:
            raise ValueError("u_seqs shape mismatch")

        # Vectorization of full robot kinematics/collision is non-trivial; keep loop over samples.
        # This is still fine for small horizons.
        costs = np.zeros((N,), dtype=np.float32)
        for i in range(N):
            costs[i] = np.float32(self._simulate_cost(s0, u_seqs[i], target_pos))
        return costs

    def get_action(self, state: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32).reshape(self.state_dim)
        target_pos = np.asarray(target_pos, dtype=np.float32).reshape(2)

        u_nom = self._nominal_pd(state, target_pos) if self.use_pd_nominal else self.u

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
