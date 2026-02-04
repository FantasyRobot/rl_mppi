#!/usr/bin/env python3

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# Ensure repo root is importable regardless of CWD.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from env.envball_utils import BallEnvironment


class MPPI:
    """Model Predictive Path Integral (MPPI) controller for ball environment."""

    def __init__(
        self,
        env,
        horizon=20,
        num_samples=100,
        lambda_coeff=1.0,
        noise_std=0.5,
        dt=0.01,
        *,
        vectorized_rollouts: bool = True,
        obstacles: list[tuple[float, float, float]] | None = None,
        obstacle_margin: float = 0.6,
        obstacle_cost_coeff: float = 40000.0,
        obstacle_safety_distance: float = 0.1,
        collision_cost: float = 1e7,
        use_los_obstacle_cost: bool = True,
        los_influence: float = 0.8,
        los_cost_coeff: float = 20000.0,
        use_pd_nominal: bool = True,
        pd_kp: float = 2.0,
        pd_kd: float = 0.6,
        pd_repulse_gain: float = 1.5,
    ):
        self.env = env
        self.horizon = horizon
        self.num_samples = num_samples
        self.lambda_coeff = lambda_coeff
        self.noise_std = noise_std
        self.dt = dt
        self.vectorized_rollouts = bool(vectorized_rollouts)

        self.obstacles = list(obstacles) if obstacles is not None else []
        self.obstacle_margin = float(obstacle_margin)
        self.obstacle_cost_coeff = float(obstacle_cost_coeff)
        self.obstacle_safety_distance = float(obstacle_safety_distance)
        self.collision_cost = float(collision_cost)
        self.use_los_obstacle_cost = bool(use_los_obstacle_cost)
        self.los_influence = float(los_influence)
        self.los_cost_coeff = float(los_cost_coeff)

        self.use_pd_nominal = bool(use_pd_nominal)
        self.pd_kp = float(pd_kp)
        self.pd_kd = float(pd_kd)
        self.pd_repulse_gain = float(pd_repulse_gain)

        self.action_dim = env.action_dim
        self.state_dim = env.state_dim

        self.action_min = -1.0
        self.action_max = 1.0

        self.u = np.zeros((self.horizon, self.action_dim))

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

    def _rollout_pd_nominal(self, current_state: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
        """Build a nominal action sequence via a simple PD controller."""
        s = np.asarray(current_state, dtype=np.float32).copy()
        target_pos = np.asarray(target_pos, dtype=np.float32).reshape(2)
        u_nom = np.zeros((self.horizon, self.action_dim), dtype=np.float32)
        for t in range(self.horizon):
            pos_err = target_pos - s[:2]
            vel = s[2:4]
            desired_acc = self.pd_kp * pos_err - self.pd_kd * vel

            # Add a small repulsive acceleration away from obstacles inside the influence distance.
            if self.obstacles and self.pd_repulse_gain > 0.0:
                rep = np.zeros((2,), dtype=np.float32)
                influence = float(self.obstacle_margin)
                for (ox, oy, r) in self.obstacles:
                    dx = float(s[0] - float(ox))
                    dy = float(s[1] - float(oy))
                    dist = float(np.sqrt(dx * dx + dy * dy))
                    dist = max(dist, 1e-6)
                    clearance = dist - (float(r) + self.obstacle_safety_distance)
                    depth = max(0.0, influence - clearance)
                    if depth > 0.0:
                        rep += (np.asarray([dx, dy], dtype=np.float32) / np.float32(dist)) * (
                            np.float32(depth) * np.float32(self.pd_repulse_gain)
                        )
                desired_acc = desired_acc + rep
            a = desired_acc / float(self.env.acceleration_bound)
            a = np.clip(a, self.action_min, self.action_max)
            a = 0.7 * a + 0.3 * self.u[t]
            a = np.clip(a, self.action_min, self.action_max)
            u_nom[t] = a
            s = self._step_dynamics(s, a)
        return u_nom

    def compute_cost(self, state, action, target_pos):
        pos_error = state[:2] - target_pos
        pos_cost = np.linalg.norm(pos_error) * 1000.0
        action_cost = 0.001 * np.linalg.norm(action)
        total_cost = pos_cost + action_cost

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
                    total_cost += self.collision_cost
                depth = max(0.0, influence - clearance)
                if depth > 0.0:
                    total_cost += self.obstacle_cost_coeff * (depth * depth)

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
                            total_cost += self.los_cost_coeff * (depth_seg * depth_seg)
        return total_cost

    def simulate_trajectory(self, initial_state, action_sequence, target_pos):
        state = initial_state.copy()
        total_cost = 0.0
        trajectory = [state.copy()]

        for t in range(self.horizon):
            x, y, vx, vy = state
            action = action_sequence[t]
            ax = action[0] * self.env.acceleration_bound
            ay = action[1] * self.env.acceleration_bound

            ax = np.clip(ax, -self.env.acceleration_bound, self.env.acceleration_bound)
            ay = np.clip(ay, -self.env.acceleration_bound, self.env.acceleration_bound)

            new_vx = vx + ax * self.dt
            new_vy = vy + ay * self.dt
            new_x = x + new_vx * self.dt + 0.5 * ax * self.dt**2
            new_y = y + new_vy * self.dt + 0.5 * ay * self.dt**2

            new_x = np.clip(new_x, -self.env.pos_bound, self.env.pos_bound)
            new_y = np.clip(new_y, -self.env.pos_bound, self.env.pos_bound)
            new_vx = np.clip(new_vx, -self.env.vel_bound, self.env.vel_bound)
            new_vy = np.clip(new_vy, -self.env.vel_bound, self.env.vel_bound)

            next_state = np.array([new_x, new_y, new_vx, new_vy])

            cost = self.compute_cost(state, action, target_pos)
            total_cost += cost * self.dt

            state = next_state.copy()
            trajectory.append(state.copy())

        return total_cost, trajectory

    def simulate_trajectories_batch(self, initial_state, action_sequences, target_pos):
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

        pos_cost_coeff = np.float32(1000.0)
        act_cost_coeff = np.float32(0.001)

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
                los_pen = np.zeros((n,), dtype=np.float32)
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
                        los_pen += depth_seg * depth_seg
                step_cost = step_cost + obs_cost_coeff * obs_pen + los_cost_coeff * los_pen

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

    def get_action(self, current_state, target_pos):
        target_pos = np.asarray(target_pos, dtype=np.float32).reshape(2)

        if self.use_pd_nominal:
            u_nom = self._rollout_pd_nominal(current_state, target_pos)
        else:
            u_nom = np.asarray(self.u, dtype=np.float32)

        noise = np.random.randn(self.num_samples, self.horizon, self.action_dim).astype(np.float32) * self.noise_std
        action_sequences = u_nom[None, :, :] + noise
        action_sequences = np.clip(action_sequences, self.action_min, self.action_max)

        if self.vectorized_rollouts:
            costs = self.simulate_trajectories_batch(current_state, action_sequences, target_pos)
        else:
            costs = np.zeros(self.num_samples)
            for i in range(self.num_samples):
                cost, _traj = self.simulate_trajectory(current_state, action_sequences[i], target_pos)
                costs[i] = cost

        cost_min = np.min(costs)
        weights = np.exp(-(costs - cost_min) / self.lambda_coeff)
        weights /= np.sum(weights)

        weighted_actions = np.sum(weights[:, np.newaxis, np.newaxis] * action_sequences, axis=0)

        self.u[:-1] = weighted_actions[1:]
        self.u[-1] = weighted_actions[-1]

        return weighted_actions[0]

    def reset(self):
        self.u = np.zeros((self.horizon, self.action_dim))


def test_mppi_ball():
    target_pos = [5.0, 5.0]
    env = BallEnvironment(target_pos=target_pos)

    mppi = MPPI(
        env=env,
        horizon=15,
        num_samples=50,
        lambda_coeff=0.01,
        noise_std=2.0,
        dt=0.01,
    )

    max_steps = 1000
    initial_state = env.reset(initial_state=[-3.0, -3.0, 0.0, 0.0])
    state = initial_state.copy()
    trajectory = [state.copy()]

    print("Testing MPPI controller on BallEnvironment...")
    print(f"Target position: {target_pos}")
    print(f"Initial state: {state}")

    for step in range(max_steps):
        if step % 50 == 0:
            print(f"Step {step}: dist={np.linalg.norm(state[:2] - target_pos):.4f}")

        action = mppi.get_action(state, target_pos)
        next_state, reward, done, info = env.step(action)

        state = next_state.copy()
        trajectory.append(state.copy())

        distance = np.linalg.norm(state[:2] - target_pos)
        if distance < 0.5:
            print(f"Reached target at step {step}!")
            break

    trajectory = np.array(trajectory)

    plt.figure(figsize=(10, 10))
    plt.plot(trajectory[:, 0], trajectory[:, 1], "b-", linewidth=2, label="Ball trajectory")
    plt.plot(target_pos[0], target_pos[1], "go", markersize=10, label="Target")
    plt.plot(initial_state[0], initial_state[1], "ro", markersize=8, label="Initial position")

    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("MPPI Controller - Ball Trajectory")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)

    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results_dir = os.path.join(root_dir, "experiments", "results")
    os.makedirs(results_dir, exist_ok=True)
    plot_path = os.path.join(results_dir, "mppi_ball_trajectory.png")
    plt.savefig(plot_path)
    print(f"Trajectory plot saved as '{plot_path}'")
    plt.close()

    print(f"Final state: {state}")


if __name__ == "__main__":
    test_mppi_ball()
