#!/usr/bin/env python3

import os
import sys

import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

# Ensure repo root is importable regardless of CWD.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from env.envball_utils import BallEnvironment


class BallMPC:
    """Model Predictive Control (MPC) controller for 2D ball environment using CasADi."""

    def __init__(
        self,
        env,
        horizon=10,
        dt=0.01,
        solver_opts=None,
        *,
        obstacles: list[tuple[float, float, float]] | None = None,
        obstacle_margin: float = 0.0,
        obstacle_safety_distance: float = 0.0,
        obstacle_cost_coeff: float = 0.0,
        collision_cost: float = 0.0,
        obstacle_smooth_eps: float = 1e-6,
        use_obstacle_constraints: bool = False,
    ):
        self.env = env
        self.horizon = horizon
        self.dt = dt

        self.obstacles = list(obstacles) if obstacles else []
        self.obstacle_margin = float(obstacle_margin)
        self.obstacle_safety_distance = float(obstacle_safety_distance)
        self.obstacle_cost_coeff = float(obstacle_cost_coeff)
        self.collision_cost = float(collision_cost)
        self.obstacle_smooth_eps = float(obstacle_smooth_eps)
        self.use_obstacle_constraints = bool(use_obstacle_constraints)

        self.state_dim = env.state_dim
        self.action_dim = env.action_dim

        self.state_bounds = {
            "x_min": -env.pos_bound,
            "x_max": env.pos_bound,
            "y_min": -env.pos_bound,
            "y_max": env.pos_bound,
            "vx_min": -env.vel_bound,
            "vx_max": env.vel_bound,
            "vy_min": -env.vel_bound,
            "vy_max": env.vel_bound,
        }

        self.control_bounds = {
            "ax_min": -env.acceleration_bound,
            "ax_max": env.acceleration_bound,
            "ay_min": -env.acceleration_bound,
            "ay_max": env.acceleration_bound,
        }

        self.target_pos = env.target_pos

        self.setup_mpc_problem()

        if solver_opts is None:
            solver_opts = {
                "ipopt.max_iter": 50,
                "ipopt.print_level": 0,
                "ipopt.tol": 1e-3,
                "ipopt.acceptable_tol": 1e-2,
                "ipopt.acceptable_iter": 5,
                "ipopt.warm_start_init_point": "yes",
                "ipopt.warm_start_bound_push": 1e-9,
                "ipopt.warm_start_bound_frac": 1e-9,
                "print_time": 0,
            }

        self.solver = ca.nlpsol("solver", "ipopt", self.nlp, solver_opts)

        self.prev_solution = None

        self.kp_pos = 2.0
        self.kd_pos = 0.5
        self.prev_pos_error = None

    def setup_mpc_problem(self):
        x = ca.MX.sym("x", self.state_dim, self.horizon + 1)
        u = ca.MX.sym("u", self.action_dim, self.horizon)

        x0 = ca.MX.sym("x0", self.state_dim)
        target = ca.MX.sym("target", 2)

        w_pos = 10.0
        w_vel = 5.0
        w_control = 1.0
        w_terminal = 50.0

        cost = 0.0
        constraints: list[ca.MX] = []
        lbg: list[float] = []
        ubg: list[float] = []

        inf = 1e19

        constraints.append(x[0, 0] - x0[0]); lbg.append(0.0); ubg.append(0.0)
        constraints.append(x[1, 0] - x0[1]); lbg.append(0.0); ubg.append(0.0)
        constraints.append(x[2, 0] - x0[2]); lbg.append(0.0); ubg.append(0.0)
        constraints.append(x[3, 0] - x0[3]); lbg.append(0.0); ubg.append(0.0)

        def smooth_pos_part(v: ca.MX) -> ca.MX:
            # Smooth approximation of max(v, 0)
            return 0.5 * (v + ca.sqrt(v * v + self.obstacle_smooth_eps))

        def add_obstacle_cost(xk: ca.MX, yk: ca.MX) -> ca.MX:
            if not self.obstacles:
                return 0.0

            total = 0.0
            for (ox, oy, r) in self.obstacles:
                ox = float(ox)
                oy = float(oy)
                r = float(r)

                # Always define a "safe" radius; inside it is collision penetration.
                r_safe = r + self.obstacle_safety_distance
                dist2 = (xk - ox) ** 2 + (yk - oy) ** 2

                if self.collision_cost > 0.0:
                    v_col = (r_safe * r_safe) - dist2
                    pen_col = smooth_pos_part(v_col)
                    total += float(self.collision_cost) * (pen_col ** 2)

                # Influence zone encourages earlier detours (like MPPI obstacle_margin).
                if self.obstacle_cost_coeff > 0.0 and self.obstacle_margin > 0.0:
                    r_inf = r_safe + self.obstacle_margin
                    v_inf = (r_inf * r_inf) - dist2
                    pen_inf = smooth_pos_part(v_inf)
                    # Normalize by influence thickness so typical magnitudes are ~O(1)
                    denom = (r_inf * r_inf) - (r_safe * r_safe) + self.obstacle_smooth_eps
                    total += float(self.obstacle_cost_coeff) * ((pen_inf / denom) ** 2)

            return total

        for k in range(self.horizon):
            x_k = x[0, k]
            y_k = x[1, k]
            vx_k = x[2, k]
            vy_k = x[3, k]

            ax_k = u[0, k]
            ay_k = u[1, k]

            # Match env.envball_utils.BallEnvironment.step() kinematics:
            # v_{k+1} = v_k + a_k * dt
            # p_{k+1} = p_k + v_{k+1} * dt + 0.5 * a_k * dt^2
            vx_kp1 = vx_k + ax_k * self.dt
            vy_kp1 = vy_k + ay_k * self.dt
            x_kp1 = x_k + vx_kp1 * self.dt + 0.5 * ax_k * (self.dt ** 2)
            y_kp1 = y_k + vy_kp1 * self.dt + 0.5 * ay_k * (self.dt ** 2)

            constraints.append(x[0, k + 1] - x_kp1)
            constraints.append(x[1, k + 1] - y_kp1)
            constraints.append(x[2, k + 1] - vx_kp1)
            constraints.append(x[3, k + 1] - vy_kp1)
            lbg.extend([0.0, 0.0, 0.0, 0.0])
            ubg.extend([0.0, 0.0, 0.0, 0.0])

            pos_error = ca.vertcat(x_k - target[0], y_k - target[1])
            control_effort = ca.vertcat(ax_k, ay_k)

            cost += w_pos * ca.norm_2(pos_error) ** 2 + w_control * ca.norm_2(control_effort) ** 2

            cost += add_obstacle_cost(x_k, y_k)

        terminal_pos_error = ca.vertcat(x[0, -1] - target[0], x[1, -1] - target[1])
        terminal_vel = ca.vertcat(x[2, -1], x[3, -1])

        cost += w_terminal * ca.norm_2(terminal_pos_error) ** 2 + w_vel * ca.norm_2(terminal_vel) ** 2

        # Also penalize terminal state proximity to obstacles (helps avoid "cutting corners").
        cost += add_obstacle_cost(x[0, -1], x[1, -1])

        # Optional hard constraints to keep predicted positions outside obstacles.
        if self.use_obstacle_constraints and self.obstacles:
            for k in range(self.horizon + 1):
                xk = x[0, k]
                yk = x[1, k]
                for (ox, oy, r) in self.obstacles:
                    ox = float(ox)
                    oy = float(oy)
                    r_safe = float(r) + self.obstacle_safety_distance
                    dist2 = (xk - ox) ** 2 + (yk - oy) ** 2
                    constraints.append(dist2 - (r_safe * r_safe))
                    lbg.append(0.0)
                    ubg.append(inf)

        lbx = []
        ubx = []

        for k in range(self.horizon + 1):
            lbx.extend([
                self.state_bounds["x_min"],
                self.state_bounds["y_min"],
                self.state_bounds["vx_min"],
                self.state_bounds["vy_min"],
            ])
            ubx.extend([
                self.state_bounds["x_max"],
                self.state_bounds["y_max"],
                self.state_bounds["vx_max"],
                self.state_bounds["vy_max"],
            ])

        for k in range(self.horizon):
            lbx.extend([
                self.control_bounds["ax_min"],
                self.control_bounds["ay_min"],
            ])
            ubx.extend([
                self.control_bounds["ax_max"],
                self.control_bounds["ay_max"],
            ])

        self.nlp = {
            "x": ca.vertcat(ca.reshape(x, -1, 1), ca.reshape(u, -1, 1)),
            "p": ca.vertcat(x0, target),
            "f": cost,
            "g": ca.vertcat(*constraints),
        }

        self.lbx = lbx
        self.ubx = ubx
        self.lbg = lbg
        self.ubg = ubg

    def pid_fallback(self, state, target_pos):
        current_pos = state[:2]
        current_vel = state[2:]

        pos_error = target_pos - current_pos
        vel_error = -current_vel

        if self.prev_pos_error is not None:
            _ = (pos_error - self.prev_pos_error) / self.dt
        else:
            _ = np.zeros(2)

        self.prev_pos_error = pos_error.copy()

        control_action = self.kp_pos * pos_error + self.kd_pos * vel_error

        # Obstacle-aware repulsion to keep the fallback controller safe.
        if self.obstacles and (self.obstacle_margin > 0.0 or self.obstacle_safety_distance > 0.0):
            repulse = np.zeros(2, dtype=float)
            eps = 1e-6
            # Tuned for env.acceleration_bound==1.0; clipped below anyway.
            repulse_gain = 1.4

            for (ox, oy, r) in self.obstacles:
                ox = float(ox)
                oy = float(oy)
                r = float(r)

                r_safe = r + float(self.obstacle_safety_distance)
                influence = r_safe + max(float(self.obstacle_margin), 0.0)

                delta = np.asarray([float(current_pos[0]) - ox, float(current_pos[1]) - oy], dtype=float)
                dist = float(np.linalg.norm(delta) + eps)

                if dist >= influence:
                    continue

                direction = delta / dist
                gap = dist - r_safe
                band = max(influence - r_safe, eps)

                if gap <= 0.0:
                    strength = 5.0
                else:
                    strength = repulse_gain * (1.0 / (gap + eps) - 1.0 / (band + eps))
                    strength = float(max(strength, 0.0))

                repulse += direction * strength

            control_action = control_action + repulse

        control_action = np.clip(
            control_action,
            [self.control_bounds["ax_min"], self.control_bounds["ay_min"]],
            [self.control_bounds["ax_max"], self.control_bounds["ay_max"]],
        )

        return control_action

    def generate_initial_guess(self, current_state, target_pos):
        if self.prev_solution is not None:
            return self.prev_solution

        # Build a dynamics-consistent guess using the same discrete model as the MPC constraints.
        x_guess: list[float] = []
        u_guess: list[float] = []

        s = np.asarray(current_state, dtype=float).reshape(4)
        target_pos = np.asarray(target_pos, dtype=float).reshape(2)

        # x[0]
        x_guess.extend([float(s[0]), float(s[1]), float(s[2]), float(s[3])])

        for _k in range(self.horizon):
            u = self.pid_fallback(s, target_pos)
            u = np.asarray(u, dtype=float).reshape(2)
            u_guess.extend([float(u[0]), float(u[1])])

            # Discrete dynamics used in constraints (matches environment):
            vx_new = s[2] + u[0] * self.dt
            vy_new = s[3] + u[1] * self.dt
            x_new = s[0] + vx_new * self.dt + 0.5 * u[0] * (self.dt ** 2)
            y_new = s[1] + vy_new * self.dt + 0.5 * u[1] * (self.dt ** 2)
            s = np.asarray([x_new, y_new, vx_new, vy_new], dtype=float)

            # Keep guess within state bounds to avoid solver issues.
            s[0] = float(np.clip(s[0], self.state_bounds["x_min"], self.state_bounds["x_max"]))
            s[1] = float(np.clip(s[1], self.state_bounds["y_min"], self.state_bounds["y_max"]))
            s[2] = float(np.clip(s[2], self.state_bounds["vx_min"], self.state_bounds["vx_max"]))
            s[3] = float(np.clip(s[3], self.state_bounds["vy_min"], self.state_bounds["vy_max"]))

            x_guess.extend([float(s[0]), float(s[1]), float(s[2]), float(s[3])])

        initial_guess = np.array(x_guess + u_guess)
        initial_guess = np.nan_to_num(initial_guess, nan=0.0, posinf=1.0, neginf=-1.0)

        return initial_guess

    def solve_mpc(self, current_state, target_pos):
        params = np.concatenate([current_state, target_pos])
        initial_guess = self.generate_initial_guess(current_state, target_pos)

        try:
            sol = self.solver(x0=initial_guess, p=params, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg)

            state_dim_total = self.state_dim * (self.horizon + 1)
            x_sol = sol["x"][:state_dim_total]
            u_sol = sol["x"][state_dim_total:]

            x_sol_reshaped = ca.reshape(x_sol, self.state_dim, self.horizon + 1)
            u_sol_reshaped = ca.reshape(u_sol, self.action_dim, self.horizon)

            x_sol_np = np.array(x_sol_reshaped.full())
            u_sol_np = np.array(u_sol_reshaped.full())

            if np.isnan(x_sol_np).any() or np.isinf(x_sol_np).any() or np.isnan(u_sol_np).any() or np.isinf(u_sol_np).any():
                raise ValueError("Solution contains NaN or inf values")

            full_sol = sol["x"].full().flatten()
            if not (np.isnan(full_sol).any() or np.isinf(full_sol).any()):
                self.prev_solution = full_sol

            return u_sol_np[:, 0], x_sol_np.T

        except Exception:
            pid_action = self.pid_fallback(current_state, target_pos)
            return pid_action, None

    def reset(self):
        self.prev_solution = None
        self.prev_pos_error = None


class MPCController:
    """Thin wrapper to use MPC in common experiment loops.

    Exposes a unified controller API:
        - reset()
        - get_action(state, target_pos) -> action in [-1, 1]
    """

    def __init__(
        self,
        *,
        env: BallEnvironment,
        horizon: int = 10,
        obstacles: list[tuple[float, float, float]] | None = None,
        obstacle_margin: float = 0.0,
        obstacle_safety_distance: float = 0.0,
        obstacle_cost_coeff: float = 0.0,
        collision_cost: float = 0.0,
        obstacle_smooth_eps: float = 1e-6,
        use_obstacle_constraints: bool = False,
        solver_opts: dict | None = None,
    ):
        self.env = env
        self.mpc = BallMPC(
            env=env,
            horizon=int(horizon),
            dt=float(env.dt),
            solver_opts=solver_opts,
            obstacles=obstacles,
            obstacle_margin=float(obstacle_margin),
            obstacle_safety_distance=float(obstacle_safety_distance),
            obstacle_cost_coeff=float(obstacle_cost_coeff),
            collision_cost=float(collision_cost),
            obstacle_smooth_eps=float(obstacle_smooth_eps),
            use_obstacle_constraints=bool(use_obstacle_constraints),
        )

    def reset(self) -> None:
        self.mpc.reset()

    def get_action(self, state: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=float).reshape(-1)
        target_pos = np.asarray(target_pos, dtype=float).reshape(2)
        u_phys, _ = self.mpc.solve_mpc(state, target_pos)
        u_phys = np.asarray(u_phys, dtype=float).reshape(2)

        # Environment expects normalized actions in [-1,1].
        u_norm = u_phys / float(self.env.acceleration_bound)
        return np.clip(u_norm, -1.0, 1.0)


def test_mpc_ball():
    target_pos = [3.0, 3.0]
    env = BallEnvironment(target_pos=target_pos, max_steps=1000)

    mpc = BallMPC(env=env, horizon=5, dt=env.dt)

    initial_state = np.array([0.0, 0.0, 0.0, 0.0])
    state = env.reset(initial_state=initial_state)

    trajectory = [state.copy()]
    control_history = []
    predicted_trajectories = []

    max_steps = 500
    target_threshold = 0.2

    for step in range(max_steps):
        control_action, predicted_traj = mpc.solve_mpc(state, target_pos)

        control_history.append(control_action)
        if predicted_traj is not None:
            predicted_trajectories.append(predicted_traj)

        scaled_action = control_action / env.acceleration_bound
        scaled_action = np.clip(scaled_action, -1.0, 1.0)

        next_state, reward, done, info = env.step(scaled_action)
        trajectory.append(next_state.copy())
        state = next_state.copy()

        distance = np.linalg.norm(state[:2] - target_pos)
        if distance < target_threshold:
            break

    trajectory = np.array(trajectory)
    control_history = np.array(control_history)

    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        ax1.plot(trajectory[:, 0], trajectory[:, 1], "b-", linewidth=2, label="Actual trajectory")
        ax1.plot(target_pos[0], target_pos[1], "go", markersize=12, label="Target")
        ax1.plot(initial_state[0], initial_state[1], "ro", markersize=8, label="Initial position")

        if predicted_trajectories:
            num_pred_to_plot = min(5, len(predicted_trajectories))
            indices = np.linspace(0, len(predicted_trajectories) - 1, num_pred_to_plot, dtype=int)
            for i in indices:
                pred_traj = predicted_trajectories[i]
                ax1.plot(pred_traj[:, 0], pred_traj[:, 1], "g--", alpha=0.3)

        ax1.set_xlabel("X Position")
        ax1.set_ylabel("Y Position")
        ax1.set_title("Ball Trajectory (MPC Control)")
        ax1.legend()
        ax1.grid(True)
        ax1.axis("equal")
        ax1.set_xlim(-6, 6)
        ax1.set_ylim(-6, 6)

        ax2.plot(control_history[:, 0], "r-", label="Control X (ax)")
        ax2.plot(control_history[:, 1], "b-", label="Control Y (ay)")
        ax2.axhline(0, color="k", linestyle="--", alpha=0.3)
        ax2.axhline(env.acceleration_bound, color="gray", linestyle="--", alpha=0.3)
        ax2.axhline(-env.acceleration_bound, color="gray", linestyle="--", alpha=0.3)
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Control Input")
        ax2.set_title("Control Inputs Over Time")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()

        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        results_dir = os.path.join(root_dir, "experiments", "results")
        os.makedirs(results_dir, exist_ok=True)
        plot_path = os.path.join(results_dir, "mpc_ball_trajectory.png")
        plt.savefig(plot_path)
        plt.close()

    except Exception:
        pass

    return trajectory


if __name__ == "__main__":
    test_mpc_ball()
