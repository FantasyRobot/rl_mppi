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


class CDFCollisionRecoveryPolicy:
    """Collision recovery policy using C-space distance field (CDF).

    Intended for 2-DoF Robot2D (q = [q1, q2]). When the robot is in collision,
        this policy produces an action that moves in configuration space using the
        CDF gradient.

        Important: CDF2D.calculate_cdf returns an *unsigned* distance-to-contact-set
        in configuration space. That means it does NOT indicate inside/outside by
        sign. Therefore, a projection step like q <- q - g*d moves *toward* the
        contact manifold and can look like “sticking” instead of escaping.

        Here we instead:
            - push outward along +grad to increase CDF distance
            - add a tangent component (rotate(grad, 90deg)) to slide around obstacles
            - choose the tangent sign that best improves goal progress

    Uses CDF2D.calculate_cdf(..., method='offline_grid') to avoid needing
    torchmin / online optimization.
    """

    def __init__(
        self,
        *,
        method: str = "offline_grid",
        proj_gain: float = 1.0,
        tangent_step: float = 0.06,
        target_cdf: float = 0.8,
        level_band: float = 0.10,
        level_k: float = 3.0,
        min_clearance: float = 0.08,
        clearance_fd_eps: float = 2e-3,
        alpha_obstacle: float = 1.0,
        alpha_goal: float = 0.35,
        obstacle_activation_distance: float | None = None,
        safety_lookahead_steps: int = 4,
        safety_min_predicted_clearance: float = 0.0,
    ):
        self.method = str(method)
        # proj_gain/tangent_step are kept for backward compatibility with existing CLIs.
        # They act like speeds (rad/s) for normal and tangential motion.
        self.proj_gain = float(proj_gain)
        self.tangent_step = float(tangent_step)
        self.target_cdf = float(target_cdf)
        self.level_band = float(level_band)
        self.level_k = float(level_k)
        self.min_clearance = float(min_clearance)
        self.clearance_fd_eps = float(clearance_fd_eps)
        self.alpha_obstacle = float(alpha_obstacle)
        self.alpha_goal = float(alpha_goal)
        self.obstacle_activation_distance = obstacle_activation_distance
        self.safety_lookahead_steps = int(max(1, safety_lookahead_steps))
        self.safety_min_predicted_clearance = float(safety_min_predicted_clearance)
        self._cdf = None
        self._device = None

        # Debug/telemetry fields (useful for plotting scripts)
        self.last_d: float | None = None
        self.last_clearance: float | None = None
        self.last_mode: str | None = None

    def _lazy_init(self):
        if self._cdf is not None:
            return
        import torch

        from algorithms.cdf_rl_mppi.cdf_2d.cdf import CDF2D

        self._device = torch.device("cpu")
        self._cdf = CDF2D(self._device)

    def __call__(self, s: np.ndarray, env: Robot2DEnvironmentObstacles) -> np.ndarray:
        if int(env.n) != 2:
            raise ValueError("CDFCollisionRecoveryPolicy currently supports only 2-DoF Robot2D (n=2)")
        if not env.obstacles:
            return np.zeros((env.action_dim,), dtype=np.float32)

        self._lazy_init()

        import torch
        import torch.nn.functional as F

        from algorithms.cdf_rl_mppi.cdf_2d.primitives2D_torch import Circle

        s = np.asarray(s, dtype=np.float32).reshape(2 * int(env.n))
        q_np = np.asarray(s[:2], dtype=np.float32).reshape(2)
        qd_np = np.asarray(s[2:4], dtype=np.float32).reshape(2)

        q_t = torch.tensor(q_np.reshape(1, 2), dtype=torch.float32, device=self._device, requires_grad=True)

        extra_inflate = float(getattr(env, "cdf_obstacle_inflate", 0.0) or 0.0)
        inflate = float(env.safety_distance) + extra_inflate
        obj_lists = [
            Circle(
                center=torch.tensor([float(o.x), float(o.y)], dtype=torch.float32, device=self._device),
                radius=float(o.r) + inflate,
                device=self._device,
            )
            for o in env.obstacles
        ]

        d, g = self._cdf.calculate_cdf(q_t, obj_lists, method=self.method, return_grad=True)
        d_cur = float(d.detach().cpu().numpy().reshape(()))

        # g: (1,2) -> normalized
        g = F.normalize(g, dim=-1)
        g_orth = torch.stack([g[:, 1], -g[:, 0]], dim=-1)
        g_np = g.detach().cpu().numpy().astype(np.float32).reshape(2)
        g_orth_np = g_orth.detach().cpu().numpy().astype(np.float32).reshape(2)

        mc = env.min_obstacle_clearance(q_np)
        clearance = float(mc[0] if isinstance(mc, tuple) else mc)

        # Compute a "safety normal" direction from the *environment* clearance,
        # so we push away from collision according to the same geometry used by
        # the simulator (segment-to-circle distance), not only the sampled CDF robot.
        eps = float(max(1e-6, self.clearance_fd_eps))
        grad_clear = np.zeros((2,), dtype=np.float32)
        for i in range(2):
            dq = np.zeros((2,), dtype=np.float32)
            dq[i] = np.float32(eps)
            mc_p = env.min_obstacle_clearance((q_np + dq).astype(np.float32))
            mc_m = env.min_obstacle_clearance((q_np - dq).astype(np.float32))
            c_p = float(mc_p[0] if isinstance(mc_p, tuple) else mc_p)
            c_m = float(mc_m[0] if isinstance(mc_m, tuple) else mc_m)
            grad_clear[i] = np.float32((c_p - c_m) / (2.0 * eps))
        norm_gc = float(np.linalg.norm(grad_clear))
        if np.isfinite(norm_gc) and norm_gc > 1e-6:
            n_safe = (grad_clear / np.float32(norm_gc)).astype(np.float32)
        else:
            # Fallback to CDF gradient if finite-difference is degenerate.
            n_safe = g_np.copy()

        # Goal direction in configuration space (Jacobian-transpose task-space direction).
        eef = env.forward_kinematics_eef(q_np)
        pos_err = (np.asarray(env.target_pos, dtype=np.float32) - np.asarray(eef, dtype=np.float32)).astype(np.float32)
        J = env.jacobian_eef(q_np).astype(np.float32)
        q_goal_dir = (J.T @ pos_err).astype(np.float32)
        norm_qg = float(np.linalg.norm(q_goal_dir))
        if np.isfinite(norm_qg) and norm_qg > 1e-6:
            q_goal_hat = (q_goal_dir / np.float32(norm_qg)).astype(np.float32)
        else:
            q_goal_hat = np.zeros((2,), dtype=np.float32)

        # --- Paper-inspired switching + angle-based motion ---
        # We keep the "target_cdf" notion to decide when we're safely away from contact,
        # but the actual motion direction is selected using an angle-based tradeoff between:
        #   - obstacle normal (here: n_safe which increases clearance)
        #   - goal direction (q_goal_hat)
        # This is more robust than pure tangent following when the CDF model is approximate.
        target = float(self.target_cdf)
        band = max(1e-6, float(self.level_band))

        # Scale speeds; also scale normal speed up when deeply in collision.
        base_normal_speed = float(max(0.0, self.proj_gain))
        base_tang_speed = float(max(0.0, self.tangent_step))
        penetration = float(max(0.0, -clearance))
        normal_speed = base_normal_speed * (1.0 + 6.0 * penetration)
        tang_speed = base_tang_speed

        # Clamp to something that makes sense w.r.t. env velocity limits.
        normal_speed = float(np.clip(normal_speed, 0.0, 0.85 * float(env.qd_max)))
        tang_speed = float(np.clip(tang_speed, 0.0, 0.85 * float(env.qd_max)))

        d_act = float(env.obstacle_margin if self.obstacle_activation_distance is None else self.obstacle_activation_distance)
        goal_dist_task = float(np.linalg.norm(pos_err))

        # Obstacle avoidance is only active when close enough to matter.
        obstacle_active = (clearance < d_act) and (clearance < goal_dist_task)

        # Force escape if we are colliding OR too close according to env clearance.
        escape = (clearance < float(self.min_clearance)) or (d_cur < target - band)

        self.last_d = float(d_cur)
        self.last_clearance = float(clearance)
        self.last_mode = "escape" if bool(escape) else "slide"

        # Normal correction term (acts like a 1D regulator on cdf value).
        level_err = float(d_cur - target)
        normal_corr = -float(self.level_k) * level_err
        normal_corr = float(np.clip(normal_corr, -normal_speed, normal_speed))

        if escape:
            # While colliding, do not add tangential motion (it tends to scrape along the obstacle).
            qdot_des = (normal_speed * n_safe).astype(np.float32)
        else:
            def _angle(u: np.ndarray, v: np.ndarray) -> float:
                nu = float(np.linalg.norm(u))
                nv = float(np.linalg.norm(v))
                if (not np.isfinite(nu)) or (not np.isfinite(nv)) or nu < 1e-8 or nv < 1e-8:
                    return float(np.pi / 2.0)
                c = float(np.dot(u, v) / (nu * nv))
                c = float(np.clip(c, -1.0, 1.0))
                return float(np.arccos(c))

            # Candidates: goal, safe normal, and combinations (plus tangent to enable sliding).
            cand_dirs: list[np.ndarray] = []
            if np.linalg.norm(q_goal_hat) > 1e-6:
                cand_dirs.append(q_goal_hat)
            cand_dirs.append(n_safe)
            cand_dirs.append((q_goal_hat + 0.8 * n_safe).astype(np.float32))
            cand_dirs.append((q_goal_hat + 0.6 * g_orth_np).astype(np.float32))
            cand_dirs.append((q_goal_hat + 0.6 * g_orth_np + 0.4 * n_safe).astype(np.float32))

            best_v = None
            best_cost = None
            for v in cand_dirs:
                nv = float(np.linalg.norm(v))
                if (not np.isfinite(nv)) or nv < 1e-8:
                    continue
                v_hat = (v / np.float32(nv)).astype(np.float32)

                # If obstacle avoidance is inactive, only care about goal alignment.
                a1 = float(self.alpha_obstacle) if bool(obstacle_active) else 0.0
                a2 = float(self.alpha_goal)

                theta1 = _angle(v_hat, n_safe)  # smaller is more "away from obstacle"
                theta2 = _angle(v_hat, q_goal_hat)  # smaller is more "toward goal"
                cost = a1 * theta1 + a2 * theta2

                # Keep a small normal correction to stay away from the contact set.
                # If we are close, bias slightly toward n_safe.
                if clearance < (d_act * 0.75):
                    cost -= 0.05 * float(np.dot(v_hat, n_safe))

                if (best_cost is None) or (cost < best_cost):
                    best_cost = cost
                    best_v = v_hat

            if best_v is None:
                best_v = n_safe

            # Convert chosen direction into desired velocity.
            speed = tang_speed if bool(obstacle_active) else max(0.5 * tang_speed, 0.25)
            qdot_des = (speed * best_v + normal_corr * n_safe).astype(np.float32)

        # --- Actuation mapping ---
        # Using qdot tracking can be too weak when qdd is saturated and qd already has inertia.
        # For collision recovery we want a more direct, strongly damped acceleration command.
        qdd_max = float(env.qdd_max) if float(env.qdd_max) != 0.0 else 1.0

        dt = float(env.dt) if float(env.dt) != 0.0 else 1.0

        def _rollout_action(a_norm: np.ndarray, *, steps: int) -> tuple[float, float, float, bool]:
            """Roll out env dynamics for a few steps with constant action.

            Returns: (min_clearance_over_rollout, end_clearance, end_goal_dist, any_collision)
            """

            a_norm = np.asarray(a_norm, dtype=np.float32).reshape(2)
            a_norm = np.clip(a_norm, -1.0, 1.0)

            q = q_np.copy()
            qd = qd_np.copy()
            min_clear = float("inf")
            any_col = False

            for _ in range(int(max(1, steps))):
                qdd = np.clip(a_norm * np.float32(qdd_max), -np.float32(qdd_max), np.float32(qdd_max)).astype(np.float32)
                qd = np.clip(qd + qdd * np.float32(dt), -np.float32(env.qd_max), np.float32(env.qd_max)).astype(np.float32)
                q = q + qd * np.float32(dt) + 0.5 * qdd * np.float32(dt * dt)
                q = (q + np.pi) % (2.0 * np.pi) - np.pi

                mc2 = env.min_obstacle_clearance(q)
                clear2 = float(mc2[0] if isinstance(mc2, tuple) else mc2)
                col2 = bool(mc2[1]) if isinstance(mc2, tuple) and len(mc2) > 1 else (clear2 < 0.0)
                min_clear = float(min(min_clear, clear2))
                any_col = bool(any_col or col2)

            eef2 = env.forward_kinematics_eef(q)
            gd2 = float(np.linalg.norm(np.asarray(env.target_pos, dtype=np.float32) - np.asarray(eef2, dtype=np.float32)))
            end_clear = float(clear2)
            return float(min_clear), float(end_clear), float(gd2), bool(any_col)

        def _choose_safe_action(cand_actions: list[np.ndarray]) -> np.ndarray:
            """Pick an action that maximizes predicted clearance, with a safety backoff.

            Preference order:
              1) maximize rollout min-clearance
              2) then maximize end clearance
              3) then minimize end goal distance

            Additionally tries scaled-down actions to avoid penetration when the
            system has momentum toward obstacles.
            """

            look_steps = int(self.safety_lookahead_steps)
            min_ok = float(self.safety_min_predicted_clearance)

            scales = (1.0, 0.7, 0.5, 0.35, 0.25, 0.15, 0.10)
            best = None
            best_tuple = None

            # First pass: only accept rollouts that stay above min_ok.
            for a in cand_actions:
                a = np.asarray(a, dtype=np.float32).reshape(2)
                for sc in scales:
                    aa = (np.float32(sc) * a).astype(np.float32)
                    minc, endc, gd2, col2 = _rollout_action(aa, steps=look_steps)
                    if bool(col2) or (minc < min_ok):
                        continue
                    tup = (minc, endc, -gd2)
                    if (best_tuple is None) or (tup > best_tuple):
                        best_tuple = tup
                        best = aa

            if best is not None:
                return np.clip(best, -1.0, 1.0)

            # Fallback: pick the least-bad action (max min-clearance), even if it still collides.
            for a in cand_actions:
                a = np.asarray(a, dtype=np.float32).reshape(2)
                minc, endc, gd2, _ = _rollout_action(a, steps=look_steps)
                tup = (minc, endc, -gd2)
                if (best_tuple is None) or (tup > best_tuple):
                    best_tuple = tup
                    best = a
            if best is None:
                best = np.zeros((2,), dtype=np.float32)
            return np.clip(best, -1.0, 1.0)

        # Damping term to quickly kill velocity that would keep scraping the obstacle.
        kd_vel = float(np.clip(10.0 + 22.0 * max(0.0, -clearance), 10.0, 45.0))
        a_damp = (-kd_vel * qd_np / np.float32(qdd_max)).astype(np.float32)

        # Emergency stop (attempt to cancel velocity in ~1 step).
        a_stop = (-(qd_np / np.float32(max(1e-6, dt))) / np.float32(qdd_max)).astype(np.float32)

        if escape:
            # Choose an action that maximizes *predicted* clearance next step.
            # Candidate set: push outward, push +/- tangent, push + goal.
            push = float(np.clip(self.proj_gain, 0.6, 3.0))
            tang = float(np.clip(self.tangent_step, 0.0, 2.0))

            cand_actions: list[np.ndarray] = []
            cand_actions.append((push * n_safe + a_damp).astype(np.float32))
            cand_actions.append((push * n_safe + 0.65 * a_stop + a_damp).astype(np.float32))
            cand_actions.append((0.85 * a_stop + a_damp).astype(np.float32))
            cand_actions.append((push * n_safe + 0.6 * tang * g_orth_np + a_damp).astype(np.float32))
            cand_actions.append((push * n_safe - 0.6 * tang * g_orth_np + a_damp).astype(np.float32))
            if np.linalg.norm(q_goal_hat) > 1e-6:
                cand_actions.append((0.65 * push * n_safe + 0.55 * tang * q_goal_hat + a_damp).astype(np.float32))

            best_a = _choose_safe_action(cand_actions)

            # If everything looks bad, bias harder outward + stop.
            minc, _, _, col2 = _rollout_action(best_a, steps=int(self.safety_lookahead_steps))
            if bool(col2) or (float(minc) < float(self.safety_min_predicted_clearance)):
                best_a = (max(1.6, push) * n_safe + 0.95 * a_stop + a_damp).astype(np.float32)
                best_a = np.clip(best_a, -1.0, 1.0)

            return np.clip(best_a, -1.0, 1.0)

        # Non-escape mode: accelerate in chosen safe direction with damping.
        v = qdot_des.astype(np.float32)
        nv = float(np.linalg.norm(v))
        v_hat = (v / np.float32(nv)).astype(np.float32) if (np.isfinite(nv) and nv > 1e-6) else n_safe

        tang = float(np.clip(self.tangent_step, 0.0, 3.0))
        a_nom = (tang * v_hat + a_damp).astype(np.float32)
        # Safety filter even in slide mode: avoid choosing an action that predicts penetration.
        a = _choose_safe_action([a_nom, (0.75 * a_nom + 0.35 * a_stop).astype(np.float32), (1.0 * n_safe + a_damp).astype(np.float32)])
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
