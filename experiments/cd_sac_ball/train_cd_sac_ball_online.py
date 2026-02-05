#!/usr/bin/env python3

"""Online interaction training for BallEnvironmentConstraints using SAC (TD-CD only).

This implementation follows the TD-CD-MPPI Eq.(6-9) stochastic-termination idea:
- Compute a soft termination signal delta_t in [0,1] from normalized constraint violation.
- Use a time-varying discount gamma_t = gamma * (1 - delta_t) in the TD target.

Environment enforces hard per-component bounds:
- |ax|,|ay| <= acc_bound
- |vx|,|vy| <= vel_bound
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass

import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from env.envball_constraints import BallEnvironmentConstraints
from algorithms.sac.sac_utils import SACAgent, ReplayBuffer


@dataclass
class EvalStats:
    avg_reward: float
    avg_final_dist: float
    violation_rate: float
    success_rate: float


def _derive_ckpt_path(path: str, suffix: str) -> str:
    root, ext = os.path.splitext(str(path))
    if ext.strip() == "":
        ext = ".pth"
    return f"{root}{suffix}{ext}"


def _is_better(a: EvalStats, b: EvalStats) -> bool:
    """Return True if eval stats a is better than b."""
    if float(a.success_rate) != float(b.success_rate):
        return float(a.success_rate) > float(b.success_rate)
    if float(a.avg_final_dist) != float(b.avg_final_dist):
        return float(a.avg_final_dist) < float(b.avg_final_dist)
    return float(a.avg_reward) > float(b.avg_reward)


def train_cd_sac_ball_online(
    *,
    target_pos: list[float],
    save_path: str,
    total_steps: int = 200_000,
    start_steps: int = 5_000,
    update_after: int = 1_000,
    update_every: int = 1,
    updates_per_step: int = 1,
    batch_size: int = 256,
    max_ep_steps: int = 2000,
    replay_size: int = 300_000,
    seed: int = 42,
    eval_every: int = 20_000,
    reset_span: float = 5.0,
    reach_threshold: float = 0.5,
    auto_entropy_tuning: bool = True,
    alpha: float = 0.1,
    normalize_state: bool = False,
    curriculum_reset_span_start: float | None = None,
    curriculum_reset_span_end: float | None = None,
    vel_bound: float = 2.0,
    acc_bound: float = 0.5,
    acc_limit: float = 1.0,
    constraint_discount_use_amount: bool = False,
    tdcd_p_max: float = 1.0,
    tdcd_tau_c: float = 0.99,
) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    env = BallEnvironmentConstraints(
        target_pos=target_pos,
        max_steps=int(max_ep_steps),
        reward_scale=100.0,
        reset_span=float(reset_span),
        reach_threshold=float(reach_threshold),
        vel_bound=float(vel_bound),
        acceleration_bound=float(acc_limit),
        acc_bound=float(acc_bound),
    )

    def _current_reset_span(step: int) -> float:
        if curriculum_reset_span_start is None or curriculum_reset_span_end is None:
            return float(env.reset_span)
        frac = float(np.clip(step / max(1, total_steps), 0.0, 1.0))
        return float(curriculum_reset_span_start + frac * (curriculum_reset_span_end - curriculum_reset_span_start))

    def _norm_state(s: np.ndarray) -> np.ndarray:
        if not normalize_state:
            return np.asarray(s, dtype=np.float32)
        out = np.asarray(s, dtype=np.float32).copy()
        out[0] = (out[0] - float(env.target_pos[0])) / float(env.pos_bound)
        out[1] = (out[1] - float(env.target_pos[1])) / float(env.pos_bound)
        out[2] = out[2] / float(env.vel_bound)
        out[3] = out[3] / float(env.vel_bound)
        return out

    agent = SACAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_dim=256,
        learning_rate=3e-4,
        alpha=float(alpha),
        gamma=0.99,
        tau=0.005,
        auto_entropy_tuning=bool(auto_entropy_tuning),
        use_lr_scheduler=False,
    )

    replay = ReplayBuffer(max_size=int(replay_size))

    save_path = os.path.expanduser(os.path.expandvars(str(save_path)))
    best_path = _derive_ckpt_path(save_path, "_best")
    last_path = _derive_ckpt_path(save_path, "_last")

    state = env.reset(reset_span=_current_reset_span(0))
    ep_reward = 0.0
    ep_steps = 0
    episode = 0

    def save_model(path: str) -> None:
        model_dir = os.path.dirname(path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        torch.save(
            {
                "policy_state_dict": agent.policy_net.state_dict(),
                "q1_state_dict": agent.q_net1.state_dict(),
                "q2_state_dict": agent.q_net2.state_dict(),
                "target_q1_state_dict": agent.target_q_net1.state_dict(),
                "target_q2_state_dict": agent.target_q_net2.state_dict(),
                "alpha": agent.alpha,
                "auto_entropy_tuning": bool(agent.auto_entropy_tuning),
                "log_alpha": (agent.log_alpha.detach().cpu() if getattr(agent, "log_alpha", None) is not None else None),
                "target_entropy": (float(getattr(agent, "target_entropy", 0.0)) if bool(agent.auto_entropy_tuning) else None),
                "state_norm": ("fixed_bounds_relative" if bool(normalize_state) else None),
                "target_pos": [float(env.target_pos[0]), float(env.target_pos[1])],
                "reset_span": float(env.reset_span),
                "reach_threshold": float(env.reach_threshold),
                "vel_bound": float(env.vel_bound),
                "acc_limit": float(env.acceleration_bound),
                "acc_bound": float(env.acc_bound),
                "constraint_discount_use_amount": bool(constraint_discount_use_amount),
                "tdcd_p_max": float(tdcd_p_max),
                "tdcd_tau_c": float(tdcd_tau_c),
            },
            path,
        )

    def rollout_eval(n_episodes: int = 5) -> EvalStats:
        rews: list[float] = []
        dists: list[float] = []
        violations = 0
        successes = 0
        for _ in range(int(n_episodes)):
            s = env.reset(reset_span=_current_reset_span(total_steps))
            total_r = 0.0
            while True:
                a = agent.select_action(_norm_state(s), evaluate=True)
                s2, r, done, info = env.step(a)
                total_r += float(r)
                s = s2
                if done:
                    rews.append(total_r)
                    dists.append(float(info.get("distance", np.linalg.norm(s[:2] - env.target_pos))))
                    if bool(info.get("constraint_violation", False)):
                        violations += 1
                    if float(info.get("distance", 1e9)) < float(env.reach_threshold):
                        successes += 1
                    break
        return EvalStats(
            avg_reward=float(np.mean(rews)) if rews else 0.0,
            avg_final_dist=float(np.mean(dists)) if dists else 0.0,
            violation_rate=float(violations / max(1, n_episodes)),
            success_rate=float(successes / max(1, n_episodes)),
        )

    # TD-CD normalization state (Eq.8 style, approximated online):
    # - c_max_seen: max |c| observed since last evaluation window
    # - c_max_ema: exponential moving average of c_max_seen
    c_max_seen = 0.0
    c_max_ema = 1.0

    p_max = float(np.clip(float(tdcd_p_max), 0.0, 1.0))
    tau_c = float(np.clip(float(tdcd_tau_c), 0.0, 1.0))

    best_stats: EvalStats | None = None
    best_step: int | None = None

    for t in range(1, int(total_steps) + 1):
        if t <= int(start_steps):
            action = np.random.uniform(-1.0, 1.0, size=env.action_dim)
        else:
            action = agent.select_action(_norm_state(state), evaluate=False)

        next_state, reward, done, info = env.step(action)

        # TD-CD style: use a constraint-dependent discount rather than terminating the backup.
        # - time-limit truncation is not terminal
        # - reached target is terminal
        done_for_buffer = float(done)
        if bool(info.get("time_limit", False)):
            done_for_buffer = 0.0

        # terminal -> discount 0
        if bool(done_for_buffer) and not bool(info.get("time_limit", False)):
            discount_for_buffer = 0.0
        else:
            # Eq.(7): delta_t from normalized constraint signal.
            if bool(constraint_discount_use_amount):
                c = float(info.get("vel_violation_amount", 0.0))
                c_abs = abs(c)
                if c_abs > c_max_seen:
                    c_max_seen = c_abs
                denom = max(1e-6, float(c_max_ema))
                delta = p_max * float(np.clip(c_abs / denom, 0.0, 1.0))
            else:
                vio = 1.0 if bool(info.get("constraint_violation", False)) else 0.0
                delta = p_max * vio

            # Eq.(9) equivalent per-step discount: gamma_t = gamma * (1 - delta_t)
            discount_for_buffer = float(agent.gamma) * (1.0 - float(np.clip(delta, 0.0, 1.0)))

        replay.add(
            _norm_state(state),
            action,
            float(reward),
            _norm_state(next_state),
            done_for_buffer,
            discount_for_buffer,
        )

        ep_reward += float(reward)
        ep_steps += 1
        state = next_state

        if done:
            episode += 1
            final_dist = float(info.get("distance", np.linalg.norm(state[:2] - env.target_pos)))
            vio = bool(info.get("constraint_violation", False))
            if episode % 10 == 0:
                print(
                    f"Episode {episode:5d} | steps {ep_steps:4d} | ep_reward {ep_reward:8.3f} | "
                    f"final_dist {final_dist:6.3f} | violation={int(vio)}"
                )
            state = env.reset(reset_span=_current_reset_span(t))
            ep_reward = 0.0
            ep_steps = 0

        if t >= int(update_after) and len(replay) >= int(batch_size) and (t % int(update_every) == 0):
            for _ in range(int(updates_per_step)):
                agent.update(replay, batch_size=int(batch_size))

        if int(eval_every) > 0 and (t % int(eval_every) == 0):
            # Eq.(8) approximation: update c_max EMA once per evaluation window.
            c_max_ema = float(tau_c * float(c_max_ema) + (1.0 - tau_c) * float(max(c_max_seen, 1e-6)))
            c_max_seen = 0.0
            es = rollout_eval(n_episodes=5)
            print(
                f"[EVAL] step={t} avg_reward={es.avg_reward:.3f} avg_final_dist={es.avg_final_dist:.3f} "
                f"success_rate={es.success_rate*100.0:.1f}% violation_rate={es.violation_rate*100.0:.1f}% alpha={agent.alpha:.4f}"
            )

            # Always save the latest checkpoint.
            save_model(last_path)
            print(f"[EVAL] saved last model to {last_path}")

            # Save the best checkpoint so far (prevents later collapse overwriting good policy).
            if best_stats is None or _is_better(es, best_stats):
                best_stats = es
                best_step = int(t)
                save_model(best_path)
                print(
                    f"[EVAL] new best @ step={t}: success={es.success_rate*100.0:.1f}% "
                    f"final_dist={es.avg_final_dist:.3f} reward={es.avg_reward:.3f}"
                )
                print(f"[EVAL] saved best model to {best_path}")

            # Also keep compatibility: update save_path to always point to current best.
            if os.path.exists(best_path):
                shutil.copyfile(best_path, save_path)
                print(f"[EVAL] updated {save_path} -> best checkpoint")

    # End of training: ensure save_path points to best checkpoint.
    save_model(last_path)
    if os.path.exists(best_path):
        shutil.copyfile(best_path, save_path)
        print(f"Training done. Best model (step={best_step}) saved to {save_path}")
        print(f"  best_path={best_path}")
        print(f"  last_path={last_path}")
    else:
        shutil.copyfile(last_path, save_path)
        print(f"Training done. Model saved to {save_path}")
        print(f"  last_path={last_path}")
