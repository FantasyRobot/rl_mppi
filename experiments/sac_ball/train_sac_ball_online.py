#!/usr/bin/env python3

"""Online interaction training for BallEnvironment using SAC.

This script trains by directly interacting with the environment and learning from
collected transitions.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch

# Ensure project root is importable (so we can import env.envball_utils)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from env.envball_utils import BallEnvironment

from algorithms.sac.sac_utils import SACAgent, ReplayBuffer


@dataclass
class TrainStats:
    episode: int
    episode_reward: float
    steps: int
    final_distance: float


def train_sac_ball_online(
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
) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = BallEnvironment(
        target_pos=target_pos,
        max_steps=max_ep_steps,
        reward_scale=100.0,
        reset_span=reset_span,
        reach_threshold=reach_threshold,
    )

    def _current_reset_span(step: int) -> float:
        if curriculum_reset_span_start is None or curriculum_reset_span_end is None:
            return float(env.reset_span)
        # Linear schedule from start -> end over total_steps
        frac = float(np.clip(step / max(1, total_steps), 0.0, 1.0))
        return float(curriculum_reset_span_start + frac * (curriculum_reset_span_end - curriculum_reset_span_start))

    def _norm_state(s: np.ndarray) -> np.ndarray:
        if not normalize_state:
            return s
        out = np.asarray(s, dtype=np.float32).copy()
        # Use relative-to-target coordinates for easier learning
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
        alpha=alpha,
        gamma=0.99,
        tau=0.005,
        auto_entropy_tuning=bool(auto_entropy_tuning),
        use_lr_scheduler=False,
    )

    replay = ReplayBuffer(max_size=replay_size)

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
            },
            path,
        )

    def rollout_eval(n_episodes: int = 5) -> tuple[float, float, float, float]:
        dists = []
        rews = []
        boundary_hits = 0
        successes = 0
        for _ in range(n_episodes):
            s = env.reset(reset_span=_current_reset_span(total_steps))
            total_r = 0.0
            hit_boundary_any = False
            while True:
                a = agent.select_action(_norm_state(s), evaluate=True)
                s2, r, done, info = env.step(a)
                total_r += float(r)
                if bool(info.get("hit_boundary", False)):
                    hit_boundary_any = True
                s = s2
                if done:
                    dists.append(float(info.get("distance", np.linalg.norm(s[:2] - env.target_pos))))
                    rews.append(total_r)
                    if float(info.get("distance", 1e9)) < float(env.reach_threshold):
                        successes += 1
                    if hit_boundary_any:
                        boundary_hits += 1
                    break
        hit_rate = boundary_hits / max(1, n_episodes)
        success_rate = successes / max(1, n_episodes)
        return float(np.mean(rews)), float(np.mean(dists)), float(hit_rate), float(success_rate)

    for t in range(1, total_steps + 1):
        # Exploration: uniform random for first start_steps
        if t <= start_steps:
            action = np.random.uniform(-1.0, 1.0, size=env.action_dim)
        else:
            action = agent.select_action(_norm_state(state), evaluate=False)

        next_state, reward, done, info = env.step(action)

        # Time-limit handling: do not treat max-steps truncation as terminal for critic targets.
        done_for_buffer = float(done)
        if bool(info.get("time_limit", False)):
            done_for_buffer = 0.0

        replay.add(_norm_state(state), action, float(reward), _norm_state(next_state), done_for_buffer)

        ep_reward += float(reward)
        ep_steps += 1

        state = next_state

        # End episode
        if done:
            episode += 1
            final_dist = float(info.get("distance", np.linalg.norm(state[:2] - env.target_pos)))
            if episode % 10 == 0:
                print(
                    f"Episode {episode:5d} | steps {ep_steps:4d} | ep_reward {ep_reward:8.3f} | final_dist {final_dist:6.3f}"
                )
            state = env.reset(reset_span=_current_reset_span(t))
            ep_reward = 0.0
            ep_steps = 0

        # Updates
        if t >= update_after and len(replay) >= batch_size and (t % update_every == 0):
            for _ in range(updates_per_step):
                agent.update(replay, batch_size=batch_size)

        # Periodic evaluation
        if eval_every > 0 and (t % eval_every == 0):
            avg_r, avg_d, hit_rate, success_rate = rollout_eval(n_episodes=5)
            print(
                f"[EVAL] step={t} avg_reward={avg_r:.3f} avg_final_dist={avg_d:.3f} "
                f"success_rate={success_rate*100.0:.1f}% boundary_hit_rate={hit_rate*100.0:.1f}% alpha={agent.alpha:.4f}"
            )
            save_model(save_path)
            print(f"[EVAL] saved model to {save_path}")

    save_model(save_path)
    print(f"Training done. Model saved to {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Online SAC training for ball")
    parser.add_argument("--target_x", type=float, default=3.0)
    parser.add_argument("--target_y", type=float, default=3.0)
    parser.add_argument("--save_path", type=str, default="models/sac_ball_model_online.pth")
    parser.add_argument("--total_steps", type=int, default=200000)
    parser.add_argument("--start_steps", type=int, default=5000)
    parser.add_argument("--update_after", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_ep_steps", type=int, default=2000)
    parser.add_argument("--replay_size", type=int, default=300000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_every", type=int, default=20000)
    parser.add_argument("--reset_span", type=float, default=5.0)
    parser.add_argument("--reach_threshold", type=float, default=0.5)
    parser.add_argument("--auto_entropy_tuning", action="store_true", help="Enable automatic entropy tuning")
    parser.add_argument("--no_auto_entropy_tuning", action="store_true", help="Disable automatic entropy tuning")
    parser.add_argument("--alpha", type=float, default=0.1, help="Fixed alpha when auto-entropy is disabled")
    parser.add_argument("--normalize_state", action="store_true", help="Normalize state by env bounds before feeding networks")
    parser.add_argument("--curriculum_reset_span_start", type=float, default=None, help="If set with *_end, linearly schedule reset_span")
    parser.add_argument("--curriculum_reset_span_end", type=float, default=None, help="If set with *_start, linearly schedule reset_span")
    args = parser.parse_args()

    if args.auto_entropy_tuning and args.no_auto_entropy_tuning:
        raise SystemExit("Choose only one: --auto_entropy_tuning or --no_auto_entropy_tuning")

    auto_entropy = True
    if args.no_auto_entropy_tuning:
        auto_entropy = False
    elif args.auto_entropy_tuning:
        auto_entropy = True

    train_sac_ball_online(
        target_pos=[args.target_x, args.target_y],
        save_path=args.save_path,
        total_steps=args.total_steps,
        start_steps=args.start_steps,
        update_after=args.update_after,
        batch_size=args.batch_size,
        max_ep_steps=args.max_ep_steps,
        replay_size=args.replay_size,
        seed=args.seed,
        eval_every=args.eval_every,
        reset_span=args.reset_span,
        reach_threshold=args.reach_threshold,
        auto_entropy_tuning=auto_entropy,
        alpha=args.alpha,
        normalize_state=bool(args.normalize_state),
        curriculum_reset_span_start=args.curriculum_reset_span_start,
        curriculum_reset_span_end=args.curriculum_reset_span_end,
    )


if __name__ == "__main__":
    main()
