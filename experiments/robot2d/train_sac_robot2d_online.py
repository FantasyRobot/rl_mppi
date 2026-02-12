#!/usr/bin/env python3

"""Online interaction training for Robot2DEnvironmentObstacles using SAC.

Observation used for SAC (default, when --normalize_state is enabled):
    obs = [q/pi, qd/qd_max, target_pos/reach, obs_1, ..., obs_K]
where reach = sum(link_lengths) and each obstacle obs_i = [x/reach, y/reach, r/reach].

If there are fewer than K obstacles, the encoding is zero-padded.
If there are more than K obstacles, extra obstacles are truncated.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from env.envrobot2d_obstacles import CircleObstacle, Robot2DEnvironmentObstacles

from algorithms.sac.sac_utils import SACAgent, ReplayBuffer


def _parse_obstacles(s: str) -> list[CircleObstacle]:
    s = (s or "").strip()
    if not s:
        return []
    out: list[CircleObstacle] = []
    for part in s.split(";"):
        xs, ys, rs = [t.strip() for t in part.split(",")]
        out.append(CircleObstacle(float(xs), float(ys), float(rs)))
    return out


def train_sac_robot2d_online(
    *,
    link_lengths: list[float],
    target_pos: list[float],
    obstacles: list[CircleObstacle],
    save_path: str,
    total_steps: int = 250_000,
    start_steps: int = 10_000,
    update_after: int = 2_000,
    update_every: int = 1,
    updates_per_step: int = 1,
    batch_size: int = 256,
    max_ep_steps: int = 450,
    replay_size: int = 400_000,
    seed: int = 42,
    eval_every: int = 25_000,
    auto_entropy_tuning: bool = True,
    alpha: float = 0.2,
    normalize_state: bool = True,
    include_obstacles_in_obs: bool = True,
    max_obstacles_in_obs: int = 4,
    reset_noise: float = 0.20,
    reach_threshold: float = 0.15,
    qd_max: float = 4.0,
    qdd_max: float = 8.0,
    # Obstacle shaping strategy (per-paper): SAC uses CDF shaping by default.
    obstacle_shaping: str = "cdf",
    cdf_method: str = "offline_grid",
    cdf_obstacle_inflate: float = 0.0,
    cdf_margin: float = 0.25,
    cdf_penalty: float = 800.0,
    cdf_penalty_power: float = 2.0,
    cdf_bonus_gain: float = 0.0,
    cdf_bonus_cap: float = 2.0,
    collision_penalty: float = 0.0,
) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    env = Robot2DEnvironmentObstacles(
        link_lengths=link_lengths,
        target_pos=target_pos,
        max_steps=int(max_ep_steps),
        obstacles=obstacles,
        obstacle_margin=0.25,
        obstacle_penalty=250.0,
        obstacle_shaping=str(obstacle_shaping),
        cdf_method=str(cdf_method),
        cdf_obstacle_inflate=float(cdf_obstacle_inflate),
        cdf_margin=float(cdf_margin),
        cdf_penalty=float(cdf_penalty),
        cdf_penalty_power=float(cdf_penalty_power),
        cdf_bonus_gain=float(cdf_bonus_gain),
        cdf_bonus_cap=float(cdf_bonus_cap),
        collision_penalty=float(collision_penalty),
        # SAC training: do NOT modify dynamics/velocity based on CDF/clearance.
        # Keep avoidance purely as reward shaping so the policy learns it.
        contour_avoidance=False,
        contour_mode="clearance",
        terminate_on_collision=True,
        collision_check="arm",
        reset_noise=float(reset_noise),
        reach_threshold=float(reach_threshold),
        qd_max=float(qd_max),
        qdd_max=float(qdd_max),
    )

    reach = float(np.sum(np.asarray(env.link_lengths, dtype=np.float32)))
    max_obstacles_in_obs = int(max(0, max_obstacles_in_obs))

    def encode_obstacles() -> np.ndarray:
        if (not include_obstacles_in_obs) or max_obstacles_in_obs <= 0:
            return np.zeros((0,), dtype=np.float32)
        denom_reach = reach if reach != 0.0 else 1.0
        # Deterministic ordering.
        obs_sorted = sorted(env.obstacles, key=lambda o: (float(o.x), float(o.y), float(o.r)))
        vec = np.zeros((3 * max_obstacles_in_obs,), dtype=np.float32)
        for i, o in enumerate(obs_sorted[:max_obstacles_in_obs]):
            vec[3 * i + 0] = np.float32(float(o.x) / denom_reach)
            vec[3 * i + 1] = np.float32(float(o.y) / denom_reach)
            vec[3 * i + 2] = np.float32(float(o.r) / denom_reach)
        return vec

    def make_obs(state: np.ndarray) -> np.ndarray:
        n = env.n
        s = np.asarray(state, dtype=np.float32).reshape(2 * n)
        q = s[:n]
        qd = s[n:]
        if not normalize_state:
            base = np.concatenate([q, qd, np.asarray(env.target_pos, dtype=np.float32)], axis=0).astype(np.float32)
            if include_obstacles_in_obs and max_obstacles_in_obs > 0:
                # Even in raw mode, keep obstacle encoding scaled by reach for stability.
                return np.concatenate([base, encode_obstacles()], axis=0).astype(np.float32)
            return base

        qn = q / np.float32(np.pi)
        denom_qd = float(env.qd_max) if float(env.qd_max) != 0.0 else 1.0
        qdn = qd / np.float32(denom_qd)
        denom_reach = reach if reach != 0.0 else 1.0
        tn = np.asarray(env.target_pos, dtype=np.float32) / np.float32(denom_reach)
        base = np.concatenate([qn, qdn, tn], axis=0).astype(np.float32)
        if include_obstacles_in_obs and max_obstacles_in_obs > 0:
            return np.concatenate([base, encode_obstacles()], axis=0).astype(np.float32)
        return base

    obs_dim = int(2 * env.n + 2 + (3 * max_obstacles_in_obs if include_obstacles_in_obs else 0))

    agent = SACAgent(
        state_dim=obs_dim,
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
                "state_norm": ("robot2d_q_pi_qd_max_target_reach" if bool(normalize_state) else None),
                "obs_config": {
                    "include_obstacles_in_obs": bool(include_obstacles_in_obs),
                    "max_obstacles_in_obs": int(max_obstacles_in_obs),
                    "obstacle_encoding": "circle_xy_r_over_reach",
                    "obstacle_sort": "x_y_r",
                },
                "obs_dim": int(obs_dim),
                "action_dim": int(env.action_dim),
                "link_lengths": [float(x) for x in env.link_lengths.tolist()],
                "dt": float(env.dt),
                "qd_max": float(env.qd_max),
                "qdd_max": float(env.qdd_max),
                "target_pos": [float(env.target_pos[0]), float(env.target_pos[1])],
                "reach_threshold": float(env.reach_threshold),
                "reset_noise": float(env.reset_noise),
                "obstacles": [[float(o.x), float(o.y), float(o.r)] for o in env.obstacles],
                "obstacle_margin": float(env.obstacle_margin),
                "obstacle_penalty": float(env.obstacle_penalty),
                "terminate_on_collision": bool(env.terminate_on_collision),
                "collision_check": str(env.collision_check),
                "safety_distance": float(env.safety_distance),
                "seed": int(seed),
            },
            path,
        )

    def rollout_eval(n_episodes: int = 5) -> tuple[float, float, float, float]:
        dists: list[float] = []
        rews: list[float] = []
        collisions = 0
        successes = 0
        for _ in range(int(n_episodes)):
            s = env.reset()
            total_r = 0.0
            hit_any = False
            while True:
                a = agent.select_action(make_obs(s), evaluate=True)
                s2, r, done, info = env.step(a)
                total_r += float(r)
                hit_any = hit_any or bool(info.get("hit_obstacle", False))
                s = s2
                if done:
                    eef = env.forward_kinematics_eef(s[: env.n])
                    dists.append(float(np.linalg.norm(eef - env.target_pos)))
                    rews.append(float(total_r))
                    if float(dists[-1]) < float(env.reach_threshold):
                        successes += 1
                    if hit_any:
                        collisions += 1
                    break

        hit_rate = collisions / max(1, int(n_episodes))
        success_rate = successes / max(1, int(n_episodes))
        return float(np.mean(rews)), float(np.mean(dists)), float(hit_rate), float(success_rate)

    state = env.reset()
    ep_reward = 0.0
    ep_steps = 0
    episode = 0

    for t in range(1, int(total_steps) + 1):
        if t <= int(start_steps):
            action = np.random.uniform(-1.0, 1.0, size=env.action_dim)
        else:
            action = agent.select_action(make_obs(state), evaluate=False)

        next_state, reward, done, info = env.step(action)

        done_for_buffer = float(done)
        if bool(info.get("time_limit", False)):
            done_for_buffer = 0.0

        replay.add(make_obs(state), action, float(reward), make_obs(next_state), done_for_buffer)

        ep_reward += float(reward)
        ep_steps += 1
        state = next_state

        if done:
            episode += 1
            eef = env.forward_kinematics_eef(state[: env.n])
            final_dist = float(np.linalg.norm(eef - env.target_pos))
            if episode % 10 == 0:
                print(
                    f"Episode {episode:5d} | steps {ep_steps:4d} | ep_reward {ep_reward:8.3f} | final_dist {final_dist:6.3f}"
                )
            state = env.reset()
            ep_reward = 0.0
            ep_steps = 0

        if t >= int(update_after) and len(replay) >= int(batch_size) and (t % int(update_every) == 0):
            for _ in range(int(updates_per_step)):
                agent.update(replay, batch_size=int(batch_size))

        if int(eval_every) > 0 and (t % int(eval_every) == 0):
            avg_r, avg_d, hit_rate, success_rate = rollout_eval(n_episodes=5)
            print(
                f"[EVAL] step={t} avg_reward={avg_r:.3f} avg_final_dist={avg_d:.3f} "
                f"success_rate={success_rate*100.0:.1f}% collision_rate={hit_rate*100.0:.1f}% alpha={agent.alpha:.4f}"
            )
            save_model(save_path)
            print(f"[EVAL] saved model to {save_path}")

    save_model(save_path)
    print(f"Training done. Model saved to {save_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Online SAC training for Robot2D")

    p.add_argument("--link_lengths", type=str, default="2.0,2.0", help="Comma-separated link lengths")
    p.add_argument("--target_x", type=float, default=-2.8)
    p.add_argument("--target_y", type=float, default=1.8)
    p.add_argument("--obstacles", type=str, default="0,1.8,0.20", help="Obstacles as 'x,y,r;x,y,r'")

    p.add_argument("--save_path", type=str, default=os.path.join(_ROOT_DIR, "experiments", "robot2d", "models", "sac_robot2d_model_online.pth"))
    p.add_argument("--total_steps", type=int, default=250000)
    p.add_argument("--start_steps", type=int, default=10000)
    p.add_argument("--update_after", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--max_ep_steps", type=int, default=450)
    p.add_argument("--replay_size", type=int, default=400000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_every", type=int, default=25000)

    p.add_argument("--auto_entropy_tuning", action="store_true")
    p.add_argument("--no_auto_entropy_tuning", action="store_true")
    p.add_argument("--alpha", type=float, default=0.2)

    p.add_argument("--normalize_state", type=int, default=1)
    p.add_argument("--include_obstacles_in_obs", type=int, default=1)
    p.add_argument("--max_obstacles_in_obs", type=int, default=4)
    p.add_argument("--reset_noise", type=float, default=0.20)
    p.add_argument("--reach_threshold", type=float, default=0.15)
    p.add_argument("--qd_max", type=float, default=4.0)
    p.add_argument("--qdd_max", type=float, default=8.0)

    # Obstacle shaping for SAC: default to CDF (dense shaping), while keeping clearance for collision detection.
    p.add_argument(
        "--obstacle_shaping",
        type=str,
        default="cdf",
        choices=["cdf", "clearance", "both", "none"],
        help="Soft obstacle penalty source for reward shaping (collision check always uses clearance).",
    )
    p.add_argument("--cdf_method", type=str, default="offline_grid", help="CDF method for shaping: offline_grid (default) or online_computation")
    p.add_argument("--cdf_obstacle_inflate", type=float, default=0.0, help="Extra obstacle inflation for CDF shaping (added on top of env.safety_distance)")
    p.add_argument("--cdf_margin", type=float, default=0.25, help="Hinge margin for CDF shaping penalty")
    p.add_argument("--cdf_penalty", type=float, default=800.0, help="Penalty coefficient for CDF shaping")
    p.add_argument("--cdf_penalty_power", type=float, default=2.0, help="Penalty power p in penalty = cdf_penalty * max(0, margin-cdf)^p")
    p.add_argument("--cdf_bonus_gain", type=float, default=0.0, help="Optional bounded positive reward: +gain*min(cdf, cap)")
    p.add_argument("--cdf_bonus_cap", type=float, default=2.0, help="Cap for CDF bonus term")
    p.add_argument("--collision_penalty", type=float, default=0.0, help="Optional extra penalty applied on collision (in addition to termination)")

    args = p.parse_args()

    if bool(args.auto_entropy_tuning) and bool(args.no_auto_entropy_tuning):
        raise SystemExit("Choose only one: --auto_entropy_tuning or --no_auto_entropy_tuning")

    auto_entropy = True
    if bool(args.no_auto_entropy_tuning):
        auto_entropy = False
    elif bool(args.auto_entropy_tuning):
        auto_entropy = True

    link_lengths = [float(x.strip()) for x in str(args.link_lengths).split(",") if x.strip()]

    train_sac_robot2d_online(
        link_lengths=link_lengths,
        target_pos=[float(args.target_x), float(args.target_y)],
        obstacles=_parse_obstacles(str(args.obstacles)),
        save_path=str(args.save_path),
        total_steps=int(args.total_steps),
        start_steps=int(args.start_steps),
        update_after=int(args.update_after),
        batch_size=int(args.batch_size),
        max_ep_steps=int(args.max_ep_steps),
        replay_size=int(args.replay_size),
        seed=int(args.seed),
        eval_every=int(args.eval_every),
        auto_entropy_tuning=bool(auto_entropy),
        alpha=float(args.alpha),
        normalize_state=bool(int(args.normalize_state)),
        include_obstacles_in_obs=bool(int(args.include_obstacles_in_obs)),
        max_obstacles_in_obs=int(args.max_obstacles_in_obs),
        reset_noise=float(args.reset_noise),
        reach_threshold=float(args.reach_threshold),
        qd_max=float(args.qd_max),
        qdd_max=float(args.qdd_max),
        obstacle_shaping=str(args.obstacle_shaping),
        cdf_method=str(args.cdf_method),
        cdf_obstacle_inflate=float(args.cdf_obstacle_inflate),
        cdf_margin=float(args.cdf_margin),
        cdf_penalty=float(args.cdf_penalty),
        cdf_penalty_power=float(args.cdf_penalty_power),
        cdf_bonus_gain=float(args.cdf_bonus_gain),
        cdf_bonus_cap=float(args.cdf_bonus_cap),
        collision_penalty=float(args.collision_penalty),
    )


if __name__ == "__main__":
    main()
