#!/usr/bin/env python3

"""Online constrained SAC training for t12a_14 using TD-CD discounting.

Mirrors experiments/ball2D/cd_sac_ball/train_cd_sac_ball_online.py but for MuJoCo arm.
Constraints:
- per-joint velocity bound |qvel_i| <= vel_bound
- per-joint acceleration bound |qacc_i| <= acc_bound (finite-diff of qvel over dt)

TD-CD idea:
- Compute delta_t in [0,1] from constraint violation (binary or amount)
- Use per-transition discount gamma_t = gamma * (1 - delta_t)
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass

import numpy as np

try:
    from scipy.io import savemat  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    savemat = None

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover
    plt = None

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from algorithms.sac.sac_utils import ReplayBuffer, SACAgent
from env.envmujoco_t12a_14_constraints import T12A14MuJoCoEnvConstraints


_RESULTS_DIR = os.path.join(_ROOT_DIR, "experiments", "results")


def _resolve_out_path(path: str, *, default_dir: str) -> str:
    path = os.path.expanduser(os.path.expandvars(str(path)))
    if not os.path.isabs(path) and os.path.dirname(path) == "":
        path = os.path.join(default_dir, path)
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    return path


def _derive_sidecar_path(path: str, suffix: str, ext: str) -> str:
    root, _ext = os.path.splitext(str(path))
    if ext and not ext.startswith("."):
        ext = f".{ext}"
    return f"{root}{suffix}{ext}"


@dataclass
class EvalStats:
    avg_reward: float
    avg_final_dist: float
    std_final_dist: float
    avg_steps: float
    violation_rate: float
    success_rate: float


def _derive_ckpt_path(path: str, suffix: str) -> str:
    root, ext = os.path.splitext(str(path))
    if ext.strip() == "":
        ext = ".pth"
    return f"{root}{suffix}{ext}"


def _is_better(a: EvalStats, b: EvalStats) -> bool:
    if float(a.success_rate) != float(b.success_rate):
        return float(a.success_rate) > float(b.success_rate)
    if float(a.avg_final_dist) != float(b.avg_final_dist):
        return float(a.avg_final_dist) < float(b.avg_final_dist)
    return float(a.avg_reward) > float(b.avg_reward)


def train_cd_sac_t12a_14_online(
    *,
    xml_path: str,
    goal_site: str = "goal",
    eef_site: str = "end_effector",
    save_path: str,
    total_steps: int = 200_000,
    start_steps: int = 10_000,
    update_after: int = 2_000,
    update_every: int = 1,
    updates_per_step: int = 1,
    batch_size: int = 256,
    max_ep_steps: int = 2500,
    replay_size: int = 400_000,
    seed: int = 42,
    eval_every: int = 20_000,
    auto_entropy_tuning: bool = True,
    alpha: float = 0.2,
    action_repeat: int = 5,
    reset_noise: float = 0.05,
    reach_tol: float = 0.03,
    qvel_scale: float = 5.0,
    reward_dist_coeff: float = 1.0,
    reward_ctrl_coeff: float = 0.01,
    success_bonus: float = 10.0,
    # constraints
    vel_bound: float = 1.0,
    acc_bound: float = 2.0,
    violation_tol: float = 1e-3,
    violation_agg: str = "max",
    constraint_discount_use_amount: bool = False,
    tdcd_p_max: float = 1.0,
    tdcd_tau_c: float = 0.99,
    # logging/plotting
    log_path: str | None = None,
    plot_path: str | None = None,
    mat_path: str | None = None,
    show_plot: bool = False,
) -> None:
    try:
        import torch
    except ModuleNotFoundError as e:
        raise SystemExit(
            "Torch is required for training. Install it in your env, e.g. (CPU):\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
        ) from e

    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    env = T12A14MuJoCoEnvConstraints(
        xml_path=str(xml_path),
        goal_site=str(goal_site),
        eef_site=str(eef_site),
        max_steps=int(max_ep_steps),
        reach_tol=float(reach_tol),
        action_repeat=int(action_repeat),
        qvel_scale=float(qvel_scale),
        reward_dist_coeff=float(reward_dist_coeff),
        reward_ctrl_coeff=float(reward_ctrl_coeff),
        success_bonus=float(success_bonus),
        seed=int(seed),
        reset_noise=float(reset_noise),
        vel_bound=float(vel_bound),
        acc_bound=float(acc_bound),
        violation_tol=float(violation_tol),
        violation_agg=str(violation_agg),
    )

    agent = SACAgent(
        state_dim=int(env.obs_dim),
        action_dim=int(env.action_dim),
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

    # Default outputs: keep alongside save_path unless user overrides.
    if log_path is None:
        log_path = _derive_sidecar_path(save_path, "_train_log", ".npz")
    if plot_path is None:
        plot_path = _derive_sidecar_path(save_path, "_train", ".png")
    if mat_path is None:
        mat_path = _derive_sidecar_path(save_path, "_train_log", ".mat")

    log_path = _resolve_out_path(str(log_path), default_dir=_RESULTS_DIR)
    plot_path = _resolve_out_path(str(plot_path), default_dir=_RESULTS_DIR)
    mat_path = _resolve_out_path(str(mat_path), default_dir=_RESULTS_DIR)

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
                "log_alpha": (
                    agent.log_alpha.detach().cpu() if getattr(agent, "log_alpha", None) is not None else None
                ),
                "target_entropy": (
                    float(getattr(agent, "target_entropy", 0.0)) if bool(agent.auto_entropy_tuning) else None
                ),
                "obs_dim": int(env.obs_dim),
                "action_dim": int(env.action_dim),
                "xml_path": str(xml_path),
                "goal_site": str(goal_site),
                "eef_site": str(eef_site),
                "reach_tol": float(env.reach_tol),
                "action_repeat": int(env.action_repeat),
                "qvel_scale": float(env.qvel_scale),
                "obs_format": "[qpos_norm(nu), qvel_norm(nv), eef_minus_goal(3)]",
                "action_format": "tanh -> [-1,1] mapped to actuator ctrlrange (position targets)",
                "seed": int(seed),
                # constraint config
                "vel_bound": float(env.vel_bound),
                "acc_bound": float(env.acc_bound),
                "violation_tol": float(getattr(env, "violation_tol", 0.0)),
                "violation_agg": str(getattr(env, "violation_agg", "max")),
                "constraint_discount_use_amount": bool(constraint_discount_use_amount),
                "tdcd_p_max": float(tdcd_p_max),
                "tdcd_tau_c": float(tdcd_tau_c),
            },
            path,
        )

    def rollout_eval(n_episodes: int = 5) -> EvalStats:
        rews: list[float] = []
        dists: list[float] = []
        steps: list[int] = []
        violations = 0
        successes = 0
        for _ in range(int(n_episodes)):
            s = env.reset()
            total_r = 0.0
            ep_violated = False
            while True:
                a = agent.select_action(s, evaluate=True)
                s2, r, done, info = env.step(a)
                total_r += float(r)
                s = s2
                if bool(info.get("constraint_violation", False)):
                    ep_violated = True
                if done:
                    rews.append(total_r)
                    dists.append(float(info.get("dist", 0.0)))
                    steps.append(int(info.get("step", 0)))
                    if bool(ep_violated):
                        violations += 1
                    if bool(info.get("success", False)):
                        successes += 1
                    break
        return EvalStats(
            avg_reward=float(np.mean(rews)) if rews else 0.0,
            avg_final_dist=float(np.mean(dists)) if dists else 0.0,
            std_final_dist=float(np.std(dists)) if dists else 0.0,
            avg_steps=float(np.mean(steps)) if steps else 0.0,
            violation_rate=float(violations / max(1, n_episodes)),
            success_rate=float(successes / max(1, n_episodes)),
        )

    # TD-CD normalization state (EMA of max constraint magnitude, ball implementation style)
    c_max_seen = 0.0
    c_max_ema = 1.0

    p_max = float(np.clip(float(tdcd_p_max), 0.0, 1.0))
    tau_c = float(np.clip(float(tdcd_tau_c), 0.0, 1.0))

    best_stats: EvalStats | None = None
    best_step: int | None = None

    # Training logs
    # - episode_* logged at episode end
    # - eval_* logged at each eval window
    episode_end_step: list[int] = []
    episode_return: list[float] = []
    episode_len: list[int] = []
    episode_final_dist: list[float] = []
    episode_success: list[int] = []
    episode_violation: list[int] = []

    eval_step: list[int] = []
    eval_avg_reward: list[float] = []
    eval_avg_final_dist: list[float] = []
    eval_std_final_dist: list[float] = []
    eval_avg_steps: list[float] = []
    eval_success_rate: list[float] = []
    eval_violation_rate: list[float] = []
    eval_alpha: list[float] = []
    eval_c_max_ema: list[float] = []

    s = env.reset()
    ep_reward = 0.0
    ep_steps = 0
    episode = 0
    ep_violated = False

    for t in range(1, int(total_steps) + 1):
        if t <= int(start_steps):
            a = np.random.uniform(-1.0, 1.0, size=(env.action_dim,)).astype(np.float32)
        else:
            a = agent.select_action(s, evaluate=False)

        s2, r, done, info = env.step(a)
        ep_reward += float(r)
        ep_steps += 1

        if bool(info.get("constraint_violation", False)):
            ep_violated = True

        # TD-CD discounting (match cd_sac_ball behavior)
        done_for_buffer = float(done)
        if bool(info.get("time_limit", False)):
            done_for_buffer = 0.0

        if bool(done_for_buffer) and not bool(info.get("time_limit", False)):
            discount_for_buffer = 0.0
        else:
            if bool(constraint_discount_use_amount):
                c = float(info.get("constraint_violation_amount", 0.0))
                c_abs = abs(c)
                if c_abs > c_max_seen:
                    c_max_seen = c_abs
                denom = max(1e-6, float(c_max_ema))
                delta = p_max * float(np.clip(c_abs / denom, 0.0, 1.0))
            else:
                vio = 1.0 if bool(info.get("constraint_violation", False)) else 0.0
                delta = p_max * vio

            discount_for_buffer = float(agent.gamma) * (1.0 - float(np.clip(delta, 0.0, 1.0)))

        replay.add(s, a, float(r), s2, done_for_buffer, discount_for_buffer)
        s = s2

        if done:
            episode += 1
            episode_end_step.append(int(t))
            episode_return.append(float(ep_reward))
            episode_len.append(int(ep_steps))
            episode_final_dist.append(float(info.get("dist", 0.0)))
            episode_success.append(1 if bool(info.get("success", False)) else 0)
            episode_violation.append(1 if bool(ep_violated) else 0)
            if episode % 10 == 0:
                print(
                    f"Episode {episode:5d} | steps {ep_steps:4d} | ep_reward {ep_reward:8.3f} | "
                    f"final_dist {float(info.get('dist', 0.0)):7.4f} | violation={int(bool(info.get('constraint_violation', False)))}"
                )
            s = env.reset()
            ep_reward = 0.0
            ep_steps = 0
            ep_violated = False

        if t >= int(update_after) and len(replay) >= int(batch_size) and (t % int(update_every) == 0):
            for _ in range(int(updates_per_step)):
                agent.update(replay, batch_size=int(batch_size))

        if int(eval_every) > 0 and (t % int(eval_every) == 0):
            # Update EMA once per eval window
            c_max_ema = float(tau_c * float(c_max_ema) + (1.0 - tau_c) * float(max(c_max_seen, 1e-6)))
            c_max_seen = 0.0

            es = rollout_eval(n_episodes=5)
            eval_step.append(int(t))
            eval_avg_reward.append(float(es.avg_reward))
            eval_avg_final_dist.append(float(es.avg_final_dist))
            eval_std_final_dist.append(float(es.std_final_dist))
            eval_avg_steps.append(float(es.avg_steps))
            eval_success_rate.append(float(es.success_rate))
            eval_violation_rate.append(float(es.violation_rate))
            eval_alpha.append(float(agent.alpha))
            eval_c_max_ema.append(float(c_max_ema))
            print(
                f"[EVAL] step={t} avg_reward={es.avg_reward:.3f} avg_final_dist={es.avg_final_dist:.4f} std_dist={es.std_final_dist:.4f} "
                f"avg_steps={es.avg_steps:.1f} success_rate={es.success_rate*100.0:.1f}% violation_rate={es.violation_rate*100.0:.1f}% alpha={agent.alpha:.4f}"
            )

            save_model(last_path)
            print(f"[EVAL] saved last model to {last_path}")

            if best_stats is None or _is_better(es, best_stats):
                best_stats = es
                best_step = int(t)
                save_model(best_path)
                print(
                    f"[EVAL] new best @ step={t}: success={es.success_rate*100.0:.1f}% "
                    f"final_dist={es.avg_final_dist:.4f} reward={es.avg_reward:.3f}"
                )
                print(f"[EVAL] saved best model to {best_path}")

            if os.path.exists(best_path):
                shutil.copyfile(best_path, save_path)
                print(f"[EVAL] updated {save_path} -> best checkpoint")

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

    # Save training logs
    np.savez(
        log_path,
        episode_end_step=np.asarray(episode_end_step, dtype=np.int32),
        episode_return=np.asarray(episode_return, dtype=np.float32),
        episode_len=np.asarray(episode_len, dtype=np.int32),
        episode_final_dist=np.asarray(episode_final_dist, dtype=np.float32),
        episode_success=np.asarray(episode_success, dtype=np.int8),
        episode_violation=np.asarray(episode_violation, dtype=np.int8),
        eval_step=np.asarray(eval_step, dtype=np.int32),
        eval_avg_reward=np.asarray(eval_avg_reward, dtype=np.float32),
        eval_avg_final_dist=np.asarray(eval_avg_final_dist, dtype=np.float32),
        eval_std_final_dist=np.asarray(eval_std_final_dist, dtype=np.float32),
        eval_avg_steps=np.asarray(eval_avg_steps, dtype=np.float32),
        eval_success_rate=np.asarray(eval_success_rate, dtype=np.float32),
        eval_violation_rate=np.asarray(eval_violation_rate, dtype=np.float32),
        eval_alpha=np.asarray(eval_alpha, dtype=np.float32),
        eval_c_max_ema=np.asarray(eval_c_max_ema, dtype=np.float32),
        # meta
        total_steps=int(total_steps),
        seed=int(seed),
        xml_path=str(xml_path),
        vel_bound=float(env.vel_bound),
        acc_bound=float(env.acc_bound),
        constraint_discount_use_amount=bool(constraint_discount_use_amount),
        tdcd_p_max=float(tdcd_p_max),
        tdcd_tau_c=float(tdcd_tau_c),
        violation_tol=float(getattr(env, "violation_tol", float(violation_tol))),
        violation_agg=str(getattr(env, "violation_agg", str(violation_agg))),
        action_repeat=int(env.action_repeat),
        max_ep_steps=int(env.max_steps),
        reach_tol=float(env.reach_tol),
    )
    print(f"[LOG] training log saved: {log_path}")

    # Save MATLAB log (.mat)
    if savemat is None:
        print(
            "[WARN] scipy not installed; skipping .mat export. "
            "Install with: pip install scipy (or convert the .npz using tools/npz_to_mat.py)"
        )
    else:
        payload = {
            "episode_end_step": np.asarray(episode_end_step, dtype=np.int32).reshape(-1),
            "episode_return": np.asarray(episode_return, dtype=np.float32).reshape(-1),
            "episode_len": np.asarray(episode_len, dtype=np.int32).reshape(-1),
            "episode_final_dist": np.asarray(episode_final_dist, dtype=np.float32).reshape(-1),
            "episode_success": np.asarray(episode_success, dtype=np.int8).reshape(-1),
            "episode_violation": np.asarray(episode_violation, dtype=np.int8).reshape(-1),
            "eval_step": np.asarray(eval_step, dtype=np.int32).reshape(-1),
            "eval_avg_reward": np.asarray(eval_avg_reward, dtype=np.float32).reshape(-1),
            "eval_avg_final_dist": np.asarray(eval_avg_final_dist, dtype=np.float32).reshape(-1),
            "eval_std_final_dist": np.asarray(eval_std_final_dist, dtype=np.float32).reshape(-1),
            "eval_avg_steps": np.asarray(eval_avg_steps, dtype=np.float32).reshape(-1),
            "eval_success_rate": np.asarray(eval_success_rate, dtype=np.float32).reshape(-1),
            "eval_violation_rate": np.asarray(eval_violation_rate, dtype=np.float32).reshape(-1),
            "eval_alpha": np.asarray(eval_alpha, dtype=np.float32).reshape(-1),
            "eval_c_max_ema": np.asarray(eval_c_max_ema, dtype=np.float32).reshape(-1),
            # meta
            "total_steps": int(total_steps),
            "seed": int(seed),
            "xml_path": str(xml_path),
            "vel_bound": float(env.vel_bound),
            "acc_bound": float(env.acc_bound),
            "violation_tol": float(getattr(env, "violation_tol", float(violation_tol))),
            "violation_agg": str(getattr(env, "violation_agg", str(violation_agg))),
            "constraint_discount_use_amount": bool(constraint_discount_use_amount),
            "tdcd_p_max": float(tdcd_p_max),
            "tdcd_tau_c": float(tdcd_tau_c),
            "action_repeat": int(env.action_repeat),
            "max_ep_steps": int(env.max_steps),
            "reach_tol": float(env.reach_tol),
        }
        savemat(str(mat_path), payload, do_compression=True)
        print(f"[MAT] training log saved: {mat_path}")

    # Plot training curves
    if plt is None:
        print("[WARN] matplotlib not installed; skipping training plots. Install with: pip install matplotlib")
        return

    fig = plt.figure(figsize=(12, 10))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    if len(episode_end_step) > 0:
        ax1.plot(episode_end_step, episode_return, linewidth=1.2)
    ax1.set_title("CD-SAC (TD-CD): episode return")
    ax1.set_xlabel("env step")
    ax1.set_ylabel("return")
    ax1.grid(True, alpha=0.3)

    if len(eval_step) > 0:
        x = np.asarray(eval_step, dtype=np.float64)
        y = np.asarray(eval_avg_final_dist, dtype=np.float64)
        ystd = np.asarray(eval_std_final_dist, dtype=np.float64) if len(eval_std_final_dist) == len(eval_avg_final_dist) else None
        ax2.plot(x, y, linewidth=1.6, label="avg_final_dist")
        if ystd is not None and np.any(ystd > 0):
            ax2.fill_between(x, y - ystd, y + ystd, alpha=0.2, label="±1 std")
        ax2.set_title("CD-SAC (TD-CD): eval distance")
        ax2.set_xlabel("env step")
        ax2.set_ylabel("dist")
        ax2.grid(True, alpha=0.3)

        if len(eval_avg_steps) == len(eval_step):
            ax2b = ax2.twinx()
            ax2b.plot(eval_step, eval_avg_steps, linewidth=1.2, alpha=0.8, color="tab:orange", label="avg_steps")
            ax2b.set_ylabel("steps")
            h1, l1 = ax2.get_legend_handles_labels()
            h2, l2 = ax2b.get_legend_handles_labels()
            ax2.legend(h1 + h2, l1 + l2, loc="best")
        else:
            ax2.legend(loc="best")

        ax3.plot(eval_step, np.asarray(eval_success_rate) * 100.0, linewidth=1.6, label="success_rate")
        ax3.plot(eval_step, np.asarray(eval_violation_rate) * 100.0, linewidth=1.6, label="violation_rate")
        ax3.set_title("CD-SAC (TD-CD): eval rates")
        ax3.set_xlabel("env step")
        ax3.set_ylabel("rate (%)")
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc="best")

        ax4.plot(eval_step, eval_avg_reward, linewidth=1.6, label="avg_reward")
        ax4_t = ax4.twinx()
        ax4_t.plot(eval_step, eval_alpha, linewidth=1.2, alpha=0.8, color="tab:orange", label="alpha")
        ax4.set_title("CD-SAC (TD-CD): eval reward / alpha")
        ax4.set_xlabel("env step")
        ax4.set_ylabel("avg_reward")
        ax4_t.set_ylabel("alpha")
        ax4.grid(True, alpha=0.3)

        # merge legends
        h1, l1 = ax4.get_legend_handles_labels()
        h2, l2 = ax4_t.get_legend_handles_labels()
        ax4.legend(h1 + h2, l1 + l2, loc="best")
    else:
        ax2.set_axis_off()
        ax3.set_axis_off()
        ax4.set_axis_off()

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    print(f"[PLOT] training curves saved: {plot_path}")
    if bool(show_plot):
        plt.show()


if __name__ == "__main__":
    raise SystemExit(
        "This module is meant to be imported and called. Use cd_sac_t12a_14_cli.py train."
    )
