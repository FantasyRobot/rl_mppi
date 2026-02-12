#!/usr/bin/env python3

from __future__ import annotations

import os
import sys

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover
    plt = None

try:
    from scipy.io import savemat  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    savemat = None

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)


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


def train_sac_t12a_14_online(
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
    # obstacles / collision handling
    collision_mode: str = "none",
    obstacle_prefix: str = "obstacle",
    collision_penalty: float = 50.0,
    terminate_on_collision: bool | None = None,
    # CDF shaping
    cdf_sigma: float = 0.05,
    cdf_margin: float = 0.0,
    cdf_scale: float = 5.0,
    # logging/plotting
    log_path: str | None = None,
    mat_path: str | None = None,
    plot_path: str | None = None,
    show_plot: bool = False,
) -> None:
    try:
        import torch
    except ModuleNotFoundError as e:
        raise SystemExit(
            "Torch is required for SAC. Install in your env, e.g. (CPU):\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
        ) from e

    from algorithms.sac.sac_utils import ReplayBuffer, SACAgent
    from env.envmujoco_t12a_14 import T12A14MuJoCoEnv
    from env.envmujoco_t12a_14_obstacles import T12A14MuJoCoEnvObstacles

    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    mode = str(collision_mode).strip().lower()
    if mode not in {"none", "stop", "cdf"}:
        raise ValueError(f"Unknown collision_mode: {collision_mode} (expected: none|stop|cdf)")

    if terminate_on_collision is None:
        terminate_on_collision = True if mode == "stop" else False

    if mode == "none":
        env = T12A14MuJoCoEnv(
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
        )
    else:
        env = T12A14MuJoCoEnvObstacles(
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
            obstacle_prefix=str(obstacle_prefix),
            collision_terminate=bool(terminate_on_collision),
            collision_penalty=float(collision_penalty),
            cdf_shaping=(mode == "cdf"),
            cdf_sigma=float(cdf_sigma),
            cdf_margin=float(cdf_margin),
            cdf_scale=float(cdf_scale),
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

    replay = ReplayBuffer(max_size=400_000)

    save_path = os.path.expanduser(os.path.expandvars(str(save_path)))
    if log_path is None:
        log_path = _derive_sidecar_path(save_path, "_train_log", ".npz")
    if plot_path is None:
        plot_path = _derive_sidecar_path(save_path, "_train", ".png")

    log_path = _resolve_out_path(str(log_path), default_dir=_RESULTS_DIR)
    plot_path = _resolve_out_path(str(plot_path), default_dir=_RESULTS_DIR)

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
            },
            path,
        )

    def eval_rollout(n_episodes: int = 5) -> tuple[float, float, float, float, float, float]:
        dists: list[float] = []
        steps: list[int] = []
        succ = 0
        collided_eps = 0
        succ_no_collision = 0
        for _ in range(int(n_episodes)):
            s = env.reset()
            ep_collided = False
            for _t in range(int(env.max_steps)):
                a = agent.select_action(s, evaluate=True)
                s, _r, done, info = env.step(a)
                if bool(info.get("collided", False)):
                    ep_collided = True
                if done:
                    dists.append(float(info.get("dist", 0.0)))
                    steps.append(int(info.get("step", 0)))
                    if bool(info.get("success", False)):
                        succ += 1
                        if not ep_collided:
                            succ_no_collision += 1
                    if ep_collided:
                        collided_eps += 1
                    break
        mean_dist = float(np.mean(dists) if dists else 0.0)
        std_dist = float(np.std(dists) if dists else 0.0)
        mean_steps = float(np.mean(steps) if steps else 0.0)
        success_rate = float(succ / max(1, int(n_episodes)))
        collision_rate = float(collided_eps / max(1, int(n_episodes)))
        success_no_collision_rate = float(succ_no_collision / max(1, int(n_episodes)))
        return mean_dist, std_dist, mean_steps, success_rate, collision_rate, success_no_collision_rate

    s = env.reset()
    ep_ret = 0.0
    ep_len = 0
    episode = 0

    # Training logs
    episode_end_step: list[int] = []
    episode_return: list[float] = []
    episode_len: list[int] = []
    episode_final_dist: list[float] = []
    episode_success: list[int] = []

    eval_step: list[int] = []
    eval_mean_dist: list[float] = []
    eval_std_dist: list[float] = []
    eval_mean_steps: list[float] = []
    eval_success_rate: list[float] = []
    eval_collision_rate: list[float] = []
    eval_success_no_collision_rate: list[float] = []
    eval_alpha: list[float] = []

    for t in range(int(total_steps)):
        if t < int(start_steps):
            a = np.random.uniform(-1.0, 1.0, size=(env.action_dim,)).astype(np.float32)
        else:
            a = agent.select_action(s, evaluate=False)

        s2, r, done, info = env.step(a)
        ep_ret += float(r)
        ep_len += 1

        replay.add(s, a, r, s2, done)
        s = s2

        if done:
            episode += 1
            episode_end_step.append(int(t + 1))
            episode_return.append(float(ep_ret))
            episode_len.append(int(ep_len))
            episode_final_dist.append(float(info.get("dist", 0.0)))
            episode_success.append(1 if bool(info.get("success", False)) else 0)
            s = env.reset()
            ep_ret = 0.0
            ep_len = 0

        if t >= int(update_after) and (t % int(update_every) == 0) and len(replay) >= int(batch_size):
            for _ in range(int(updates_per_step)):
                agent.update(replay, batch_size=int(batch_size))

        if (t + 1) % int(eval_every) == 0:
            mean_dist, std_dist, mean_steps, success_rate, collision_rate, success_nocoll = eval_rollout(n_episodes=5)
            print(
                f"[EVAL] step={t+1} mean_dist={mean_dist:.4f} std_dist={std_dist:.4f} "
                f"mean_steps={mean_steps:.1f} success={success_rate:.2f} collided={collision_rate:.2f} success_no_coll={success_nocoll:.2f}"
            )
            eval_step.append(int(t + 1))
            eval_mean_dist.append(float(mean_dist))
            eval_std_dist.append(float(std_dist))
            eval_mean_steps.append(float(mean_steps))
            eval_success_rate.append(float(success_rate))
            eval_collision_rate.append(float(collision_rate))
            eval_success_no_collision_rate.append(float(success_nocoll))
            eval_alpha.append(float(agent.alpha))
            save_model(str(save_path))

    save_model(str(save_path))
    print(f"Saved SAC checkpoint: {save_path}")

    # Save training logs
    np.savez(
        log_path,
        episode_end_step=np.asarray(episode_end_step, dtype=np.int32),
        episode_return=np.asarray(episode_return, dtype=np.float32),
        episode_len=np.asarray(episode_len, dtype=np.int32),
        episode_final_dist=np.asarray(episode_final_dist, dtype=np.float32),
        episode_success=np.asarray(episode_success, dtype=np.int8),
        eval_step=np.asarray(eval_step, dtype=np.int32),
        eval_mean_dist=np.asarray(eval_mean_dist, dtype=np.float32),
        eval_std_dist=np.asarray(eval_std_dist, dtype=np.float32),
        eval_mean_steps=np.asarray(eval_mean_steps, dtype=np.float32),
        eval_success_rate=np.asarray(eval_success_rate, dtype=np.float32),
        eval_collision_rate=np.asarray(eval_collision_rate, dtype=np.float32),
        eval_success_no_collision_rate=np.asarray(eval_success_no_collision_rate, dtype=np.float32),
        eval_alpha=np.asarray(eval_alpha, dtype=np.float32),
        # meta
        total_steps=int(total_steps),
        seed=int(seed),
        xml_path=str(xml_path),
        collision_mode=str(mode),
        obstacle_prefix=str(obstacle_prefix),
        collision_penalty=float(collision_penalty),
        terminate_on_collision=bool(terminate_on_collision),
        cdf_sigma=float(cdf_sigma),
        cdf_margin=float(cdf_margin),
        cdf_scale=float(cdf_scale),
        action_repeat=int(env.action_repeat),
        max_ep_steps=int(env.max_steps),
        reach_tol=float(env.reach_tol),
    )
    print(f"[LOG] training log saved: {log_path}")

    # Export to MATLAB .mat (for paper plotting)
    # Default behavior: when training CDF-SAC (collision_mode=cdf), export a sibling .mat next to the .npz.
    if mat_path is None and str(mode) == "cdf":
        mat_path = os.path.splitext(str(log_path))[0] + ".mat"

    if mat_path:
        mat_path = _resolve_out_path(str(mat_path), default_dir=os.path.dirname(str(log_path)) or _RESULTS_DIR)
        if savemat is None:
            print(
                "[WARN] scipy not installed; cannot export .mat. Install with: pip install scipy\n"
                f"       (kept .npz log at: {log_path})"
            )
        else:
            payload: dict[str, object] = {
                "episode_end_step": np.asarray(episode_end_step, dtype=np.int32),
                "episode_return": np.asarray(episode_return, dtype=np.float32),
                "episode_len": np.asarray(episode_len, dtype=np.int32),
                "episode_final_dist": np.asarray(episode_final_dist, dtype=np.float32),
                "episode_success": np.asarray(episode_success, dtype=np.int8),
                "eval_step": np.asarray(eval_step, dtype=np.int32),
                "eval_mean_dist": np.asarray(eval_mean_dist, dtype=np.float32),
                "eval_std_dist": np.asarray(eval_std_dist, dtype=np.float32),
                "eval_mean_steps": np.asarray(eval_mean_steps, dtype=np.float32),
                "eval_success_rate": np.asarray(eval_success_rate, dtype=np.float32),
                "eval_collision_rate": np.asarray(eval_collision_rate, dtype=np.float32),
                "eval_success_no_collision_rate": np.asarray(eval_success_no_collision_rate, dtype=np.float32),
                "eval_alpha": np.asarray(eval_alpha, dtype=np.float32),
                # meta
                "total_steps": int(total_steps),
                "seed": int(seed),
                "xml_path": str(xml_path),
                "collision_mode": str(mode),
                "obstacle_prefix": str(obstacle_prefix),
                "collision_penalty": float(collision_penalty),
                "terminate_on_collision": bool(terminate_on_collision),
                "cdf_sigma": float(cdf_sigma),
                "cdf_margin": float(cdf_margin),
                "cdf_scale": float(cdf_scale),
                "action_repeat": int(env.action_repeat),
                "max_ep_steps": int(env.max_steps),
                "reach_tol": float(env.reach_tol),
            }
            savemat(mat_path, payload, do_compression=True)
            print(f"[MAT] training log exported: {mat_path}")

    # Plot training curves
    if plt is None:
        print("[WARN] matplotlib not installed; skipping training plots. Install with: pip install matplotlib")
        return

    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    if len(episode_end_step) > 0:
        ax1.plot(episode_end_step, episode_return, linewidth=1.2)
    ax1.set_title("SAC T12A14: episode return")
    ax1.set_xlabel("env step")
    ax1.set_ylabel("return")
    ax1.grid(True, alpha=0.3)

    if len(eval_step) > 0:
        x = np.asarray(eval_step, dtype=np.float64)
        y = np.asarray(eval_mean_dist, dtype=np.float64)
        ystd = np.asarray(eval_std_dist, dtype=np.float64) if len(eval_std_dist) == len(eval_mean_dist) else None
        ax2.plot(x, y, linewidth=1.6, label="mean dist")
        if ystd is not None and np.any(ystd > 0):
            ax2.fill_between(x, y - ystd, y + ystd, alpha=0.2, label="±1 std")
    ax2.set_title("SAC T12A14: eval mean dist")
    ax2.set_xlabel("env step")
    ax2.set_ylabel("dist")
    ax2.grid(True, alpha=0.3)

    # Overlay mean steps on a secondary axis (helps interpret return).
    if len(eval_step) > 0 and len(eval_mean_steps) == len(eval_step):
        ax2b = ax2.twinx()
        ax2b.plot(eval_step, eval_mean_steps, linewidth=1.2, alpha=0.8, color="tab:orange", label="mean steps")
        ax2b.set_ylabel("steps")
        h1, l1 = ax2.get_legend_handles_labels()
        h2, l2 = ax2b.get_legend_handles_labels()
        ax2.legend(h1 + h2, l1 + l2, loc="best")
    else:
        ax2.legend(loc="best")

    if len(eval_step) > 0:
        ax3.plot(eval_step, np.asarray(eval_success_rate) * 100.0, linewidth=1.6, label="success")
        if len(eval_success_no_collision_rate) == len(eval_step):
            ax3.plot(
                eval_step,
                np.asarray(eval_success_no_collision_rate) * 100.0,
                linewidth=1.6,
                label="success_no_collision",
            )
        if len(eval_collision_rate) == len(eval_step):
            ax3.plot(eval_step, np.asarray(eval_collision_rate) * 100.0, linewidth=1.6, label="collision")
    ax3.set_title("SAC T12A14: eval success rate")
    ax3.set_xlabel("env step")
    ax3.set_ylabel("success (%)")
    ax3.grid(True, alpha=0.3)
    if len(eval_step) > 0:
        ax3.legend(loc="best")

    if len(eval_step) > 0:
        ax4.plot(eval_step, eval_alpha, linewidth=1.6)
    ax4.set_title("SAC T12A14: alpha")
    ax4.set_xlabel("env step")
    ax4.set_ylabel("alpha")
    ax4.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    print(f"[PLOT] training curves saved: {plot_path}")
    if bool(show_plot):
        plt.show()
