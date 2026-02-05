#!/usr/bin/env python3

from __future__ import annotations

import os
import sys

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover
    plt = None

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))
for _p in (_THIS_DIR, _ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_RESULTS_DIR = os.path.join(_ROOT_DIR, "experiments", "results")


def _resolve_plot_path(plot_path: str) -> str:
    plot_path = os.path.expanduser(os.path.expandvars(str(plot_path)))
    if not os.path.isabs(plot_path) and os.path.dirname(plot_path) == "":
        plot_path = os.path.join(_RESULTS_DIR, plot_path)
    out_dir = os.path.dirname(plot_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    return plot_path


def _derive_signals_plot_path(plot_path: str) -> str:
    root, ext = os.path.splitext(str(plot_path))
    if ext.strip() == "":
        ext = ".png"
    return f"{root}_signals{ext}"


from env.envball_constraints import BallEnvironmentConstraints
from algorithms.sac.sac_utils import SACAgent


def load_model(save_path: str, state_dim: int, action_dim: int):
    import torch

    if not os.path.exists(save_path):
        raise FileNotFoundError(f"Model file not found: {save_path}")

    checkpoint = torch.load(save_path, map_location="cpu")
    auto_entropy_tuning = bool(checkpoint.get("auto_entropy_tuning", False))

    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        learning_rate=3e-4,
        alpha=float(checkpoint.get("alpha", 0.2)),
        gamma=0.99,
        tau=0.005,
        auto_entropy_tuning=auto_entropy_tuning,
        use_lr_scheduler=False,
    )

    agent.policy_net.load_state_dict(checkpoint["policy_state_dict"])
    agent.q_net1.load_state_dict(checkpoint["q1_state_dict"])
    agent.q_net2.load_state_dict(checkpoint["q2_state_dict"])
    agent.target_q_net1.load_state_dict(checkpoint["target_q1_state_dict"])
    agent.target_q_net2.load_state_dict(checkpoint["target_q2_state_dict"])

    agent.alpha = float(checkpoint.get("alpha", agent.alpha))
    if agent.auto_entropy_tuning:
        log_alpha = checkpoint.get("log_alpha", None)
        if log_alpha is not None:
            if isinstance(log_alpha, torch.Tensor):
                agent.log_alpha = log_alpha.to(torch.float32)
                agent.alpha = float(agent.log_alpha.exp().item())
            else:
                agent.log_alpha = torch.tensor(float(log_alpha), requires_grad=False)
                agent.alpha = float(agent.log_alpha.exp().item())

    return agent, checkpoint


def test_cd_sac_ball(
    *,
    model_path: str,
    target_pos: list[float],
    num_tests: int = 10,
    max_steps: int = 2000,
    plot_path: str = "cd_sac_ball_trajectories_sac.png",
    show_plot: bool = False,
    init_near_target_radius: float | None = None,
    normalize_state: bool | None = None,
    vel_bound: float | None = None,
    acc_bound: float | None = None,
    acc_limit: float | None = None,
    pd_fallback: bool = False,
    pd_radius: float = 2.0,
    pd_kp: float = 2.0,
    pd_kd: float = 0.6,
):
    plot_path = _resolve_plot_path(plot_path)

    # Read config from checkpoint when present.
    tmp_env = BallEnvironmentConstraints(target_pos=target_pos, max_steps=int(max_steps))
    agent, checkpoint = load_model(model_path, tmp_env.state_dim, tmp_env.action_dim)

    ckpt_norm = checkpoint.get("state_norm", None)
    if normalize_state is None:
        normalize_state = ckpt_norm in ("fixed_bounds", "fixed_bounds_relative")

    ckpt_vel = float(checkpoint.get("vel_bound", tmp_env.vel_bound))
    ckpt_acc = float(checkpoint.get("acc_bound", tmp_env.acceleration_bound))

    if vel_bound is None:
        vel_bound = ckpt_vel
    ckpt_acc_limit = float(checkpoint.get("acc_limit", ckpt_acc))

    if acc_bound is None:
        acc_bound = ckpt_acc
    if acc_limit is None:
        acc_limit = ckpt_acc_limit

    if float(vel_bound) != float(ckpt_vel) or float(acc_bound) != float(ckpt_acc) or float(acc_limit) != float(ckpt_acc_limit):
        print(
            "[WARN] Test bounds override differs from checkpoint: "
            f"ckpt(vel={ckpt_vel}, acc_limit={ckpt_acc_limit}, acc_bound={ckpt_acc}) vs "
            f"test(vel={vel_bound}, acc_limit={acc_limit}, acc_bound={acc_bound}). "
            "This can change success rate because dynamics/normalization effectively change."
        )

    env = BallEnvironmentConstraints(
        target_pos=target_pos,
        max_steps=int(max_steps),
        reward_scale=100.0,
        vel_bound=float(vel_bound),
        acceleration_bound=float(acc_limit),
        acc_bound=float(acc_bound),
    )

    def _norm_state(s: np.ndarray) -> np.ndarray:
        if not normalize_state:
            return np.asarray(s, dtype=np.float32)
        out = np.asarray(s, dtype=np.float32).copy()
        if ckpt_norm == "fixed_bounds_relative":
            out[0] = (out[0] - float(env.target_pos[0])) / float(env.pos_bound)
            out[1] = (out[1] - float(env.target_pos[1])) / float(env.pos_bound)
        else:
            out[0] = out[0] / float(env.pos_bound)
            out[1] = out[1] / float(env.pos_bound)
        out[2] = out[2] / float(env.vel_bound)
        out[3] = out[3] / float(env.vel_bound)
        return out

    trajectories: list[np.ndarray] = []
    signals: list[dict[str, np.ndarray]] = []
    successes = 0
    violations = 0

    for k in range(int(num_tests)):
        if init_near_target_radius is not None:
            r = float(init_near_target_radius)
            init_state = np.zeros(4, dtype=np.float32)
            init_state[0] = float(env.target_pos[0]) + np.random.uniform(-r, r)
            init_state[1] = float(env.target_pos[1]) + np.random.uniform(-r, r)
            state = env.reset(initial_state=init_state)
        else:
            state = env.reset()

        traj_xy = [np.asarray(state[:2], dtype=np.float32).copy()]
        ts_vx: list[float] = [float(state[2])]
        ts_vy: list[float] = [float(state[3])]
        ts_ax: list[float] = []
        ts_ay: list[float] = []
        while True:
            if float(np.linalg.norm(state[:2] - env.target_pos)) < float(env.reach_threshold):
                break
            a_sac = agent.select_action(_norm_state(state), evaluate=True)
            a_sac = np.clip(np.asarray(a_sac, dtype=np.float32), -1.0, 1.0)

            # Optional stabilizing fallback near the target (helps reduce occasional limit-cycles).
            a = a_sac
            if bool(pd_fallback):
                dist = float(np.linalg.norm(state[:2] - env.target_pos))
                if dist < float(pd_radius):
                    pos_err = np.asarray(env.target_pos - state[:2], dtype=np.float32)
                    vel = np.asarray(state[2:4], dtype=np.float32)
                    desired_acc = float(pd_kp) * pos_err - float(pd_kd) * vel
                    a_pd = desired_acc / float(env.acceleration_bound)
                    a_pd = np.clip(np.asarray(a_pd, dtype=np.float32), -1.0, 1.0)

                    w = float(np.clip((float(pd_radius) - dist) / max(1e-6, float(pd_radius)), 0.0, 1.0))
                    a = (1.0 - w) * a_sac + w * a_pd
                    a = np.clip(np.asarray(a, dtype=np.float32), -1.0, 1.0)

            next_state, _, done, info = env.step(a)
            # Record *applied* acceleration (after clipping to acc_bound).
            ts_ax.append(float(info.get("ax", float(a[0]) * float(env.acceleration_bound))))
            ts_ay.append(float(info.get("ay", float(a[1]) * float(env.acceleration_bound))))
            traj_xy.append(np.asarray(next_state[:2], dtype=np.float32).copy())
            ts_vx.append(float(next_state[2]))
            ts_vy.append(float(next_state[3]))
            state = next_state
            if done:
                if bool(info.get("constraint_violation", False)):
                    violations += 1
                break

        final_dist = float(np.linalg.norm(state[:2] - env.target_pos))
        if final_dist < float(env.reach_threshold):
            successes += 1

        trajectories.append(np.asarray(traj_xy, dtype=np.float32))

        signals.append(
            {
                "vx": np.asarray(ts_vx, dtype=np.float32),
                "vy": np.asarray(ts_vy, dtype=np.float32),
                "ax": np.asarray(ts_ax, dtype=np.float32),
                "ay": np.asarray(ts_ay, dtype=np.float32),
            }
        )

    print(f"Success Rate: {successes}/{num_tests} ({(successes/max(1,int(num_tests))*100.0):.1f}%)")
    print(f"Violation Rate: {violations}/{num_tests} ({(violations/max(1,int(num_tests))*100.0):.1f}%)")

    if plt is None:
        print("[WARN] matplotlib is not installed; skipping plots. Install with: pip install matplotlib")
        return trajectories

    fig = plt.figure(figsize=(10, 10))
    ax_xy = fig.add_subplot(1, 1, 1)
    for i, traj in enumerate(trajectories):
        ax_xy.plot(traj[:, 0], traj[:, 1], linewidth=1.8, alpha=0.7, label=f"Test {i+1}" if i < 1 else "")
        ax_xy.plot(traj[0, 0], traj[0, 1], "s", markersize=7, color="black", label="Start" if i == 0 else "")

    ax_xy.plot(float(env.target_pos[0]), float(env.target_pos[1]), "rx", markersize=12, label="Target")
    ax_xy.set_xlim(-10, 10)
    ax_xy.set_ylim(-10, 10)
    ax_xy.grid(True)
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.set_title("CD SAC Ball: SAC trajectories")
    ax_xy.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    if show_plot:
        plt.show()
    plt.close(fig)

    # Save signals plot: 2x2 subplots (vx, vy, ax, ay) with bounds, overlay all rollouts.
    signals_plot_path = _derive_signals_plot_path(plot_path)
    fig_s = plt.figure(figsize=(14, 8))
    ax_vx = fig_s.add_subplot(2, 2, 1)
    ax_vy = fig_s.add_subplot(2, 2, 2)
    ax_ax = fig_s.add_subplot(2, 2, 3)
    ax_ay = fig_s.add_subplot(2, 2, 4)

    vb = float(env.vel_bound)
    ab = float(getattr(env, "acc_bound", env.acceleration_bound))

    for i, sig in enumerate(signals):
        vx = sig["vx"]
        vy = sig["vy"]
        ax_ = sig["ax"]
        ay_ = sig["ay"]

        t_v = np.arange(vx.shape[0], dtype=np.int32)
        t_a = np.arange(ax_.shape[0], dtype=np.int32)
        alpha = 0.22 if len(signals) > 1 else 0.9
        lw = 1.2

        ax_vx.plot(t_v, vx, alpha=alpha, linewidth=lw)
        ax_vy.plot(t_v, vy, alpha=alpha, linewidth=lw)
        ax_ax.plot(t_a, ax_, alpha=alpha, linewidth=lw)
        ax_ay.plot(t_a, ay_, alpha=alpha, linewidth=lw)

        if i == 0:
            ax_vx.axhline(+vb, color="black", alpha=0.35, linestyle=":", linewidth=1.5)
            ax_vx.axhline(-vb, color="black", alpha=0.35, linestyle=":", linewidth=1.5)
            ax_vy.axhline(+vb, color="black", alpha=0.35, linestyle=":", linewidth=1.5)
            ax_vy.axhline(-vb, color="black", alpha=0.35, linestyle=":", linewidth=1.5)
            ax_ax.axhline(+ab, color="black", alpha=0.35, linestyle=":", linewidth=1.5)
            ax_ax.axhline(-ab, color="black", alpha=0.35, linestyle=":", linewidth=1.5)
            ax_ay.axhline(+ab, color="black", alpha=0.35, linestyle=":", linewidth=1.5)
            ax_ay.axhline(-ab, color="black", alpha=0.35, linestyle=":", linewidth=1.5)

    ax_vx.set_title(r"$v_x$ with bounds")
    ax_vy.set_title(r"$v_y$ with bounds")
    ax_ax.set_title(r"$a_x$ with bounds")
    ax_ay.set_title(r"$a_y$ with bounds")

    for ax in (ax_vx, ax_vy, ax_ax, ax_ay):
        ax.set_xlabel("Step")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.25)

    fig_s.suptitle("All rollouts: velocity/acceleration components and bounds", y=1.02)
    fig_s.tight_layout()
    fig_s.savefig(signals_plot_path)
    print(f"Signals plot saved to {signals_plot_path}")
    if show_plot:
        plt.show()
    plt.close(fig_s)

    return trajectories


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Test SAC on BallEnvironmentConstraints (TD-CD style)")
    p.add_argument("--model_path", type=str, default=os.path.join(_THIS_DIR, "models", "cd_sac_ball_model_online.pth"))
    p.add_argument("--target_x", type=float, default=3.0)
    p.add_argument("--target_y", type=float, default=3.0)
    p.add_argument("--num_tests", type=int, default=10)
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--plot_path", type=str, default=os.path.join(_RESULTS_DIR, "cd_sac_ball_trajectories_sac.png"))
    p.add_argument("--show_plot", action="store_true")
    p.add_argument("--init_near_target_radius", type=float, default=None)

    p.add_argument("--vel_bound", type=float, default=None)
    p.add_argument("--acc_bound", type=float, default=None)
    p.add_argument("--acc_limit", type=float, default=None)
    p.add_argument("--normalize_state", type=int, default=None, help="Override checkpoint: 1/0")

    args = p.parse_args()

    norm = None if args.normalize_state is None else bool(int(args.normalize_state))
    test_cd_sac_ball(
        model_path=str(args.model_path),
        target_pos=[float(args.target_x), float(args.target_y)],
        num_tests=int(args.num_tests),
        max_steps=int(args.max_steps),
        plot_path=str(args.plot_path),
        show_plot=bool(args.show_plot),
        init_near_target_radius=args.init_near_target_radius,
        normalize_state=norm,
        vel_bound=args.vel_bound,
        acc_limit=args.acc_limit,
        acc_bound=args.acc_bound,
    )


if __name__ == "__main__":
    main()
