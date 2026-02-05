"""Unified CLI for CD SAC Ball (TD-CD constraints).

This mirrors experiments/sac_ball/sac_ball_cli.py but uses a constrained env:
- acceleration and velocity are constrained per component
- constraints are handled via TD-CD (Eq.6-9 style) discounting in TD backup

Quick start:
    python cd_sac_ball_cli.py train
    python cd_sac_ball_cli.py test_near
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

_RESULTS_DIR = os.path.join(_ROOT_DIR, "experiments", "results")


def _derive_ckpt_path(path: str, suffix: str) -> str:
    root, ext = os.path.splitext(str(path))
    if ext.strip() == "":
        ext = ".pth"
    return f"{root}{suffix}{ext}"


@dataclass(frozen=True)
class Target:
    x: float
    y: float

    @property
    def pos(self) -> list[float]:
        return [self.x, self.y]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CD SAC Ball (TD-CD) unified CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_target(p: argparse.ArgumentParser) -> None:
        p.add_argument("--target_x", type=float, default=3.0)
        p.add_argument("--target_y", type=float, default=3.0)

    def add_constraints_train(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--vel_bound",
            type=float,
            default=2.0,
            help="Velocity component bound |vx|<=vel_bound and |vy|<=vel_bound",
        )
        p.add_argument(
            "--acc_bound",
            type=float,
            default=0.5,
            help="Effective acceleration component bound |ax|<=acc_bound and |ay|<=acc_bound",
        )
        p.add_argument(
            "--acc_limit",
            type=float,
            default=1.0,
            help="Physical acceleration limit used to scale action -> raw accel before acc_bound clipping",
        )
        p.add_argument(
            "--constraint_discount_use_amount",
            type=int,
            default=0,
            help="TD-CD: use continuous violation amount (sum of component overflow) instead of binary (1/0)",
        )
        p.add_argument("--tdcd_p_max", type=float, default=1.0, help="TD-CD Eq.(7): p_max in delta")
        p.add_argument("--tdcd_tau_c", type=float, default=0.99, help="TD-CD Eq.(8): EMA factor for c_max")

    def add_constraints_test(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--vel_bound",
            type=float,
            default=None,
            help="Override checkpoint: velocity component bound |vx|<=vel_bound and |vy|<=vel_bound",
        )
        p.add_argument(
            "--acc_bound",
            type=float,
            default=None,
            help="Override checkpoint: effective acceleration bound |ax|<=acc_bound and |ay|<=acc_bound",
        )
        p.add_argument(
            "--acc_limit",
            type=float,
            default=None,
            help="Override checkpoint: physical acceleration limit used to scale action -> raw accel",
        )
        p.add_argument(
            "--constraint_discount_use_amount",
            type=int,
            default=None,
            help="Override checkpoint: use continuous violation amount (1) instead of binary (0)",
        )
        p.add_argument("--tdcd_p_max", type=float, default=None, help="Override checkpoint: TD-CD Eq.(7) p_max")
        p.add_argument("--tdcd_tau_c", type=float, default=None, help="Override checkpoint: TD-CD Eq.(8) tau_c")

    # train
    p_train = sub.add_parser("train", help="Train SAC online with constraints")
    add_target(p_train)
    add_constraints_train(p_train)
    p_train.add_argument("--save_path", type=str, default="models/cd_sac_ball_model_online.pth")
    p_train.add_argument("--total_steps", type=int, default=200000)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--eval_every", type=int, default=20000)
    p_train.add_argument("--normalize_state", type=int, default=1, help="Use fixed-bounds relative normalization (1/0)")

    # test
    p_test = sub.add_parser("test", help="Test SAC and save trajectory plot")
    add_target(p_test)
    add_constraints_test(p_test)
    p_test.add_argument("--model_path", type=str, default="models/cd_sac_ball_model_online.pth")
    p_test.add_argument(
        "--use_best",
        type=int,
        default=1,
        help="If a sibling *_best.pth exists, use it for testing (1/0)",
    )
    p_test.add_argument("--num_tests", type=int, default=10)
    p_test.add_argument("--max_steps", type=int, default=2000)
    p_test.add_argument("--plot_path", type=str, default=os.path.join(_RESULTS_DIR, "cd_sac_ball_trajectories_sac.png"))
    p_test.add_argument("--show_plot", action="store_true")
    p_test.add_argument("--init_near_target_radius", type=float, default=None)
    p_test.add_argument("--pd_fallback", type=int, default=0, help="Blend in a PD stabilizer near target (1/0)")
    p_test.add_argument("--pd_radius", type=float, default=2.0)
    p_test.add_argument("--pd_kp", type=float, default=2.0)
    p_test.add_argument("--pd_kd", type=float, default=0.6)

    # test_near
    p_test_near = sub.add_parser("test_near", help="Convenience test: init near target (radius=1)")
    add_target(p_test_near)
    add_constraints_test(p_test_near)
    p_test_near.add_argument("--model_path", type=str, default="models/cd_sac_ball_model_online.pth")
    p_test_near.add_argument(
        "--use_best",
        type=int,
        default=1,
        help="If a sibling *_best.pth exists, use it for testing (1/0)",
    )
    p_test_near.add_argument("--num_tests", type=int, default=10)
    p_test_near.add_argument("--max_steps", type=int, default=2000)
    p_test_near.add_argument(
        "--plot_path",
        type=str,
        default=os.path.join(_RESULTS_DIR, "cd_sac_ball_trajectories_sac_near.png"),
    )
    p_test_near.add_argument("--show_plot", action="store_true")
    p_test_near.add_argument("--init_near_target_radius", type=float, default=1.0)
    p_test_near.add_argument("--pd_fallback", type=int, default=1, help="Blend in a PD stabilizer near target (1/0)")
    p_test_near.add_argument("--pd_radius", type=float, default=2.0)
    p_test_near.add_argument("--pd_kp", type=float, default=2.0)
    p_test_near.add_argument("--pd_kd", type=float, default=0.6)

    return parser.parse_args()


def _target_from_args(args: argparse.Namespace) -> Target:
    return Target(x=float(args.target_x), y=float(args.target_y))


def cmd_train_online(*, args: argparse.Namespace, target: Target) -> None:
    from train_cd_sac_ball_online import train_cd_sac_ball_online

    curriculum_start = 1.5
    curriculum_end = 5.0

    train_cd_sac_ball_online(
        target_pos=target.pos,
        save_path=str(args.save_path),
        total_steps=int(args.total_steps),
        seed=int(args.seed),
        eval_every=int(args.eval_every),
        reach_threshold=0.5,
        auto_entropy_tuning=True,
        normalize_state=bool(int(args.normalize_state)),
        curriculum_reset_span_start=curriculum_start,
        curriculum_reset_span_end=curriculum_end,
        vel_bound=float(args.vel_bound),
        acc_bound=float(args.acc_bound),
        acc_limit=float(args.acc_limit),
        constraint_discount_use_amount=bool(int(args.constraint_discount_use_amount)),
        tdcd_p_max=float(args.tdcd_p_max),
        tdcd_tau_c=float(args.tdcd_tau_c),
    )


def cmd_test(*, args: argparse.Namespace, target: Target) -> None:
    from test_cd_sac_ball import test_cd_sac_ball

    model_path = os.path.expanduser(os.path.expandvars(str(args.model_path)))
    if bool(int(getattr(args, "use_best", 1))):
        best_path = _derive_ckpt_path(model_path, "_best")
        if os.path.exists(best_path):
            print(f"[INFO] Using best checkpoint: {best_path}")
            model_path = best_path
        else:
            print(f"[INFO] Best checkpoint not found: {best_path} (using {model_path})")

    if not os.path.exists(str(model_path)):
        raise SystemExit(
            f"Model file not found: {model_path}\n"
            "Run: python cd_sac_ball_cli.py train (or pass --model_path to an existing .pth)"
        )

    test_cd_sac_ball(
        model_path=str(model_path),
        target_pos=target.pos,
        num_tests=int(args.num_tests),
        max_steps=int(args.max_steps),
        plot_path=str(args.plot_path),
        show_plot=bool(args.show_plot),
        init_near_target_radius=args.init_near_target_radius,
        normalize_state=None,
        vel_bound=args.vel_bound,
        acc_limit=args.acc_limit,
        acc_bound=args.acc_bound,
        pd_fallback=bool(int(args.pd_fallback)),
        pd_radius=float(args.pd_radius),
        pd_kp=float(args.pd_kp),
        pd_kd=float(args.pd_kd),
    )


def main() -> None:
    args = _parse_args()
    target = _target_from_args(args)

    if args.cmd == "train":
        cmd_train_online(args=args, target=target)
        return

    if args.cmd in ("test", "test_near"):
        cmd_test(args=args, target=target)
        return

    raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
