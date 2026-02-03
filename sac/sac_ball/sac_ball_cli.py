"""Unified CLI for the Ball + SAC demo.

This CLI intentionally supports only:
    - Online interaction training ("train")
    - Testing / plotting trajectories ("test", "test_near")

Offline dataset generation / offline training workflows were removed.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

# Ensure imports work no matter where this is launched from.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# Project root is .../rl_mppi (two levels up from .../sac/sac_ball)
_ROOT_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)


@dataclass(frozen=True)
class Target:
    x: float
    y: float

    @property
    def pos(self) -> list[float]:
        return [self.x, self.y]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAC Ball unified CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_target(p: argparse.ArgumentParser) -> None:
        p.add_argument("--target_x", type=float, default=3.0)
        p.add_argument("--target_y", type=float, default=3.0)

    # train
    p_train = sub.add_parser("train", help="Train SAC online by interacting with the environment")
    add_target(p_train)
    p_train.add_argument("--save_path", type=str, default="models/sac_ball_model_online.pth")
    p_train.add_argument("--total_steps", type=int, default=200000)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--eval_every", type=int, default=20000)

    # test
    p_test = sub.add_parser("test", help="Test SAC and save trajectory plot")
    add_target(p_test)
    p_test.add_argument("--model_path", type=str, default="models/sac_ball_model_online.pth")
    p_test.add_argument("--num_tests", type=int, default=10)
    p_test.add_argument("--max_steps", type=int, default=2000)
    p_test.add_argument("--plot_path", type=str, default="ball_trajectories.png")
    p_test.add_argument("--show_plot", action="store_true", help="Show GUI plot (may block)")
    p_test.add_argument("--init_near_target_radius", type=float, default=None, help="Initialize near target (test only)")

    # test_near (convenience)
    p_test_near = sub.add_parser("test_near", help="Convenience test: init near target (radius=1) and save plot to outputs/")
    add_target(p_test_near)
    p_test_near.add_argument("--model_path", type=str, default="models/sac_ball_model_online.pth")
    p_test_near.add_argument("--num_tests", type=int, default=10)
    p_test_near.add_argument("--max_steps", type=int, default=2000)
    p_test_near.add_argument("--plot_path", type=str, default="outputs/ball_trajectories.png")
    p_test_near.add_argument("--show_plot", action="store_true", help="Show GUI plot (may block)")
    p_test_near.add_argument("--init_near_target_radius", type=float, default=1.0)

    return parser.parse_args()


def _run_default() -> None:
    """Default workflow when running without CLI args.

    Runs online training with stable defaults, then runs a near-target test and
    saves a trajectory plot under outputs/.
    """
    target = Target(x=3.0, y=3.0)
    save_path = "models/sac_ball_model_online.pth"
    plot_path = "outputs/ball_trajectories.png"

    print("[DEFAULT] Running online training...")
    cmd_train_online(
        target=target,
        save_path=save_path,
        total_steps=200000,
        seed=42,
        eval_every=20000,
    )

    print("[DEFAULT] Running test_near...")
    cmd_test(
        target=target,
        model_path=save_path,
        num_tests=10,
        max_steps=4000,
        plot_path=plot_path,
        show_plot=False,
        init_near_target_radius=1.0,
    )


def _target_from_args(args: argparse.Namespace) -> Target:
    return Target(x=float(args.target_x), y=float(args.target_y))


def cmd_train_online(
    *,
    target: Target,
    save_path: str,
    total_steps: int,
    seed: int,
    eval_every: int,
) -> None:
    from train_sac_ball_online import train_sac_ball_online

    # Proven stable defaults for this project:
    # - relative-to-target state normalization
    # - curriculum reset span (near target -> wider)
    # - auto entropy tuning
    # - reach threshold consistent with test success criterion
    curriculum_start = 1.5
    curriculum_end = 5.0

    train_sac_ball_online(
        target_pos=target.pos,
        save_path=save_path,
        total_steps=total_steps,
        seed=seed,
        eval_every=eval_every,
        reach_threshold=0.5,
        auto_entropy_tuning=True,
        normalize_state=True,
        curriculum_reset_span_start=curriculum_start,
        curriculum_reset_span_end=curriculum_end,
    )


def cmd_test(
    *,
    target: Target,
    model_path: str,
    num_tests: int,
    max_steps: int,
    plot_path: str,
    show_plot: bool,
    init_near_target_radius: float | None,
) -> None:
    from test_sac_ball import test_sac_ball

    if not os.path.exists(model_path):
        raise SystemExit(
            f"Model file not found: {model_path}\n"
            "Run: python sac_ball_cli.py train (or pass --model_path to an existing .pth)"
        )

    test_sac_ball(
        model_path=model_path,
        target_pos=target.pos,
        num_tests=num_tests,
        max_steps=max_steps,
        plot_path=plot_path,
        show_plot=show_plot,
        init_near_target_radius=init_near_target_radius,
    )


def main() -> None:
    # If launched with no arguments, run the default train -> test workflow.
    if len(sys.argv) == 1:
        _run_default()
        return

    args = _parse_args()
    target = _target_from_args(args)

    if args.cmd == "train":
        cmd_train_online(
            target=target,
            save_path=str(args.save_path),
            total_steps=int(args.total_steps),
            seed=int(args.seed),
            eval_every=int(args.eval_every),
        )
        return

    if args.cmd == "test":
        cmd_test(
            target=target,
            model_path=str(args.model_path),
            num_tests=int(args.num_tests),
            max_steps=int(args.max_steps),
            plot_path=str(args.plot_path),
            show_plot=bool(args.show_plot),
            init_near_target_radius=args.init_near_target_radius,
        )
        return

    if args.cmd == "test_near":
        cmd_test(
            target=target,
            model_path=str(args.model_path),
            num_tests=int(args.num_tests),
            max_steps=int(args.max_steps),
            plot_path=str(args.plot_path),
            show_plot=bool(args.show_plot),
            init_near_target_radius=args.init_near_target_radius,
        )
        return

    raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
