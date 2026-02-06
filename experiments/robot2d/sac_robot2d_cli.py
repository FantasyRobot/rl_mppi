#!/usr/bin/env python3

"""Unified CLI for Robot2D + SAC.

Commands:
  - train: online interaction training
  - test:  evaluation and optional trajectory plot

Example:
  python experiments/robot2d/sac_robot2d_cli.py train --total_steps 250000
  python experiments/robot2d/sac_robot2d_cli.py test --model_path experiments/robot2d/models/sac_robot2d_model_online.pth
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
_MODELS_DIR = os.path.join(_ROOT_DIR, "experiments", "robot2d", "models")


@dataclass(frozen=True)
class Target:
    x: float
    y: float

    @property
    def pos(self) -> list[float]:
        return [float(self.x), float(self.y)]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robot2D SAC unified CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_target(p: argparse.ArgumentParser) -> None:
        p.add_argument("--target_x", type=float, default=-2.8)
        p.add_argument("--target_y", type=float, default=1.8)

    def add_robot(p: argparse.ArgumentParser) -> None:
        p.add_argument("--link_lengths", type=str, default="2.0,2.0", help="Comma-separated link lengths")
        p.add_argument("--obstacles", type=str, default="0,1.8,0.20", help="Obstacles as 'x,y,r;x,y,r'")

    p_train = sub.add_parser("train", help="Train SAC online by interacting with the environment")
    add_robot(p_train)
    add_target(p_train)
    p_train.add_argument("--save_path", type=str, default=os.path.join(_MODELS_DIR, "sac_robot2d_model_online.pth"))
    p_train.add_argument("--total_steps", type=int, default=250000)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--eval_every", type=int, default=25000)
    p_train.add_argument("--include_obstacles_in_obs", type=int, default=1)
    p_train.add_argument("--max_obstacles_in_obs", type=int, default=4)

    p_test = sub.add_parser("test", help="Test SAC and optionally save a trajectory plot")
    add_robot(p_test)
    add_target(p_test)
    p_test.add_argument("--model_path", type=str, default=os.path.join(_MODELS_DIR, "sac_robot2d_model_online.pth"))
    p_test.add_argument("--num_tests", type=int, default=10)
    p_test.add_argument("--max_steps", type=int, default=450)
    p_test.add_argument("--plot_path", type=str, default=os.path.join(_RESULTS_DIR, "robot2d_sac_test.png"))
    p_test.add_argument("--show_plot", action="store_true")

    return parser.parse_args()


def _target_from_args(args: argparse.Namespace) -> Target:
    return Target(x=float(args.target_x), y=float(args.target_y))


def _link_lengths_from_args(args: argparse.Namespace) -> list[float]:
    return [float(x.strip()) for x in str(args.link_lengths).split(",") if x.strip()]


def cmd_train(
    *,
    link_lengths: list[float],
    target: Target,
    obstacles: str,
    save_path: str,
    total_steps: int,
    seed: int,
    eval_every: int,
    include_obstacles_in_obs: int,
    max_obstacles_in_obs: int,
) -> None:
    from train_sac_robot2d_online import train_sac_robot2d_online, _parse_obstacles

    train_sac_robot2d_online(
        link_lengths=link_lengths,
        target_pos=target.pos,
        obstacles=_parse_obstacles(str(obstacles)),
        save_path=str(save_path),
        total_steps=int(total_steps),
        seed=int(seed),
        eval_every=int(eval_every),
        auto_entropy_tuning=True,
        normalize_state=True,
        include_obstacles_in_obs=bool(int(include_obstacles_in_obs)),
        max_obstacles_in_obs=int(max_obstacles_in_obs),
    )


def cmd_test(*, link_lengths: list[float], target: Target, obstacles: str, model_path: str, num_tests: int, max_steps: int, plot_path: str, show_plot: bool) -> None:
    from test_sac_robot2d import test_sac_robot2d, _parse_obstacles

    if not os.path.exists(model_path):
        raise SystemExit(
            f"Model file not found: {model_path}\n"
            "Run: python experiments/robot2d/sac_robot2d_cli.py train (or pass --model_path)"
        )

    out = test_sac_robot2d(
        model_path=str(model_path),
        link_lengths=link_lengths,
        target_pos=target.pos,
        obstacles=_parse_obstacles(str(obstacles)),
        num_tests=int(num_tests),
        max_steps=int(max_steps),
        plot_path=str(plot_path) if str(plot_path).strip() else None,
        show_plot=bool(show_plot),
    )
    print(out)


def main() -> None:
    if len(sys.argv) == 1:
        print(__doc__)
        print("\nTip: run with 'train' or 'test'.")
        raise SystemExit(2)

    args = _parse_args()
    target = _target_from_args(args)
    link_lengths = _link_lengths_from_args(args)

    if args.cmd == "train":
        cmd_train(
            link_lengths=link_lengths,
            target=target,
            obstacles=str(args.obstacles),
            save_path=str(args.save_path),
            total_steps=int(args.total_steps),
            seed=int(args.seed),
            eval_every=int(args.eval_every),
            include_obstacles_in_obs=int(args.include_obstacles_in_obs),
            max_obstacles_in_obs=int(args.max_obstacles_in_obs),
        )
        return

    if args.cmd == "test":
        cmd_test(
            link_lengths=link_lengths,
            target=target,
            obstacles=str(args.obstacles),
            model_path=str(args.model_path),
            num_tests=int(args.num_tests),
            max_steps=int(args.max_steps),
            plot_path=str(args.plot_path),
            show_plot=bool(args.show_plot),
        )
        return

    raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
