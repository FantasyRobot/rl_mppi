#!/usr/bin/env python3

"""Unified CLI for CD-SAC T12A14 (TD-CD constraints).

Mirrors experiments/ball2D/cd_sac_ball/cd_sac_ball_cli.py.
Default constraints:
- |qvel_i| <= 1
- |qacc_i| <= 2

Quick start:
  python experiments/robotT12B/cd_sac_t12a_14/cd_sac_t12a_14_cli.py train
  python experiments/robotT12B/cd_sac_t12a_14/cd_sac_t12a_14_cli.py test
"""

from __future__ import annotations

import argparse
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))
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


def _default_xml() -> str:
    # experiments/robotT12B/urdf/t12a_14.xml
    robot_dir = os.path.dirname(_THIS_DIR)
    return os.path.join(robot_dir, "urdf", "t12a_14_clear.xml")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CD-SAC T12A14 (TD-CD) unified CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--xml", type=str, default=_default_xml())
        p.add_argument("--goal_site", type=str, default="goal")
        p.add_argument("--eef_site", type=str, default="end_effector")
        p.add_argument("--action_repeat", type=int, default=5)
        p.add_argument("--reach_tol", type=float, default=0.03)

    def add_constraints_train(p: argparse.ArgumentParser) -> None:
        p.add_argument("--vel_bound", type=float, default=1.0, help="Joint velocity bound |qvel_i|<=vel_bound")
        p.add_argument("--acc_bound", type=float, default=2.0, help="Joint acceleration bound |qacc_i|<=acc_bound")
        p.add_argument(
            "--violation_agg",
            type=str,
            default="max",
            choices=["max", "sum"],
            help="Constraint violation aggregation across joints/action_repeat (paper default: max)",
        )
        p.add_argument(
            "--violation_tol",
            type=float,
            default=1e-3,
            help="Tolerance for counting a constraint violation (paper default: 1e-3)",
        )
        p.add_argument(
            "--constraint_discount_use_amount",
            type=int,
            default=1,
            help="TD-CD: use continuous violation amount (1) instead of binary (0)",
        )
        p.add_argument("--tdcd_p_max", type=float, default=1.0, help="TD-CD Eq.(7): p_max in delta")
        p.add_argument("--tdcd_tau_c", type=float, default=0.995, help="TD-CD Eq.(8): EMA factor for c_max")

    def add_constraints_test(p: argparse.ArgumentParser) -> None:
        p.add_argument("--vel_bound", type=float, default=None, help="Override checkpoint vel bound")
        p.add_argument("--acc_bound", type=float, default=None, help="Override checkpoint acc bound")

    # train
    p_train = sub.add_parser("train", help="Train constrained SAC (TD-CD)")
    add_common(p_train)
    add_constraints_train(p_train)
    p_train.add_argument(
        "--save_path",
        type=str,
        default=os.path.join(os.path.dirname(_THIS_DIR), "models", "cd_sac_t12a_14_model_online.pth"),
    )
    p_train.add_argument(
        "--log_path",
        type=str,
        default="",
        help="Optional: where to save training log (.npz). Default: <save_path>_train_log.npz",
    )
    p_train.add_argument(
        "--plot_path",
        type=str,
        default="",
        help="Optional: where to save training curves (.png). Default: <save_path>_train.png",
    )
    p_train.add_argument(
        "--mat_path",
        type=str,
        default="",
        help="Optional: where to save MATLAB training log (.mat). Default: <save_path>_train_log.mat",
    )
    p_train.add_argument("--show_plot", action="store_true", help="Show training curves at end")
    p_train.add_argument("--total_steps", type=int, default=200000)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--eval_every", type=int, default=20000)

    # test
    p_test = sub.add_parser("test", help="Test model and save plots")
    add_common(p_test)
    add_constraints_test(p_test)
    p_test.add_argument(
        "--model_path",
        type=str,
        default=os.path.join(os.path.dirname(_THIS_DIR), "models", "cd_sac_t12a_14_model_online.pth"),
    )
    p_test.add_argument(
        "--use_best",
        type=int,
        default=1,
        help="If a sibling *_best.pth exists, use it for testing (1/0)",
    )
    p_test.add_argument("--num_tests", type=int, default=5)
    p_test.add_argument("--max_steps", type=int, default=2500)
    p_test.add_argument(
        "--plot_path",
        type=str,
        default=os.path.join(_RESULTS_DIR, "cd_sac_t12a_14_curves.png"),
    )
    p_test.add_argument("--show_plot", action="store_true")

    # viewer
    p_test.add_argument("--viewer", action="store_true", help="Run a MuJoCo viewer rollout like MPPI")
    p_test.add_argument("--exit_on_done", action="store_true", help="Exit viewer immediately when done")
    p_test.add_argument("--no_draw_traj", action="store_true", help="Disable drawing end-effector trajectory")
    p_test.add_argument("--traj_max_points", type=int, default=400)
    p_test.add_argument("--traj_stride", type=int, default=1)
    p_test.add_argument("--traj_width", type=float, default=4.0)
    p_test.add_argument("--sleep", type=float, default=0.0, help="Optional sleep per viewer step (seconds)")
    p_test.add_argument(
        "--settle_steps",
        type=int,
        default=0,
        help="After reaching goal, keep stepping while holding current joint targets (records qvel/qacc)",
    )
    p_test.add_argument(
        "--vel_tol",
        type=float,
        default=0.02,
        help="Early-stop settling when max|qvel| drops below this threshold",
    )
    p_test.add_argument(
        "--export_joint_csv",
        type=str,
        default="",
        help="If set, export per-step joint qvel/qacc to CSV: <base>_qvel.csv and <base>_qacc.csv",
    )

    return parser.parse_args()


def cmd_train(args: argparse.Namespace) -> None:
    from train_cd_sac_t12a_14_online import train_cd_sac_t12a_14_online

    train_cd_sac_t12a_14_online(
        xml_path=str(args.xml),
        goal_site=str(args.goal_site),
        eef_site=str(args.eef_site),
        save_path=str(args.save_path),
        total_steps=int(args.total_steps),
        seed=int(args.seed),
        eval_every=int(args.eval_every),
        action_repeat=int(args.action_repeat),
        reach_tol=float(args.reach_tol),
        vel_bound=float(args.vel_bound),
        acc_bound=float(args.acc_bound),
        violation_agg=str(getattr(args, "violation_agg", "max")),
        violation_tol=float(getattr(args, "violation_tol", 1e-3)),
        constraint_discount_use_amount=bool(int(args.constraint_discount_use_amount)),
        tdcd_p_max=float(args.tdcd_p_max),
        tdcd_tau_c=float(args.tdcd_tau_c),
        log_path=(str(getattr(args, "log_path", "")).strip() or None),
        plot_path=(str(getattr(args, "plot_path", "")).strip() or None),
        mat_path=(str(getattr(args, "mat_path", "")).strip() or None),
        show_plot=bool(getattr(args, "show_plot", False)),
    )


def cmd_test(args: argparse.Namespace) -> None:
    from test_cd_sac_t12a_14 import test_cd_sac_t12a_14, test_cd_sac_t12a_14_viewer

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
            "Run: python cd_sac_t12a_14_cli.py train (or pass --model_path to an existing .pth)"
        )

    if bool(getattr(args, "viewer", False)):
        test_cd_sac_t12a_14_viewer(
            model_path=str(model_path),
            xml_path=str(args.xml),
            goal_site=str(args.goal_site),
            eef_site=str(args.eef_site),
            max_steps=int(args.max_steps),
            action_repeat=int(args.action_repeat),
            exit_on_done=bool(getattr(args, "exit_on_done", False)),
            no_draw_traj=bool(getattr(args, "no_draw_traj", False)),
            traj_max_points=int(getattr(args, "traj_max_points", 400)),
            traj_stride=int(getattr(args, "traj_stride", 1)),
            traj_width=float(getattr(args, "traj_width", 4.0)),
            sleep_sec=float(getattr(args, "sleep", 0.0)),
            export_csv_base=(str(getattr(args, "export_joint_csv", "")).strip() or None),
            settle_steps=int(getattr(args, "settle_steps", 0)),
            vel_tol=float(getattr(args, "vel_tol", 0.02)),
        )
        return

    test_cd_sac_t12a_14(
        model_path=str(model_path),
        xml_path=str(args.xml),
        goal_site=str(args.goal_site),
        eef_site=str(args.eef_site),
        num_tests=int(args.num_tests),
        max_steps=int(args.max_steps),
        action_repeat=int(args.action_repeat),
        plot_path=str(args.plot_path),
        show_plot=bool(args.show_plot),
        vel_bound=args.vel_bound,
        acc_bound=args.acc_bound,
    )


def main() -> None:
    args = _parse_args()

    if args.cmd == "train":
        cmd_train(args)
        return

    if args.cmd == "test":
        cmd_test(args)
        return

    raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
