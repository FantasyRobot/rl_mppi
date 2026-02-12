#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

_MODELS_DIR = os.path.join(_THIS_DIR, "models")
_RESULTS_DIR = os.path.join(_ROOT_DIR, "experiments", "results")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RobotT12B (t12a_14) SAC CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        default_xml = os.path.join(_THIS_DIR, "urdf", "t12a_14_normal.xml")
        p.add_argument("--xml", type=str, default=default_xml)
        p.add_argument("--goal_site", type=str, default="goal")
        p.add_argument("--eef_site", type=str, default="end_effector")
        p.add_argument("--action_repeat", type=int, default=5)
        p.add_argument("--max_steps", type=int, default=2500)
        p.add_argument("--reach_tol", type=float, default=0.03)

    p_train = sub.add_parser("train", help="Train SAC online")
    add_common(p_train)
    p_train.add_argument("--save_path", type=str, default=os.path.join(_MODELS_DIR, "sac_t12a_14_model.pth"))
    p_train.add_argument(
        "--mat_path",
        type=str,
        default=None,
        help=(
            "Optional .mat export path for training curves. "
            "If omitted and --collision_mode=cdf, a sibling .mat will be written next to the .npz log (requires scipy)."
        ),
    )
    p_train.add_argument("--total_steps", type=int, default=200000)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--eval_every", type=int, default=20000)
    p_train.add_argument(
        "--collision_mode",
        type=str,
        default="none",
        choices=["none", "stop", "cdf"],
        help="Collision handling: none (ignore), stop (terminate on collision), cdf (CDF safety shaping)",
    )
    p_train.add_argument("--obstacle_prefix", type=str, default="obstacle", help="Obstacle geom name prefix")
    p_train.add_argument("--collision_penalty", type=float, default=50.0, help="Penalty added when collision occurs")
    p_train.add_argument(
        "--terminate_on_collision",
        type=int,
        default=-1,
        help="Override termination on collision (1/0). Default: stop->1, cdf->0, none->0",
    )
    p_train.add_argument("--cdf_sigma", type=float, default=0.05)
    p_train.add_argument("--cdf_margin", type=float, default=0.0)
    p_train.add_argument("--cdf_scale", type=float, default=5.0)

    p_test = sub.add_parser("test", help="Test SAC policy")
    add_common(p_test)
    p_test.add_argument("--model_path", type=str, default=os.path.join(_MODELS_DIR, "sac_t12a_14_model.pth"))
    p_test.add_argument("--save_npz", type=str, default=os.path.join(_RESULTS_DIR, "t12a_14_sac_test.npz"))
    p_test.add_argument("--viewer", action="store_true", help="Show MuJoCo viewer during test rollout")
    p_test.add_argument("--exit_on_done", action="store_true", help="Close immediately when done (viewer only)")
    p_test.add_argument("--no_draw_traj", action="store_true", help="Disable drawing end-effector trajectory")
    p_test.add_argument("--traj_max_points", type=int, default=400, help="Max trajectory points")
    p_test.add_argument("--traj_stride", type=int, default=1, help="Add a point every N steps")
    p_test.add_argument("--traj_width", type=float, default=4.0, help="Trajectory line width")

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.cmd == "train":
        from train_sac_t12a_14_online import train_sac_t12a_14_online

        term_override = int(getattr(args, "terminate_on_collision", -1))
        terminate_on_collision = None if term_override < 0 else bool(term_override)

        train_sac_t12a_14_online(
            xml_path=str(args.xml),
            goal_site=str(args.goal_site),
            eef_site=str(args.eef_site),
            save_path=str(args.save_path),
            mat_path=(None if getattr(args, "mat_path", None) in (None, "") else str(args.mat_path)),
            total_steps=int(args.total_steps),
            seed=int(args.seed),
            eval_every=int(args.eval_every),
            action_repeat=int(args.action_repeat),
            max_ep_steps=int(args.max_steps),
            reach_tol=float(args.reach_tol),
            collision_mode=str(getattr(args, "collision_mode", "none")),
            obstacle_prefix=str(getattr(args, "obstacle_prefix", "obstacle")),
            collision_penalty=float(getattr(args, "collision_penalty", 50.0)),
            terminate_on_collision=terminate_on_collision,
            cdf_sigma=float(getattr(args, "cdf_sigma", 0.05)),
            cdf_margin=float(getattr(args, "cdf_margin", 0.0)),
            cdf_scale=float(getattr(args, "cdf_scale", 5.0)),
        )
        return

    if args.cmd == "test":
        if bool(args.viewer):
            from test_sac_t12a_14 import test_sac_t12a_14_viewer

            out = test_sac_t12a_14_viewer(
                model_path=str(args.model_path),
                xml_path=str(args.xml),
                goal_site=str(args.goal_site),
                eef_site=str(args.eef_site),
                max_steps=int(args.max_steps),
                action_repeat=int(args.action_repeat),
                exit_on_done=bool(args.exit_on_done),
                no_draw_traj=bool(args.no_draw_traj),
                traj_max_points=int(args.traj_max_points),
                traj_stride=int(args.traj_stride),
                traj_width=float(args.traj_width),
            )
        else:
            from test_sac_t12a_14 import test_sac_t12a_14

            out = test_sac_t12a_14(
                model_path=str(args.model_path),
                xml_path=str(args.xml),
                goal_site=str(args.goal_site),
                eef_site=str(args.eef_site),
                max_steps=int(args.max_steps),
                action_repeat=int(args.action_repeat),
            )

        # Save minimal trajectories for later plotting.
        import numpy as np

        save_npz = os.path.abspath(str(args.save_npz))
        out_dir = os.path.dirname(save_npz)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        np.savez(save_npz, eef=out["eef"], dist=out["dist"], success=np.array([out["success"]], dtype=np.int32))
        print("[DONE] Saved:", save_npz)
        print(f"success={out['success']} final_dist={out['final_dist']:.4f} steps={len(out['dist'])}")
        return


if __name__ == "__main__":
    main()
