#!/usr/bin/env python3

"""Plot Robot2D obstacle distance-field contours in configuration space and overlay two planned trajectories.

This script visualizes the configuration-space distance field produced by
`CDF2D.calculate_cdf` (level sets / contours in q1-q2), for a given set of
workspace circle obstacles.

It generates two *deterministic* trajectories in configuration space:
    1) No-CDF: a smooth "spline" (minimum-jerk) interpolation from start -> goal (IK).
    2) CDF: follow the *start contour* (cdf(q)=cdf(q_start)) while moving toward the goal (IK).

Then overlays both trajectories in:
  1) configuration space (q1-q2)
  2) task space (EEF path + optional link snapshots)

Notes:
- Requires PyTorch for CDF (CDF2D is torch-based).
- Uses `method=offline_grid` by default (does NOT require torchmin).
- Intended for 2-link Robot2D (n=2) to match CDF2D.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover
    plt = None

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

_RESULTS_DIR = os.path.join(_ROOT_DIR, "experiments", "results")

from env.envrobot2d_obstacles import CircleObstacle, Robot2DEnvironmentObstacles



def _parse_obstacles(s: str) -> list[CircleObstacle]:
    s = (s or "").strip()
    if not s:
        return []
    out: list[CircleObstacle] = []
    for part in s.split(";"):
        xs, ys, rs = [t.strip() for t in part.split(",")]
        out.append(CircleObstacle(float(xs), float(ys), float(rs)))
    return out


def _resolve_plot_path(plot_path: str) -> str:
    plot_path = os.path.expanduser(os.path.expandvars(str(plot_path)))
    if not os.path.isabs(plot_path) and os.path.dirname(plot_path) == "":
        plot_path = os.path.join(_RESULTS_DIR, plot_path)
    out_dir = os.path.dirname(plot_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    return plot_path


def _resolve_data_path(path: str) -> str:
    path = os.path.expanduser(os.path.expandvars(str(path)))
    if not os.path.isabs(path) and os.path.dirname(path) == "":
        path = os.path.join(_RESULTS_DIR, path)
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    return path


def _export_matlab_file(out_path: str, payload: dict) -> str:
    out_path = os.path.expanduser(os.path.expandvars(str(out_path)))
    if not os.path.isabs(out_path) and os.path.dirname(out_path) == "":
        out_path = os.path.join(_RESULTS_DIR, out_path)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    low = out_path.lower()
    if low.endswith(".mat"):
        try:
            from scipy.io import savemat  # type: ignore
        except ModuleNotFoundError as e:
            raise SystemExit(
                "scipy is required to export .mat. Install with: pip install scipy\n"
                "Or export to .h5 by using a .h5 extension (requires h5py)."
            ) from e
        savemat(out_path, payload, do_compression=True)
        return out_path

    if not (low.endswith(".h5") or low.endswith(".hdf5")):
        out_path = out_path + ".mat"
        try:
            from scipy.io import savemat  # type: ignore
        except ModuleNotFoundError as e:
            raise SystemExit(
                "scipy is required to export .mat. Install with: pip install scipy\n"
                "Or set --export_mat to end with .h5 and install h5py."
            ) from e
        savemat(out_path, payload, do_compression=True)
        return out_path

    try:
        import h5py  # type: ignore
    except ModuleNotFoundError as e:
        raise SystemExit("h5py is required to export .h5. Install with: pip install h5py") from e

    with h5py.File(out_path, "w") as f:
        for k, v in payload.items():
            vv = np.asarray(v)
            if vv.dtype.kind in {"U", "S"}:
                dt = h5py.string_dtype(encoding="utf-8")
                f.create_dataset(str(k), data=vv.astype(object), dtype=dt)
            else:
                f.create_dataset(str(k), data=vv)
    return out_path


def _draw_obstacles(ax, obstacles: list[CircleObstacle], *, alpha: float = 0.35) -> None:
    if plt is None:
        return
    for obs in obstacles:
        circ = plt.Circle((obs.x, obs.y), obs.r, color="gray", alpha=float(alpha))
        ax.add_patch(circ)


def _draw_robot_links(ax, env: Robot2DEnvironmentObstacles, q: np.ndarray, *, alpha: float, color: str = "tab:blue") -> None:
    pts = env.forward_kinematics_all_joints(np.asarray(q, dtype=np.float32))
    ax.plot(pts[:, 0], pts[:, 1], "-", color=color, alpha=float(alpha), linewidth=2.0)
    ax.plot(pts[:, 0], pts[:, 1], "o", color=color, alpha=float(alpha), markersize=2.0)


def _break_on_angle_wrap(q_traj: np.ndarray, *, jump: float = np.pi) -> np.ndarray:
    """Insert NaNs into q_traj to break plotted lines on angle wrapping.

    The environment wraps joint angles to [-pi, pi] each step. When a joint crosses
    the boundary, the recorded trajectory can jump by ~2*pi; plotting it as a
    continuous line draws a long straight segment across the whole figure.
    """

    q = np.asarray(q_traj, dtype=np.float32)
    if q.ndim != 2 or q.shape[0] < 2:
        return q

    diffs = np.abs(np.diff(q, axis=0))
    breaks = np.any(diffs > float(jump), axis=1)
    if not np.any(breaks):
        return q

    rows: list[np.ndarray] = []
    nan_row = np.full((q.shape[1],), np.nan, dtype=np.float32)
    for i in range(q.shape[0] - 1):
        rows.append(q[i])
        if bool(breaks[i]):
            rows.append(nan_row)
    rows.append(q[-1])
    return np.vstack(rows)


def _ik_2link_solutions(target_xy: np.ndarray, *, l1: float, l2: float) -> list[np.ndarray]:
    """Analytic IK for a planar 2-link arm.

    Returns up to two solutions [q1, q2] in [-pi, pi], or empty if unreachable.
    """

    t = np.asarray(target_xy, dtype=np.float32).reshape(2)
    x = float(t[0])
    y = float(t[1])
    r2 = x * x + y * y
    denom = 2.0 * float(l1) * float(l2)
    if denom <= 1e-9:
        return []

    c2 = (r2 - float(l1) * float(l1) - float(l2) * float(l2)) / denom
    if c2 < -1.0 - 1e-6 or c2 > 1.0 + 1e-6:
        return []
    c2 = float(np.clip(c2, -1.0, 1.0))
    s2_abs = float(np.sqrt(max(0.0, 1.0 - c2 * c2)))

    sols: list[np.ndarray] = []
    for s2 in (+s2_abs, -s2_abs):
        q2 = float(np.arctan2(s2, c2))
        k1 = float(l1) + float(l2) * float(c2)
        k2 = float(l2) * float(s2)
        q1 = float(np.arctan2(y, x) - np.arctan2(k2, k1))
        q = np.asarray([q1, q2], dtype=np.float32)
        q = (q + np.pi) % (2.0 * np.pi) - np.pi
        sols.append(q)
    return sols


def _wrap_angles(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    return (q + np.pi) % (2.0 * np.pi) - np.pi


def _angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute wrapped difference a-b into [-pi, pi]."""

    return _wrap_angles(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32))


def _choose_goal_ik_by_cdf_level(
    *,
    init_q: np.ndarray,
    ik_solutions: list[np.ndarray],
    cdf,
    obj_lists,
    method: str,
    level0: float,
) -> np.ndarray:
    """Pick an IK solution whose CDF value best matches the start contour.

    This makes "沿起点等高线" more likely to approach the goal neighborhood.
    Tie-breaker: prefer smaller wrapped joint displacement from init_q.
    """

    import torch

    if not ik_solutions:
        raise ValueError("No IK solutions for the goal; target may be unreachable")

    init_q = np.asarray(init_q, dtype=np.float32).reshape(2)
    best = ik_solutions[0]
    best_key = None
    for qg in ik_solutions:
        q_t = torch.tensor(np.asarray(qg, dtype=np.float32).reshape(1, 2), dtype=torch.float32, device=torch.device("cpu"))
        d = cdf.calculate_cdf(q_t, obj_lists, method=str(method), return_grad=False)
        dg = float(d.detach().cpu().numpy().reshape(()))

        dq = _angle_diff(qg, init_q)
        dist2 = float(np.dot(dq, dq))
        key = (abs(dg - float(level0)), dist2)
        if (best_key is None) or (key < best_key):
            best_key = key
            best = qg
    return np.asarray(best, dtype=np.float32).reshape(2)


def _choose_goal_ik_nearest(init_q: np.ndarray, ik_solutions: list[np.ndarray]) -> np.ndarray:
    """Pick IK solution with smallest wrapped joint displacement from init."""

    if not ik_solutions:
        raise ValueError("No IK solutions for the goal; target may be unreachable")
    init_q = np.asarray(init_q, dtype=np.float32).reshape(2)
    best = ik_solutions[0]
    best_cost = None
    for qg in ik_solutions:
        dq = _angle_diff(np.asarray(qg, dtype=np.float32), init_q)
        cost = float(np.dot(dq, dq))
        if (best_cost is None) or (cost < best_cost):
            best_cost = cost
            best = qg
    return np.asarray(best, dtype=np.float32).reshape(2)


def _eval_traj(env: Robot2DEnvironmentObstacles, q_traj: np.ndarray) -> dict:
    q_traj = np.asarray(q_traj, dtype=np.float32)
    eef_traj = np.asarray([env.forward_kinematics_eef(q) for q in q_traj], dtype=np.float32)
    clear: list[float] = []
    col: list[bool] = []
    for q in q_traj:
        c, hit = env.min_obstacle_clearance(q)
        clear.append(float(c))
        col.append(bool(hit))
    return {
        "q_traj": q_traj,
        "eef_traj": eef_traj,
        "clearance": np.asarray(clear, dtype=np.float32),
        "collided": np.asarray(col, dtype=bool),
    }


def _traj_minimum_jerk_spline(q0: np.ndarray, q1: np.ndarray, *, steps: int) -> np.ndarray:
    """"Spline" interpolation using minimum-jerk (quintic) time-scaling."""

    q0 = np.asarray(q0, dtype=np.float32).reshape(2)
    q1 = np.asarray(q1, dtype=np.float32).reshape(2)
    steps = int(max(2, steps))
    s = np.linspace(0.0, 1.0, steps, dtype=np.float32)
    w = (10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5).astype(np.float32)
    dq = _angle_diff(q1, q0)
    q = q0.reshape(1, 2) + w.reshape(-1, 1) * dq.reshape(1, 2)
    return _wrap_angles(q)


def _traj_follow_start_contour(
    *,
    cdf,
    obj_lists,
    q0: np.ndarray,
    q_goal: np.ndarray,
    env: Robot2DEnvironmentObstacles,
    steps: int,
    method: str,
    level_gain: float,
    level_tol: float,
    tangent_gain: float,
    normal_gain: float,
    step_size: float,
    goal_task_tol: float,
    phase1_frac: float,
):
    """Two-phase motion:

    Phase 1: move along CDF gradient to reach the goal's CDF level.
    Phase 2: slide clockwise along that level-set, stop when near goal.
    """

    import torch
    import torch.nn.functional as F

    q = np.asarray(q0, dtype=np.float32).reshape(2)
    q_goal = np.asarray(q_goal, dtype=np.float32).reshape(2)

    level_gain = float(level_gain)
    level_tol = float(max(1e-6, level_tol))
    tangent_gain = float(tangent_gain)
    normal_gain = float(normal_gain)
    step_size = float(step_size)
    goal_task_tol = float(max(0.0, goal_task_tol))

    phase1_max = int(max(1, min(int(steps), int(round(float(steps) * float(np.clip(phase1_frac, 0.0, 1.0)))))))

    # Compute start contour level.
    q_t0 = torch.tensor(q.reshape(1, 2), dtype=torch.float32, device=torch.device("cpu"), requires_grad=True)
    d0, _ = cdf.calculate_cdf(q_t0, obj_lists, method=str(method), return_grad=True)
    level0 = float(d0.detach().cpu().numpy().reshape(()))

    # Compute goal contour level (target level to match).
    q_tg = torch.tensor(q_goal.reshape(1, 2), dtype=torch.float32, device=torch.device("cpu"), requires_grad=False)
    dg = cdf.calculate_cdf(q_tg, obj_lists, method=str(method), return_grad=False)
    level_goal = float(dg.detach().cpu().numpy().reshape(()))

    level_tgt = float(level_goal)

    out: list[np.ndarray] = [q.copy()]
    phase = 1
    for k in range(int(max(1, steps - 1))):
        q_t = torch.tensor(q.reshape(1, 2), dtype=torch.float32, device=torch.device("cpu"), requires_grad=True)
        d, g = cdf.calculate_cdf(q_t, obj_lists, method=str(method), return_grad=True)
        d_cur = float(d.detach().cpu().numpy().reshape(()))

        g = F.normalize(g, dim=-1)
        n = g.detach().cpu().numpy().astype(np.float32).reshape(2)

        # If gradient is degenerate, stop.
        nn = float(np.linalg.norm(n))
        if (not np.isfinite(nn)) or nn < 1e-8:
            break
        n = (n / np.float32(nn)).astype(np.float32)

        level_err = float(d_cur - level_tgt)

        # Switch to phase 2 once we match the goal's contour level (or we ran out of phase1 budget).
        if phase == 1:
            if (abs(level_err) <= level_tol) or (k >= int(phase1_max)):
                phase = 2

        if phase == 1:
            # Move across level sets to match the goal's level.
            # Gradient descent on 0.5*(d - d_goal)^2 => qdot = -k*(d - d_goal)*grad(d)
            qdot = (-level_gain * np.float32(level_err) * n).astype(np.float32)
        else:
            # Slide clockwise along the matched level-set, with normal correction to stay on it.
            # NOTE: In (q1,q2) plotting coordinates, the visually observed clockwise direction
            # corresponds to +rotate(n). If this looks reversed, flip the sign here.
            t_hat = np.asarray([n[1], -n[0]], dtype=np.float32)  # clockwise (plot convention)
            nt = float(np.linalg.norm(t_hat))
            if (not np.isfinite(nt)) or nt < 1e-8:
                t_hat = np.zeros((2,), dtype=np.float32)
            else:
                t_hat = (t_hat / np.float32(nt)).astype(np.float32)
            qdot = (tangent_gain * t_hat - normal_gain * np.float32(level_err) * n).astype(np.float32)

        q = _wrap_angles(q + np.float32(step_size) * qdot)
        out.append(q.copy())

        if phase == 2 and goal_task_tol > 0.0:
            eef = env.forward_kinematics_eef(q)
            if float(np.linalg.norm(np.asarray(eef, dtype=np.float32) - np.asarray(env.target_pos, dtype=np.float32))) <= goal_task_tol:
                break
    return np.asarray(out, dtype=np.float32), float(level0), float(level_goal)


def _simulate_planned_trajectories(
    *,
    env: Robot2DEnvironmentObstacles,
    init_q: np.ndarray,
    steps: int,
    cdf_method: str,
    cdf_obstacle_inflate: float,
    cdf_tangent_gain: float,
    cdf_normal_gain: float,
    contour_step: float,
    level_gain: float,
    level_tol: float,
    goal_task_tol: float,
    phase1_frac: float,
    goal_ik_branch: int,
) -> tuple[dict, dict, float]:
    """Return (traj_spline, traj_contour, start_contour_level)."""

    # Resolve goal in configuration space via analytic IK.
    if int(env.n) != 2:
        raise ValueError("This script currently supports only 2-DoF Robot2D (n=2)")
    sols = _ik_2link_solutions(np.asarray(env.target_pos, dtype=np.float32), l1=float(env.link_lengths[0]), l2=float(env.link_lengths[1]))

    # CDF trajectory: follow the start contour toward q_goal.
    try:
        import torch
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ModuleNotFoundError("PyTorch is required for CDF contour trajectory") from e

    from algorithms.cdf_rl_mppi.cdf_2d.cdf import CDF2D
    from algorithms.cdf_rl_mppi.cdf_2d.primitives2D_torch import Circle

    device = torch.device("cpu")
    cdf = CDF2D(device)

    inflate = float(env.safety_distance) + float(cdf_obstacle_inflate)
    obj_lists = [
        Circle(
            center=torch.tensor([float(o.x), float(o.y)], dtype=torch.float32, device=device),
            radius=float(o.r) + inflate,
            device=device,
        )
        for o in env.obstacles
    ]

    # Determine start contour level for IK selection (cheap single query).
    q_t0 = torch.tensor(np.asarray(init_q, dtype=np.float32).reshape(1, 2), dtype=torch.float32, device=device)
    d0 = cdf.calculate_cdf(q_t0, obj_lists, method=str(cdf_method), return_grad=False)
    level0 = float(d0.detach().cpu().numpy().reshape(()))

    # Pick which IK goal we use.
    branch = int(goal_ik_branch)
    if branch in (0, 1) and len(sols) > branch:
        q_goal = np.asarray(sols[branch], dtype=np.float32).reshape(2)
    else:
        q_goal = _choose_goal_ik_nearest(init_q, sols)

    # Non-CDF trajectory: spline interpolation in joint space.
    q_spline = _traj_minimum_jerk_spline(init_q, q_goal, steps=int(steps))
    traj_spline = _eval_traj(env, q_spline)

    q_contour, level0_out, level_goal = _traj_follow_start_contour(
        cdf=cdf,
        obj_lists=obj_lists,
        q0=init_q,
        q_goal=q_goal,
        env=env,
        steps=int(steps),
        method=str(cdf_method),
        level_gain=float(level_gain),
        level_tol=float(level_tol),
        tangent_gain=float(cdf_tangent_gain),
        normal_gain=float(cdf_normal_gain),
        step_size=float(contour_step),
        goal_task_tol=float(goal_task_tol),
        phase1_frac=float(phase1_frac),
    )
    traj_contour = _eval_traj(env, q_contour)

    # For plotting: return the goal contour level we are targeting.
    return traj_spline, traj_contour, float(level_goal)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot Robot2D CDF contours and planned trajectories")

    p.add_argument("--obstacles", type=str, default="0,2.5,0.20", help="Obstacles as 'x,y,r;x,y,r'")
    #p.add_argument("--obstacles", type=str, default="0,2.5,0.20;-2,0.5,0.20;3,2.5,0.20", help="Obstacles as 'x,y,r;x,y,r'")
    p.add_argument("--target_x", type=float, default=-2.8)
    p.add_argument("--target_y", type=float, default=1.8)

    p.add_argument("--init_q", type=str, default="0.25,0.10", help="Initial q as 'q1,q2'")
    p.add_argument("--steps", type=int, default=350)

    p.add_argument("--cdf_nb", type=int, default=120, help="CDF contour grid resolution per axis")
    p.add_argument("--cdf_method", type=str, default="offline_grid", help="offline_grid (default) or online_computation")
    p.add_argument("--cdf_normal_gain", type=float, default=1.2, help="Normal correction gain to stay on the matched contour")
    p.add_argument("--cdf_tangent_gain", type=float, default=0.4, help="Tangent gain to slide clockwise on the contour")
    p.add_argument("--cdf_obstacle_inflate", type=float, default=0.03, help="Extra obstacle radius inflation for CDF (in addition to env.safety_distance)")
    p.add_argument("--contour_step", type=float, default=0.12, help="Step size in joint space for contour-following integration")
    p.add_argument("--level_gain", type=float, default=1.6, help="Gain for moving across contours to match the goal level")
    p.add_argument("--level_tol", type=float, default=0.03, help="Tolerance for matching the goal contour level")
    p.add_argument("--goal_task_tol", type=float, default=0.25, help="Stop when within this task-space distance to goal (0 disables)")
    p.add_argument("--phase1_frac", type=float, default=0.45, help="Max fraction of steps used for phase-1 (level matching)")
    p.add_argument("--goal_ik_branch", type=int, default=-1, help="Choose IK branch: 0 or 1. Default (-1) picks nearest to init")

    p.add_argument("--draw_links", type=int, default=1)
    p.add_argument("--num_link_frames", type=int, default=14)

    p.add_argument("--plot_path", type=str, default=os.path.join(_RESULTS_DIR, "robot2d_cdf_contours_collision.png"))
    p.add_argument(
        "--export_npz",
        type=str,
        default="",
        help="Optional: export plot data to .npz for MATLAB re-plotting (path or filename under experiments/results)",
    )
    p.add_argument(
        "--export_mat",
        type=str,
        default="",
        help="Optional: export plot data to MATLAB-readable .mat (scipy) or .h5 (h5py). Provide a path ending with .mat or .h5",
    )
    p.add_argument("--show_plot", action="store_true")

    args = p.parse_args()

    if plt is None:
        raise ModuleNotFoundError("matplotlib is required for plotting")

    obstacles = _parse_obstacles(str(args.obstacles))

    # Env for task-space rollout (numpy implementation).
    env = Robot2DEnvironmentObstacles(
        link_lengths=[2.0, 2.0],
        target_pos=[float(args.target_x), float(args.target_y)],
        max_steps=int(args.steps),
        obstacles=obstacles,
        obstacle_margin=0.25,
        obstacle_penalty=250.0,
        terminate_on_collision=False,  # keep going to observe collision behavior
        collision_check="arm",
    )

    # Match CDF obstacle modeling to env collision buffer.
    env.cdf_obstacle_inflate = float(args.cdf_obstacle_inflate)

    init_q = np.asarray([float(x.strip()) for x in str(args.init_q).split(",")], dtype=np.float32)
    if init_q.shape[0] != env.n:
        raise SystemExit(f"init_q must have {env.n} values")

    traj, traj_recovery, start_level = _simulate_planned_trajectories(
        env=env,
        init_q=init_q,
        steps=int(args.steps),
        cdf_method=str(args.cdf_method),
        cdf_obstacle_inflate=float(args.cdf_obstacle_inflate),
        cdf_tangent_gain=float(args.cdf_tangent_gain),
        cdf_normal_gain=float(args.cdf_normal_gain),
        contour_step=float(args.contour_step),
        level_gain=float(args.level_gain),
        level_tol=float(args.level_tol),
        goal_task_tol=float(args.goal_task_tol),
        phase1_frac=float(args.phase1_frac),
        goal_ik_branch=int(args.goal_ik_branch),
    )

    if traj.get("clearance") is not None and traj["clearance"].size:
        q_end = np.asarray(traj["q_traj"][-1], dtype=np.float32)
        e_end = np.asarray(traj["eef_traj"][-1], dtype=np.float32)
        # Compare task distance to the *workspace* goal.
        task_d = float(np.linalg.norm(e_end - np.asarray(env.target_pos, dtype=np.float32)))
        print(
            f"Spline-only: min_clearance={float(np.min(traj['clearance'])):.4f}, "
            f"collisions={int(np.sum(traj['collided']))}/{int(traj['collided'].size)}"
        )
        print(f"  spline final task_dist={task_d:.4f}")
    if traj_recovery.get("clearance") is not None and traj_recovery["clearance"].size:
        q2_end = np.asarray(traj_recovery["q_traj"][-1], dtype=np.float32)
        e2_end = np.asarray(traj_recovery["eef_traj"][-1], dtype=np.float32)
        task_d2 = float(np.linalg.norm(e2_end - np.asarray(env.target_pos, dtype=np.float32)))
        print(
            f"Two-phase(CDF): goal_level={float(start_level):.4f}, "
            f"min_clearance={float(np.min(traj_recovery['clearance'])):.4f}, "
            f"collisions={int(np.sum(traj_recovery['collided']))}/{int(traj_recovery['collided'].size)}"
        )
        print(f"  two-phase final task_dist={task_d2:.4f}")

    # CDF contours in configuration space (torch implementation).
    try:
        import torch
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ModuleNotFoundError("PyTorch is required for CDF plotting") from e

    from algorithms.cdf_rl_mppi.cdf_2d.cdf import CDF2D
    from algorithms.cdf_rl_mppi.cdf_2d.primitives2D_torch import Circle

    device = torch.device("cpu")
    cdf = CDF2D(device)

    # Build torch obstacles.
    inflate = float(env.safety_distance) + float(args.cdf_obstacle_inflate)
    obj_lists = [
        Circle(
            center=torch.tensor([float(o.x), float(o.y)], dtype=torch.float32, device=device),
            radius=float(o.r) + inflate,
            device=device,
        )
        for o in obstacles
    ]

    nb = int(max(20, args.cdf_nb))
    # Build a custom grid to contour at higher resolution than cdf.nbData.
    q_lin = np.linspace(-np.pi, np.pi, nb, dtype=np.float32)
    q0, q1 = np.meshgrid(q_lin, q_lin)
    Q = np.stack([q0.reshape(-1), q1.reshape(-1)], axis=-1)
    Q_t = torch.tensor(Q, dtype=torch.float32, device=device)

    with torch.no_grad():
        d = cdf.calculate_cdf(Q_t, obj_lists, method=str(args.cdf_method), return_grad=False)
    d_np = d.detach().cpu().numpy().reshape(nb, nb)

    # Plot: left = C-space contours, right = task-space trajectory.
    fig, (ax_c, ax_t) = plt.subplots(1, 2, figsize=(14, 6.5))

    # C-space contours
    ax_c.set_title("Configuration space CDF (q1-q2) + trajectory")
    ax_c.set_xlabel("q1")
    ax_c.set_ylabel("q2")
    ax_c.set_aspect("equal", adjustable="box")
    ax_c.set_xlim(-np.pi, np.pi)
    ax_c.set_ylim(-np.pi, np.pi)
    ax_c.grid(True, alpha=0.25)

    levels = 18
    ct = ax_c.contourf(q0, q1, d_np, levels=levels, cmap="coolwarm", alpha=0.95)
    ax_c.contour(q0, q1, d_np, levels=[0.0], colors="black", linewidths=2.0)
    # Target contour level (same as goal IK's cdf level).
    ax_c.contour(q0, q1, d_np, levels=[float(start_level)], colors="tab:green", linewidths=1.5, linestyles="--")
    fig.colorbar(ct, ax=ax_c, fraction=0.046, pad=0.04, label="cdf(q)")

    q_traj = traj["q_traj"]
    q_traj_line = _break_on_angle_wrap(q_traj)
    ax_c.plot(q_traj_line[:, 0], q_traj_line[:, 1], "k-", lw=2.0, label="Spline (no CDF)")
    ax_c.plot(q_traj[0, 0], q_traj[0, 1], "go", markersize=7, label="start")

    # Mark goal (in configuration space) via analytic IK (2-link only).
    if int(env.n) == 2:
        sols = _ik_2link_solutions(
            np.asarray(env.target_pos, dtype=np.float32),
            l1=float(env.link_lengths[0]),
            l2=float(env.link_lengths[1]),
        )
        for k, qg in enumerate(sols):
            ax_c.plot(qg[0], qg[1], "mx", markersize=10, mew=2.0, label=("goal (IK)" if k == 0 else None))

    collided = traj["collided"]
    if collided.size:
        hit_idx = np.where(collided)[0]
        if hit_idx.size:
            ax_c.plot(q_traj[hit_idx, 0], q_traj[hit_idx, 1], "r.", markersize=6, label="collision")

    q2 = traj_recovery["q_traj"]
    q2_line = _break_on_angle_wrap(q2)
    ax_c.plot(q2_line[:, 0], q2_line[:, 1], color="tab:green", lw=2.2, label="Level match -> CW slide")
    ax_c.scatter(q2[:, 0], q2[:, 1], s=10, color="tab:green", alpha=0.20, label="CDF traj points")
    ax_c.plot(q2[-1, 0], q2[-1, 1], "o", color="tab:green", markersize=6, label="end")

    ax_c.legend(loc="lower left", bbox_to_anchor=(0.02, 0.02), framealpha=0.9)

    # Task-space view
    ax_t.set_title("Task space (EEF) + obstacles")
    ax_t.set_xlabel("x")
    ax_t.set_ylabel("y")
    ax_t.set_aspect("equal", adjustable="box")
    ax_t.grid(True, alpha=0.25)

    _draw_obstacles(ax_t, obstacles, alpha=0.35)
    eef_traj = traj["eef_traj"]
    ax_t.plot(eef_traj[:, 0], eef_traj[:, 1], color="tab:blue", lw=2.0, label="EEF (spline)")
    ax_t.plot(eef_traj[0, 0], eef_traj[0, 1], "gs", markersize=7, label="start")
    ax_t.plot(float(env.target_pos[0]), float(env.target_pos[1]), "rx", markersize=12, label="goal")

    e2 = traj_recovery["eef_traj"]
    ax_t.plot(e2[:, 0], e2[:, 1], color="tab:green", lw=2.2, label="EEF (two-phase contour)")
    col2 = traj_recovery.get("collided", None)
    if col2 is not None and col2.size:
        hit2 = np.where(col2)[0]
        if hit2.size:
            ax_t.plot(e2[hit2, 0], e2[hit2, 1], "r.", markersize=6, label="collision (contour)")

    # Link snapshots
    if bool(int(args.draw_links)):
        k = int(max(2, int(args.num_link_frames)))
        idx = np.linspace(0, q_traj.shape[0] - 1, num=k, dtype=int)
        for j, t in enumerate(idx):
            alpha = 0.08 + 0.55 * (j / max(1, (len(idx) - 1)))
            _draw_robot_links(ax_t, env, q_traj[int(t)], alpha=float(alpha), color="tab:blue")
        q2 = traj_recovery["q_traj"]
        idx2 = np.linspace(0, q2.shape[0] - 1, num=k, dtype=int)
        for j, t in enumerate(idx2):
            alpha = 0.08 + 0.55 * (j / max(1, (len(idx2) - 1)))
            _draw_robot_links(ax_t, env, q2[int(t)], alpha=float(alpha), color="tab:green")

    reach = float(np.sum(env.link_lengths))
    ax_t.set_xlim(-reach - 0.6, reach + 0.6)
    ax_t.set_ylim(-reach - 0.6, reach + 0.6)
    ax_t.legend(loc="lower left", bbox_to_anchor=(0.02, 0.02), framealpha=0.9)

    fig.tight_layout()
    plot_path = _resolve_plot_path(str(args.plot_path))
    fig.savefig(plot_path, dpi=150)
    print(f"Saved plot to {plot_path}")

    export_npz = str(getattr(args, "export_npz", "") or "").strip()
    export_mat = str(getattr(args, "export_mat", "") or "").strip()

    payload = None
    if export_npz:
        export_npz = _resolve_data_path(export_npz)
        if not export_npz.lower().endswith(".npz"):
            export_npz = export_npz + ".npz"

        # Save all numeric data needed to reproduce the plot in MATLAB.
        obs_arr = np.asarray([[float(o.x), float(o.y), float(o.r)] for o in obstacles], dtype=np.float32)
        payload = {
            # grid
            "q0": np.asarray(q0, dtype=np.float32),
            "q1": np.asarray(q1, dtype=np.float32),
            "cdf": np.asarray(d_np, dtype=np.float32),
            # meta
            "start_level": float(start_level),
            "target_pos": np.asarray(env.target_pos, dtype=np.float32).reshape(2),
            "init_q": np.asarray(init_q, dtype=np.float32).reshape(2),
            "obstacles": obs_arr,
            "link_lengths": np.asarray(getattr(env, "link_lengths", [2.0, 2.0]), dtype=np.float32).reshape(-1),
            # trajectories (raw)
            "q_spline": np.asarray(traj["q_traj"], dtype=np.float32),
            "eef_spline": np.asarray(traj["eef_traj"], dtype=np.float32),
            "clearance_spline": np.asarray(traj.get("clearance", []), dtype=np.float32),
            "collided_spline": np.asarray(traj.get("collided", []), dtype=np.int8),
            "q_contour": np.asarray(traj_recovery["q_traj"], dtype=np.float32),
            "eef_contour": np.asarray(traj_recovery["eef_traj"], dtype=np.float32),
            "clearance_contour": np.asarray(traj_recovery.get("clearance", []), dtype=np.float32),
            "collided_contour": np.asarray(traj_recovery.get("collided", []), dtype=np.int8),
            # trajectories (with NaN breaks for angle wrap)
            "q_spline_line": np.asarray(q_traj_line, dtype=np.float32),
            "q_contour_line": np.asarray(q2_line, dtype=np.float32),
        }

        np.savez(export_npz, **payload)
        print(f"[EXPORT] Saved plot data to {export_npz}")

    if export_mat:
        if payload is None:
            obs_arr = np.asarray([[float(o.x), float(o.y), float(o.r)] for o in obstacles], dtype=np.float32)
            payload = {
                "q0": np.asarray(q0, dtype=np.float32),
                "q1": np.asarray(q1, dtype=np.float32),
                "cdf": np.asarray(d_np, dtype=np.float32),
                "start_level": float(start_level),
                "target_pos": np.asarray(env.target_pos, dtype=np.float32).reshape(2),
                "init_q": np.asarray(init_q, dtype=np.float32).reshape(2),
                "obstacles": obs_arr,
                "link_lengths": np.asarray(getattr(env, "link_lengths", [2.0, 2.0]), dtype=np.float32).reshape(-1),
                "q_spline": np.asarray(traj["q_traj"], dtype=np.float32),
                "eef_spline": np.asarray(traj["eef_traj"], dtype=np.float32),
                "clearance_spline": np.asarray(traj.get("clearance", []), dtype=np.float32),
                "collided_spline": np.asarray(traj.get("collided", []), dtype=np.int8),
                "q_contour": np.asarray(traj_recovery["q_traj"], dtype=np.float32),
                "eef_contour": np.asarray(traj_recovery["eef_traj"], dtype=np.float32),
                "clearance_contour": np.asarray(traj_recovery.get("clearance", []), dtype=np.float32),
                "collided_contour": np.asarray(traj_recovery.get("collided", []), dtype=np.int8),
                "q_spline_line": np.asarray(q_traj_line, dtype=np.float32),
                "q_contour_line": np.asarray(q2_line, dtype=np.float32),
            }

        out_path = _export_matlab_file(export_mat, payload)
        print(f"[EXPORT] Saved MATLAB data to {out_path}")

    if bool(args.show_plot):
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
