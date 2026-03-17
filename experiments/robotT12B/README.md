# RobotT12B (t12a_14) Experiments

This folder contains **runnable scripts** for the MuJoCo arm task `t12a_14`, including MPPI / SAC / RL‑MPPI, collision handling (stop vs CDF shaping), and constrained SAC (TD‑CD / CD‑SAC).

## Dependencies

- Python: NumPy
- RL: PyTorch
- Simulation: `mujoco` (Python bindings)
- Plotting (Python): `matplotlib`
- MATLAB export: `scipy` (only needed when exporting `.mat` directly)

## Assets / Models / Outputs

- XML/Mesh assets: `urdf/` (e.g. `t12a_14_normal.xml`, `t12a_14_clear.xml`, `t12a_14_dyn.xml`)
- Default checkpoints: `models/`
- Default results/logs: `../results/` (i.e. `experiments/results/`)

## Shared parameter meanings (common across scripts)

- `--xml`: MJCF XML path. Many scripts `chdir` to the XML directory so relative mesh paths resolve.
- `--goal_site`: goal site name in XML (default: `goal`).
- `--eef_site`: end-effector site name in XML (default: `end_effector`).
- `--action_repeat`: hold the same control for `N` physics steps; effectively reduces control frequency and speeds up planning.
- `--max_steps` / `--steps`: max env steps (outer loop), not physics steps.
- `--reach_tol` / `--tol`: success threshold on distance `||eef - goal||`.
- `--seed`: RNG seed.

---

## 1) Visualize the robot model

### Script: [mujoco_visualize_t12a_14.py](mujoco_visualize_t12a_14.py)

Opens a passive MuJoCo viewer for an XML.

Key args:
- `--xml`: which XML to open (default: `urdf/t12a_14_simple.xml`).

Notes:
- Press `P` in the viewer to print camera parameters (useful for reproducing viewpoints in MJCF).

---

## 2) MPPI control (no learning)

### Script: [mppi_control_t12a_14.py](mppi_control_t12a_14.py)

Runs MPPI to drive the end-effector to the goal.

Key args:
- Common: `--xml --goal_site --eef_site --steps --tol --action_repeat`
- Viewer: `--no_viewer`, trajectory overlay control via `--no_draw_traj --traj_*`
- MPPI knobs: `--horizon --num_samples --lambda_coeff --noise_std --pos_cost --action_cost --smooth_cost`

---

## 3) RL‑MPPI control (SAC proposal + MPPI)

### Script: [rl_mppi_control_t12a_14.py](rl_mppi_control_t12a_14.py)

Runs RL‑MPPI where a trained SAC policy is used as a proposal / guide for MPPI.

Key args:
- Common: `--xml --goal_site --eef_site --steps --tol --action_repeat`
- `--sac_model`: SAC checkpoint path (`.pth`).
- MPPI/RL‑MPPI knobs: same as `mppi_control_t12a_14.py`.

---

## 4) SAC training / testing (includes CDF‑SAC)

### CLI: [sac_t12a_14_cli.py](sac_t12a_14_cli.py)

This is the **main entry point** for training/testing SAC on `t12a_14`.

### Train

Example (plain SAC, no collision handling):

```bash
python experiments/robotT12B/sac_t12a_14_cli.py train --collision_mode none
```

Example (CDF‑SAC: CDF shaping, do not terminate on collision):

```bash
python experiments/robotT12B/sac_t12a_14_cli.py train --collision_mode cdf
```

Key train args:
- `--save_path`: checkpoint `.pth`.
- `--total_steps`: total env steps.
- `--eval_every`: evaluation interval (env steps).
- `--collision_mode`: `none | stop | cdf`
  - `none`: ignore obstacles (uses the obstacle‑free env)
  - `stop`: collision ends episode
  - `cdf`: **CDF shaping** (safety reward shaping), collision does **not** terminate by default
- `--obstacle_prefix`: geom name prefix for obstacles in MJCF (default: `obstacle`).
- `--collision_penalty`: penalty when collision occurs.
- `--terminate_on_collision`: override termination (1/0). If not set: `stop->1`, `cdf/none->0`.
- CDF shaping knobs (only meaningful for `cdf`):
  - `--cdf_sigma`: smoothing parameter
  - `--cdf_margin`: distance margin
  - `--cdf_scale`: scaling factor
- `--mat_path`: optional `.mat` export path for MATLAB.
  - If omitted and `--collision_mode=cdf`, the trainer also writes a sibling `.mat` next to the `.npz` log (requires SciPy).

Outputs:
- Checkpoint: `*.pth`
- Training log: `*_train_log.npz`
- Training curves: `*_train.png`
- (Optional) MATLAB log: `*_train_log.mat`

### Test

Example (headless test + save minimal trajectories):

```bash
python experiments/robotT12B/sac_t12a_14_cli.py test --model_path experiments/robotT12B/models/sac_t12a_14_model.pth
```

Key test args:
- `--viewer`: run in MuJoCo viewer and draw trajectory
- `--traj_width`, `--traj_max_points`, `--traj_stride`, `--no_draw_traj`: trajectory overlay style

### Core training function

If you call training from other scripts, the implementation is in:
- [train_sac_t12a_14_online.py](train_sac_t12a_14_online.py) (`train_sac_t12a_14_online(...)`)

---

## 5) Compare collision modes: stop vs CDF

### Script: [compare_sac_collision_modes.py](compare_sac_collision_modes.py)

Runs two trainings back‑to‑back:
- `stop`: terminate on collision
- `cdf`: CDF shaping, no termination on collision

Key args:
- Shared: `--xml --goal_site --eef_site --action_repeat --max_steps --reach_tol`
- Train length: `--total_steps --eval_every --seed`
- Collision/CDF: `--collision_penalty --cdf_sigma --cdf_margin --cdf_scale`
- Outputs: `--out_dir --prefix --out_mat`

Outputs:
- `..._stop.pth` + `..._stop_train_log.npz` (+ png)
- `..._cdf.pth` + `..._cdf_train_log.npz` (+ png)
- Overlay plot: `*_overlay.png`
- MATLAB export: `*_overlay.mat` (or `--out_mat`)

MATLAB plotting:
- [matlab/plot_sac_collision_modes_mat.m](matlab/plot_sac_collision_modes_mat.m)

---

## 6) Compare MPPI vs SAC vs RL‑MPPI (single rollout)

### Script: [compare_t12a_14_methods.py](compare_t12a_14_methods.py)

Runs one rollout for each method (MPPI / SAC / RL‑MPPI) and saves trajectories + joint signals to `.npz`.

Key args:
- `--steps`: rollout length
- `--tol`: success tolerance
- `--sac_model`: SAC checkpoint used by SAC and RL‑MPPI
- `--save_npz`: output file

### Script: [benchmark_t12a_14_methods.py](benchmark_t12a_14_methods.py)

Runs multiple trials with shared initial conditions and compares MPPI / SAC / RL‑MPPI on:
- planning time per control step
- goal tracking error (`final_dist`, `mean_dist`, `rms_dist`)
- joint impact indicators (`max_abs_qacc`, `rms_qacc`, `max_abs_qjerk`, `rms_qjerk`)

**Output format:**
- The console summary table displays metrics as `Mean ± Std` (e.g., `Plan(ms): 21.45 ± 3.12`).
- `Std` represents the standard deviation across all trials (inter-trial variability).
- If `num_trials=1`, the standard deviation will be `0.00`.

Key args:
- `--num_trials --steps --action_repeat --tol --seed`
- `--start_margin` or `--init_qpos` for start state control
- MPPI knobs: `--horizon --num_samples --lambda_coeff --noise_std --pos_cost --action_cost --smooth_cost`
- Outputs: `--save_npz --save_json --save_csv`

### Plot the saved `.npz`

Script: [plot_t12a_14_compare_npz.py](plot_t12a_14_compare_npz.py)

Key args:
- `--npz`: input from `compare_t12a_14_methods.py`
- `--save_dir`: if set, writes PNGs
- `--max_joints`: limits plotted joint dimensions

---

## 7) Random-start success rate evaluation (MPPI / SAC / RL‑MPPI)

### Script: [eval_random_starts_success_rate.py](eval_random_starts_success_rate.py)

Evaluates success under **random collision‑free initial joint states**.
Success definition in this script:
- `success = reached_goal AND NOT collided_with_obstacle`

Key args:
- `--num_starts`: number of random starts
- `--start_margin`: avoid joint limits by fraction of ctrlrange
- `--max_resample`: max attempts to find collision‑free starts
- `--out`: output `.npz`
- `--save_json`: optional summary `.json`

Outputs:
- `.npz` with per‑trial results and summary stats

---

## 8) Constrained SAC (TD‑CD / CD‑SAC)

Everything for constrained training/testing is under:
- `cd_sac_t12a_14/`

### CLI: [cd_sac_t12a_14/cd_sac_t12a_14_cli.py](cd_sac_t12a_14/cd_sac_t12a_14_cli.py)

#### Train

```bash
python experiments/robotT12B/cd_sac_t12a_14/cd_sac_t12a_14_cli.py train
```

Constraint parameters:
- `--vel_bound`: per‑joint velocity bound $|qvel_i| \le \text{vel_bound}$
- `--acc_bound`: per‑joint acceleration bound $|qacc_i| \le \text{acc_bound}$ (finite diff)
- `--violation_agg`: aggregation across joints/action_repeat (`max` paper default)
- `--violation_tol`: tolerance before counting violation (`1e-3` paper default)

TD‑CD parameters:
- `--constraint_discount_use_amount`: use continuous violation amount (1) vs binary (0)
- `--tdcd_p_max`: Eq.(7) $p_{max}$
- `--tdcd_tau_c`: Eq.(8) EMA factor for $c_{max}$

Outputs:
- Checkpoints: `*_best.pth`, `*_last.pth` (and main `save_path`)
- Logs: `*_train_log.npz`, `*_train_log.mat`
- Curves: `*_train.png`

MATLAB plotting:
- [cd_sac_t12a_14/plot_cd_sac_train_mat.m](cd_sac_t12a_14/plot_cd_sac_train_mat.m)

#### Test

```bash
python experiments/robotT12B/cd_sac_t12a_14/cd_sac_t12a_14_cli.py test --viewer
```

Extra test args:
- `--export_joint_csv <base>`: exports per-step joint `qvel/qacc` CSVs + meta txt
- `--settle_steps`, `--vel_tol`: optional post‑success settling to measure residual motion

MATLAB plotting for joint CSV signals:
- [matlab/plot_cd_sac_t12a_14_joint_signals.m](matlab/plot_cd_sac_t12a_14_joint_signals.m)

---

## 9) MATLAB: plot SAC training curves exported as `.mat`

### Script: [plot_sac_train_mat.m](plot_sac_train_mat.m)

Plots a 2×2 figure from a SAC/CDF‑SAC exported `.mat` training log:
- Episode return
- Eval mean dist (±std) + eval mean steps
- Eval success/collision rates
- Alpha

Usage:

```matlab
plot_sac_train_mat('path/to/sac_t12a_14_model_train_log.mat');
```

---

## Tips / gotchas

- If meshes fail to load, use XMLs under `urdf/` and keep relative paths intact.
- `.mat` export requires `scipy`. If you prefer not to install SciPy, you can convert an existing `.npz` using [tools/npz_to_mat.py](../../tools/npz_to_mat.py).
