# cd_sac_ball (TD-CD constraints)

This folder contains the TD-CD-MPPI paper PDF and a SAC demo that handles constraints via
**TD-CD stochastic-termination discounting (Eq.6–9)**.

## SAC with acceleration + velocity constraints

Entry point: [cd_sac_ball_cli.py](cd_sac_ball_cli.py)

### Train

```bash
cd experiments/cd_sac_ball
python cd_sac_ball_cli.py train \
  --total_steps 200000 \
  --vel_bound 2.0 \
  --acc_bound 0.5 \
  --constraint_discount_use_amount 0 \
  --tdcd_p_max 1.0 \
  --tdcd_tau_c 0.99
```

### Test

```bash
cd experiments/cd_sac_ball
python cd_sac_ball_cli.py test_near --num_tests 10
```

Plots are saved under `experiments/results/` by default.

## Implementation notes

- Environment: `env/envball_constraints.py` (`BallEnvironmentConstraints`)
  - **Acceleration constraint (per-component)**: `|ax|<=acc_bound` and `|ay|<=acc_bound`.
    - SAC outputs `u in [-1,1]^2`, environment applies `ax=acc_bound*u_x`, `ay=acc_bound*u_y`.
  - **Velocity constraint (per-component)**: `|vx|<=vel_bound` and `|vy|<=vel_bound`.
    - Environment computes raw next velocity `(vx_raw, vy_raw)` and reports
      - `constraint_violation = (|vx_raw|>vel_bound) or (|vy_raw|>vel_bound)`
      - `vel_violation_amount = max(0,|vx_raw|-vel_bound) + max(0,|vy_raw|-vel_bound)`
    - State is kept bounded by clipping `(vx,vy)` to `[-vel_bound, vel_bound]` per component.
- Training: [train_cd_sac_ball_online.py](train_cd_sac_ball_online.py)
  - Time-limit truncation is handled as non-terminal for critic targets.
  - **TD-CD (Eq.6–9) implementation**:
    - Compute a soft termination signal $\delta_t\in[0,1]$ from a normalized constraint signal.
    - Use per-transition discount $\gamma_t = \gamma\,(1-\delta_t)$ in the TD backup.
    - `constraint_discount_use_amount=0`: $c_t\in\{0,1\}$ from `constraint_violation`.
    - `constraint_discount_use_amount=1`: continuous $c_t$ from `vel_violation_amount`.
    - `tdcd_p_max` corresponds to Eq.(7) $p^{max}$; `tdcd_tau_c` corresponds to Eq.(8) $\tau_c$.

For a paper-ready explanation (Eq.6–9 → TD learning), see: `paper_tdcd_constraints.md`.

For a cleaner TD‑CD‑SAC principle + equations + advantages write-up (aligned to this repo’s implementation), see:
`TDCD_SAC_Principle.md`.
