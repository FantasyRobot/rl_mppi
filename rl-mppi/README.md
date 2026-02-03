# RL-MPPI (Ball 2D)

This folder contains a small **RL-guided MPPI** demo for the project’s 2D ball environment.

- RL prior: loads the SAC policy checkpoint from `sac/sac_ball/models/`.
- MPPI reference: follows the structure of `mppi/mppi_ball.py`.

## Run

From repo root:

```bash
python rl-mppi/test_rl_mppi_ball.py --model_path sac/sac_ball/models/sac_ball_model_online.pth --target_x 3 --target_y 3 --num_tests 10 --max_steps 2000
```

Outputs a plot to `rl-mppi/outputs/ball_trajectories_rl_mppi.png` by default.

## Notes

The controller implements MPPI path-integral weighting:

- weights: $w_i \propto \exp(-(S_i - S_{\min})/\lambda)$
- policy provides the nominal sequence $u_{nom}$ (proposal mean)
