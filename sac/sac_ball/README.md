# sac_ball

This folder contains a small SAC-on-Ball demo.

## Single entrypoint
Use [sac_ball_cli.py](sac_ball_cli.py) as the unified runner.

- One-shot (train then test): `python sac_ball_cli.py`

- Train (online interaction): `python sac_ball_cli.py train --total_steps 200000 --save_path models/sac_ball_model_online.pth`
- Test: `python sac_ball_cli.py test --model_path models/sac_ball_model_online.pth --num_tests 10 --max_steps 2000`
- Test near target (recommended for quick sanity check): `python sac_ball_cli.py test_near --model_path models/sac_ball_model_online.pth --num_tests 10 --max_steps 2000`

Notes:
- Testing saves a trajectory plot to `ball_trajectories.png` by default.
- To show the plot interactively, add `--show_plot`.
