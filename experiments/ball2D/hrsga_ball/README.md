# hrsga_ball

该目录用于验证 HRSGA 在二维多球任务分配与避障场景中的训练、测试和对比流程，当前已整理为三条主线：

- HRSGA 监督学习主线。
- TD3-HRSGA 强化学习主线。
- Standard GNN 基线主线，用于和 HRSGA 做同环境对比。

当前组织原则：

- 根目录只保留环境、模型定义和稳定产物入口。
- `scripts/` 按训练、评估、对比、绘图拆分。
- `runs/` 保存原始运行产物，`runs/benchmarks/` 保存聚合统计与程序生成图。
- `docs/paper/` 只保留论文正文和需要稳定引用的论文配图快照。

## 目录说明

### 根目录

- `envball_hrsga.py`
  - 环境主体实现与统一导入入口。

- `hrsga_ball_model.py`
  - HRSGA 模型、专家控制器、数据集构造、评估函数与 TD3 复用组件。

- `standard_gnn_ball_model.py`
  - Standard GNN 基线模型定义。

### `scripts/train/`

- `train_hrsga_ball.py`
  - HRSGA 监督训练入口。

- `train_standard_gnn_ball.py`
  - Standard GNN 监督训练入口。

- `train_td3_hrsga_ball.py`
  - TD3-HRSGA 强化学习训练入口。

- `train_ablation_hrsga_ball.py`
  - HRSGA 消融训练入口，支持 `no_temporal`、`dense`、`unified_relation`、`no_geometric`。

- `train_hrsga_ball.example.json`
  - HRSGA 正式训练配置示例。

- `train_standard_gnn_ball.example.json`
  - Standard GNN 正式训练配置示例。

### `scripts/eval/`

- `test_hrsga_ball.py`
  - HRSGA 监督模型测试入口。
  - 默认使用代表性起始位置/目标位置集合；可用 `--layout_mode representative|structured|random` 切换。

- `test_td3_hrsga_ball.py`
  - TD3-HRSGA 测试入口。

- `benchmark_policy_ball.py`
  - 通用正式评估脚本，可对 HRSGA、Standard GNN、TD3 进行多随机种子统计。

### `scripts/compare/`

- `compare_trained_gnn_hrsga_ball.py`
  - 使用相同随机种子，对训练后的 Standard GNN 和 HRSGA 进行轨迹对比。

- `compare_policy_metrics_ball.py`
  - 对两个训练后模型做统一随机种子的聚合指标对比。

- `compare_ablations_ball.py`
  - 对多个已训练 HRSGA 消融版本做统一随机种子的聚合指标对比。

### `scripts/plot/`

- `plot_training_curves_ball.py`
  - 从 `metrics.jsonl` 绘制训练损失和评估曲线。

### `docs/paper/`

- `draft.md`
  - 论文草稿正文。

- `experiment_plan.md`
  - 论文实验计划与执行清单。

- `hrsga_formal_training_curves.png`
  - 训练曲线图。

- `compare_hrsga_vs_standard_gnn_representative.png`
  - 正式主结果对比图。

- `hrsga_ablations.png`
  - 消融实验对比图。

### `docs/references/`

- 参考论文、广告页和外部资料归档。

### 结果与产物目录

- `models/`
  - 存放当前可直接测试的权重文件。
  - 现保留：
    - `hrsga_ball_best.pt`
    - `hrsga_ball_latest.pt`
    - `standard_gnn_ball_best.pt`
    - `standard_gnn_ball_latest.pt`
    - `td3_hrsga_best.pt`
    - `td3_hrsga_latest.pt`

- `runs/`
  - 存放按运行名归档的训练配置、日志和 checkpoint。
  - 当前保留：
    - `hrsga_formal/`：当前正式 HRSGA 监督训练结果。
    - `standard_gnn_formal/`：当前正式 Standard GNN 监督训练结果。
    - `hrsga_no_temporal/`、`hrsga_dense/`、`hrsga_unified_relation/`、`hrsga_no_geometric/`：四组正式消融运行结果。

- `runs/benchmarks/`
  - 存放正式统计摘要、逐次运行记录和程序导出的图表。
  - 其中以 `smoke_` 开头的文件仅用于快速连通性验证，不作为论文结果引用。
  - 当前正式主结果包括：
    - `hrsga_representative_summary.json`
    - `hrsga_representative_runs.jsonl`
    - `compare_hrsga_vs_standard_gnn_representative.json`
    - `compare_hrsga_vs_standard_gnn_representative.png`
    - `hrsga_ablations.json`
    - `hrsga_ablations.png`
    - `hrsga_formal_training_curves.png`

## 按实验阶段查找入口

1. 训练 HRSGA 正式模型：`scripts/train/train_hrsga_ball.py`
2. 训练 Standard GNN 基线：`scripts/train/train_standard_gnn_ball.py`
3. 训练 TD3-HRSGA：`scripts/train/train_td3_hrsga_ball.py`
4. 运行单模型测试或生成轨迹图：`scripts/eval/test_hrsga_ball.py`、`scripts/eval/test_td3_hrsga_ball.py`
5. 生成正式统计结果：`scripts/eval/benchmark_policy_ball.py`
6. 比较 HRSGA 与 Standard GNN：`scripts/compare/compare_trained_gnn_hrsga_ball.py`、`scripts/compare/compare_policy_metrics_ball.py`
7. 比较 HRSGA 消融版本：`scripts/train/train_ablation_hrsga_ball.py`、`scripts/compare/compare_ablations_ball.py`
8. 绘制训练曲线：`scripts/plot/plot_training_curves_ball.py`
9. 查看论文与实验计划：`docs/paper/draft.md`、`docs/paper/experiment_plan.md`

当前 `scripts/train/train_hrsga_ball.example.json` 已默认启用 `"expert_max_collisions": 0`，表示构建 imitation 数据集时只保留无碰撞 expert episode；脚本会自动继续采样，直到凑够配置中要求的训练和验证 episode 数。

## 图表与结果归档规范

- 程序运行产生的原始结果统一写入 `runs/` 和 `runs/benchmarks/`。
- 论文正文默认引用 `docs/paper/` 下的稳定图片快照，避免后续重复运行覆盖论文插图。
- 当 `runs/benchmarks/` 中生成新的正式图表后，如需在论文中固定引用，再将对应 PNG 同步到 `docs/paper/`。
- `smoke_*.json`、`smoke_*.png`、`*_smoke.png` 不进入论文正文，也不覆盖 `docs/paper/` 中的正式快照。
- 新增 benchmark 文件时，优先沿用 `任务名_场景名_指标名` 或 `方法对比_场景名` 的命名方式，避免 `final`、`new` 之类不可追踪命名。

## Standard GNN 正式训练与对比

如果要重新生成 Standard GNN 的正式权重，可直接运行：

```powershell
cd d:\VirtualSpace\rl_mppi\experiments\ball2D\hrsga_ball
D:/python3.12/python.exe scripts/train/train_standard_gnn_ball.py --config_path scripts/train/train_standard_gnn_ball.example.json --run_name standard_gnn_formal
```

训练完成后，脚本会自动同步输出：

- `runs/standard_gnn_formal/`
- `models/standard_gnn_ball_best.pt`
- `models/standard_gnn_ball_latest.pt`

如需和当前 HRSGA 最优权重做轨迹对比，可运行：

```powershell
cd d:\VirtualSpace\rl_mppi\experiments\ball2D\hrsga_ball
D:/python3.12/python.exe scripts/compare/compare_trained_gnn_hrsga_ball.py --hrsga_model_path models/hrsga_ball_best.pt --gnn_model_path models/standard_gnn_ball_best.pt --num_tests 5 --plot_path runs/benchmarks/compare_trained_gnn_hrsga_representative.png --summary_path runs/benchmarks/compare_trained_gnn_hrsga_representative.json
```

当前 `scripts/compare/compare_trained_gnn_hrsga_ball.py` 也已默认将输出导入 `runs/benchmarks/`：

- 轨迹对比图默认写到 `runs/benchmarks/compare_trained_gnn_hrsga_{layout_mode}.png`
- 逐 seed 对比摘要默认写到 `runs/benchmarks/compare_trained_gnn_hrsga_{layout_mode}.json`

如需执行 HRSGA 消融训练，可运行：

```powershell
cd d:\VirtualSpace\rl_mppi\experiments\ball2D\hrsga_ball
D:/python3.12/python.exe scripts/train/train_ablation_hrsga_ball.py --ablation_name no_temporal --run_name hrsga_no_temporal
```

如需汇总多个消融版本的正式指标，可运行：

```powershell
cd d:\VirtualSpace\rl_mppi\experiments\ball2D\hrsga_ball
D:/python3.12/python.exe scripts/compare/compare_ablations_ball.py --labels HRSGA NoTemporal Dense UnifiedRelation NoGeometric --model_paths runs/hrsga_formal/checkpoints/hrsga_ball_best.pt runs/hrsga_no_temporal/checkpoints/hrsga_ball_best.pt runs/hrsga_dense/checkpoints/hrsga_ball_best.pt runs/hrsga_unified_relation/checkpoints/hrsga_ball_best.pt runs/hrsga_no_geometric/checkpoints/hrsga_ball_best.pt --num_tests 20 --layout_mode representative --summary_path runs/benchmarks/hrsga_ablations.json --plot_path runs/benchmarks/hrsga_ablations.png
```

## 生成正式 Benchmark 统计

`scripts/eval/benchmark_policy_ball.py` 用于对单个训练完成的策略 checkpoint 做批量评测，并输出正式统计结果。它会在指定布局上重复运行多个随机种子 episode，逐次记录每次测试结果，并汇总成总体指标，适合生成论文中的单模型 benchmark 数据。

当前脚本会统计这些指标：

- success rate
- average completed tasks
- average completed fraction
- average deadline satisfaction
- average collisions
- average mean task distance
- average minimum pair distance
- average missed deadlines
- average steps
- collision-stop rate
- average inference time
- p95 inference time

必填参数：

- `--model_type`：模型类型，当前常用为 `hrsga` 或 `standard_gnn`

常用可选参数：

- `--model_path`：checkpoint 路径；不填时默认从 `models/` 读取对应的 best checkpoint
- `--num_tests`：独立测试次数，默认 `20`
- `--seed`：起始随机种子，默认 `5300`
- `--max_steps`：单次 episode 最大步数；不填则使用 checkpoint 自带配置
- `--layout_mode`：测试布局，支持 `representative`、`structured`、`random`
- `--strict_collision_stop`：一旦发生碰撞立即终止该 episode，并统计 `collision_stop_rate`
- `--summary_path`：聚合统计 JSON 输出路径
- `--runs_path`：逐次测试结果 JSONL 输出路径

默认输出规则：

- summary JSON 默认写到 `runs/benchmarks/{model_type}_{layout_mode}_summary.json`
- runs JSONL 默认写到 `runs/benchmarks/{model_type}_{layout_mode}_runs.jsonl`

如果要评测当前正式 HRSGA 最优权重，可运行：

```powershell
cd d:\VirtualSpace\rl_mppi\experiments\ball2D\hrsga_ball
D:/python3.12/python.exe scripts/eval/benchmark_policy_ball.py --model_type hrsga --model_path models/hrsga_ball_best.pt --num_tests 20 --layout_mode representative --strict_collision_stop --summary_path runs/benchmarks/hrsga_representative_summary.json --runs_path runs/benchmarks/hrsga_representative_runs.jsonl
```

如果要评测当前正式 Standard GNN 权重，可运行：

```powershell
cd d:\VirtualSpace\rl_mppi\experiments\ball2D\hrsga_ball
D:/python3.12/python.exe scripts/eval/benchmark_policy_ball.py --model_type standard_gnn --model_path runs/standard_gnn_formal/checkpoints/standard_gnn_ball_best.pt --num_tests 20 --layout_mode representative --strict_collision_stop --summary_path runs/benchmarks/standard_gnn_representative_summary.json --runs_path runs/benchmarks/standard_gnn_representative_runs.jsonl
```

`summary.json` 适合直接引用到论文表格中，`runs.jsonl` 则更适合后续排查单个 seed 的成功、碰撞和步长表现。

## 绘制训练曲线

`scripts/plot/plot_training_curves_ball.py` 用于从训练日志 `metrics.jsonl` 生成训练过程图。当前脚本会绘制四张子图：

- train loss
- val loss
- eval success rate
- eval deadline satisfaction
- eval collisions

其中评估子图会同时保留原始评估点，并额外绘制平滑后的趋势线；如果日志里包含 `best_updated` 和 `collision_stop_rate`，图中还会标出最佳 checkpoint 所在 epoch，并在碰撞图里额外叠加 collision-stop rate 趋势。

必填参数：

- `--metrics_path`：训练过程中生成的 `metrics.jsonl` 路径
- `--plot_path`：输出图片路径

可选参数：

- `--title`：图表标题，默认是 `Training Metrics`
- `--smooth_window`：评估曲线平滑窗口，默认是 `3`
- `--hide_raw_points`：隐藏原始散点，只保留趋势线

如果要绘制当前正式 HRSGA 训练曲线，可运行：

```powershell
cd d:\VirtualSpace\rl_mppi\experiments\ball2D\hrsga_ball
D:/python3.12/python.exe scripts/plot/plot_training_curves_ball.py --metrics_path runs/hrsga_formal/logs/metrics.jsonl --plot_path runs/benchmarks/hrsga_formal_training_curves.png --title "HRSGA Formal Training Curves"
```

如果你想让评估曲线更平滑一些，可显式增大平滑窗口：

```powershell
cd d:\VirtualSpace\rl_mppi\experiments\ball2D\hrsga_ball
D:/python3.12/python.exe scripts/plot/plot_training_curves_ball.py --metrics_path runs/hrsga_formal/logs/metrics.jsonl --plot_path runs/benchmarks/hrsga_formal_training_curves_smooth.png --title "HRSGA Formal Training Curves" --smooth_window 5
```


输出图片建议统一写到 `runs/benchmarks/`，如果后续要在论文正文中固定引用，再把对应 PNG 复制到 `docs/paper/`。

## 推荐使用方式

如果你现在只关心当前主线，优先看这些文件：

1. `scripts/train/train_hrsga_ball.py`
2. `scripts/eval/test_hrsga_ball.py`
3. `hrsga_ball_model.py`
4. `scripts/train/train_standard_gnn_ball.py`
5. `scripts/compare/compare_trained_gnn_hrsga_ball.py`

## 备注

- 当前目录已按论文实验流程拆分为训练、评估、对比、可视化、论文文档和参考资料六类。
- `models/` 与 `runs/` 保持不变，便于延续现有检查点和实验结果。
