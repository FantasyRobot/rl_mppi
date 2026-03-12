# hrsga_multi_balls

该目录用于验证 HRSGA 在二维多机器人多目标遍历与避障场景中的训练、测试和对比流程，当前已整理为三条主线：

- HRSGA 监督学习主线。
- TD3-HRSGA 强化学习主线。
- Standard GNN 基线主线，用于和 HRSGA 做同环境对比。

当前环境语义已经从“固定一机器人对应一个目标”改为“多机器人协同遍历多目标”：

- `num_balls` 与 `num_tasks` 可以独立设置，且测试时可覆盖 checkpoint 内的默认数量。
- 任意机器人都可以完成任意一个已释放、未完成的目标。
- `task_id` 与 `assigned_agent` 仍然保留，但现在是每一步动态更新的参考分配提示，不再表示永久绑定关系。
- 机器人可以提前朝“尚未释放”或“前序尚未完成”的后续目标移动并等待，但任务真正完成仍受释放时间与访问顺序约束。
- episode 成功条件改为所有目标均被完成，而不是每个机器人只到达自己的专属目标。

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
  - 支持 `--num_balls`、`--num_tasks` 在支持范围内覆盖测试规模。

- `test_td3_hrsga_ball.py`
  - TD3-HRSGA 测试入口。

- `benchmark_policy_ball.py`
  - 通用正式评估脚本，可对 HRSGA、Standard GNN、TD3 进行多随机种子统计。
  - 支持 `--num_balls`、`--num_tasks` 覆盖评测规模。

### `scripts/compare/`

- `compare_trained_gnn_hrsga_ball.py`
  - 使用相同随机种子，对训练后的 Standard GNN 和 HRSGA 进行轨迹对比。
  - 支持 `--num_balls`、`--num_tasks` 在相同规模下比较两种策略。

- `compare_policy_metrics_ball.py`
  - 对两个训练后模型做统一随机种子的聚合指标对比。
  - 支持 `--num_balls`、`--num_tasks` 统一覆盖评测规模。

- `compare_ablations_ball.py`
  - 对多个已训练 HRSGA 消融版本做统一随机种子的聚合指标对比。
  - 支持 `--num_balls`、`--num_tasks` 统一覆盖评测规模。

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

当前 HRSGA 与 Standard GNN 的监督训练入口都支持 `--num_tasks`，可以直接训练“机器人数量小于目标数量”的多目标遍历策略。例如可用 `--num_balls 2 --num_tasks 4` 训练两机器人协同遍历四目标的策略。

当前 `scripts/train/train_hrsga_ball.example.json` 已默认启用 `"expert_max_collisions": 0`，表示构建 imitation 数据集时只保留无碰撞 expert episode；脚本会自动继续采样，直到凑够配置中要求的训练和验证 episode 数。

## 实验准备

建议统一在当前目录下执行脚本：

```powershell
cd d:\VirtualSpace\rl_mppi\experiments\ball2D\hrsga_multi_balls
```

当前环境使用的 Python 可执行文件示例：

```powershell
D:/python3.12/python.exe
```

后文所有命令都默认从 `hrsga_multi_balls` 根目录执行。

### 配置文件与命令行覆盖规则

- `train_hrsga_ball.py`、`train_standard_gnn_ball.py`、`train_ablation_hrsga_ball.py` 支持 `--config_path`。
- 当配置文件中含有同名字段时，会先加载配置文件，再用命令行显式传入的参数覆盖。
- 适合把大部分正式训练参数写到 JSON，再只在命令行上覆盖 `--run_name`、`--resume_path`、`--num_balls`、`--num_tasks` 等少量字段。

### 任务顺序与驻留时间说明

当前“任务访问顺序”和“驻留步数”是环境级规则，不是训练脚本 CLI 参数：

- `enforce_visit_order`：是否强制按任务顺序访问。
- `min_dwell_steps`：每个任务最少需要连续驻留的步数。
- `max_dwell_steps`：每个任务最多的驻留步数上界。

当前实现中，`enforce_visit_order` 约束的是“何时允许开始驻留并完成任务”，而不是“何时允许机器人朝该任务移动”。因此多个机器人可以同时向不同候选目标预定位，等目标真正可用后再开始完成。

它们定义在 `envball_hrsga.py` 的 `HRSGABallEnvironment(...)` 构造函数中，所有训练、测试、benchmark、compare 脚本都会沿用这里的环境默认值。如果你要改任务顺序/驻留语义，应优先改环境构造参数，而不是训练命令。

## 常用实验命令

### 1. 训练 HRSGA 监督模型

```powershell
D:/python3.12/python.exe scripts/train/train_hrsga_ball.py --config_path scripts/train/train_hrsga_ball.example.json --run_name hrsga_formal
```

常用变体：

```powershell
D:/python3.12/python.exe scripts/train/train_hrsga_ball.py --config_path scripts/train/train_hrsga_ball.example.json --run_name hrsga_2ball_4task --num_balls 2 --num_tasks 4
```

核心参数：

- `--config_path`：训练配置 JSON 路径。
- `--epochs`：监督训练 epoch 数。
- `--batch_size`：每次梯度更新的样本批大小。
- `--learning_rate`：AdamW 学习率。
- `--weight_decay`：权重衰减系数。
- `--train_episodes`：构造 imitation 训练集时采样的 expert episode 数。
- `--val_episodes`：构造验证集时采样的 expert episode 数。
- `--rollout_episodes`：每轮评估时的 rollout 数量。
- `--eval_interval`：每隔多少个 epoch 做一次评估。
- `--save_interval`：每隔多少个 epoch 保存一次 latest checkpoint。
- `--num_balls`：机器人数量。
- `--num_tasks`：任务数量；可以大于机器人数量。
- `--max_steps`：单个 episode 的最大步数。
- `--hidden_dim`：隐藏层特征维度。
- `--num_heads`：关系注意力头数。
- `--topk_robot`：机器人-机器人关系保留的邻居数。
- `--topk_task`：机器人-任务关系保留的邻居数。
- `--topk_obstacle`：机器人-障碍物关系保留的邻居数。
- `--disable_temporal_bias`：关闭时间相关特征的使用，用于消融。
- `--disable_geometric_bias`：关闭几何关系偏置，用于消融。
- `--shared_relation_attention`：让不同关系类型共享关系注意力模块。
- `--use_dense_residual`：启用更密集的残差连接。
- `--dataset_layout_pattern`：expert 数据集的布局采样模式序列，默认混合 `structured/random/representative`。
- `--expert_max_collisions`：过滤 expert episode 的最大碰撞数阈值；`0` 表示只保留无碰撞轨迹。
- `--eval_layout_mode`：训练中评估阶段使用的布局类型。
- `--eval_strict_collision_stop` / `--no-eval_strict_collision_stop`：评估时是否一碰撞就终止。
- `--resume_path`：从已有 checkpoint 恢复训练。
- `--seed`：训练与采样起始随机种子。
- `--run_name`：运行名，决定 `runs/` 下归档目录名称。
- `--output_root`：训练产物输出根目录。

### 2. 训练 Standard GNN 监督基线

```powershell
D:/python3.12/python.exe scripts/train/train_standard_gnn_ball.py --config_path scripts/train/train_standard_gnn_ball.example.json --run_name standard_gnn_formal
```

常用变体：

```powershell
D:/python3.12/python.exe scripts/train/train_standard_gnn_ball.py --config_path scripts/train/train_standard_gnn_ball.example.json --run_name standard_gnn_2ball_4task --num_balls 2 --num_tasks 4
```

核心参数：

- `--config_path`：训练配置 JSON。
- `--epochs`、`--batch_size`、`--learning_rate`、`--weight_decay`：含义与 HRSGA 训练一致。
- `--train_episodes`、`--val_episodes`、`--rollout_episodes`：训练集、验证集、评估 rollout 数量。
- `--eval_interval`、`--save_interval`：评估和保存周期。
- `--num_balls`、`--num_tasks`、`--max_steps`：环境规模与 episode 长度。
- `--hidden_dim`：节点隐藏特征维度。
- `--edge_hidden_dim`：边特征编码隐藏维度。
- `--resume_path`：恢复训练的 checkpoint。
- `--seed`、`--run_name`、`--output_root`：随机种子、运行名、输出目录。

### 3. 训练 TD3-HRSGA 强化学习模型

```powershell
D:/python3.12/python.exe scripts/train/train_td3_hrsga_ball.py --total_steps 30000 --save_interval 2500 --eval_interval 2500
```

核心参数：

- `--total_steps`：总环境交互步数。
- `--batch_size`：每次 TD3 更新的采样批大小。
- `--update_after`：经验池至少积累多少步后才开始学习。
- `--updates_per_step`：每个环境步之后执行多少次参数更新。
- `--buffer_size`：经验回放池容量。
- `--random_exploration_episodes`：前期纯随机探索 episode 数。
- `--policy_sigma`：动作探索噪声标准差初值。
- `--policy_sigma_decay_rate`：探索噪声衰减率。
- `--resume_path`：从已有 TD3 checkpoint 继续训练。
- `--save_interval`：latest checkpoint 保存间隔。
- `--eval_interval`：周期性评估间隔。
- `--log_interval`：打印训练日志的步长。
- `--eval_episodes`：每次评估的 episode 数。
- `--seed`：随机种子。

说明：当前 TD3 训练脚本的 CLI 没有暴露 `--num_balls`、`--num_tasks`、`--min_dwell_steps`、`--max_dwell_steps` 这类环境规模/规则参数，默认沿用脚本和环境内部设置。

### 4. 训练 HRSGA 消融版本

```powershell
D:/python3.12/python.exe scripts/train/train_ablation_hrsga_ball.py --ablation_name no_temporal --config_path scripts/train/train_hrsga_ball.example.json --run_name hrsga_no_temporal
```

可选 `--ablation_name`：

- `no_temporal`：禁用时间相关特征。
- `dense`：把 top-k 关系近似切到稠密连接。
- `unified_relation`：共享关系注意力模块。
- `no_geometric`：禁用几何偏置特征。

除 `--ablation_name` 外，其余参数与 `train_hrsga_ball.py` 基本一致。

### 5. 测试单个 HRSGA 监督模型并出轨迹图

```powershell
D:/python3.12/python.exe scripts/eval/test_hrsga_ball.py --model_path models/hrsga_ball_best.pt --num_tests 5 --layout_mode representative --plot_path runs/benchmarks/hrsga_eval.png
```

核心参数：

- `--model_path`：待测试模型路径。
- `--num_tests`：测试 episode 数。
- `--max_steps`：单次测试最大步数；为空则沿用 checkpoint 内配置。
- `--plot_path`：轨迹图输出路径。
- `--show_plot`：直接弹出图窗。
- `--seed`：测试起始随机种子。
- `--layout_mode`：布局类型，支持 `representative`、`structured`、`random`。
- `--num_balls`、`--num_tasks`：覆盖测试时使用的规模。
- `--fixed_layout`：兼容旧入口，等价于 `--layout_mode structured`。
- `--random_layout`：兼容旧入口，等价于 `--layout_mode random`。

### 6. 测试单个 TD3-HRSGA 模型并出轨迹图

```powershell
D:/python3.12/python.exe scripts/eval/test_td3_hrsga_ball.py --model_path models/td3_hrsga_best.pt --num_tests 5 --layout_mode representative --plot_path runs/benchmarks/td3_eval.png
```

核心参数：

- `--model_path`、`--num_tests`、`--max_steps`、`--plot_path`、`--show_plot`、`--seed`、`--layout_mode`：含义与 HRSGA 测试一致。
- `--fixed_layout`、`--random_layout`：兼容旧布局入口。

说明：当前 TD3 测试脚本没有单独暴露 `--num_balls`、`--num_tasks`，测试规模默认来自 checkpoint 和环境默认值。

### 7. 对单个策略做正式 benchmark

```powershell
D:/python3.12/python.exe scripts/eval/benchmark_policy_ball.py --model_type hrsga --model_path models/hrsga_ball_best.pt --num_tests 20 --layout_mode representative --strict_collision_stop --summary_path runs/benchmarks/hrsga_representative_summary.json --runs_path runs/benchmarks/hrsga_representative_runs.jsonl
```

如果要评测 Standard GNN：

```powershell
D:/python3.12/python.exe scripts/eval/benchmark_policy_ball.py --model_type standard_gnn --model_path models/standard_gnn_ball_best.pt --num_tests 20 --layout_mode representative --strict_collision_stop
```

核心参数：

- `--model_type`：模型类型，支持 `hrsga`、`standard_gnn`、`td3`。
- `--model_path`：checkpoint 路径；为空时会自动从 `models/` 取对应 best 权重。
- `--num_tests`：重复测试次数。
- `--seed`：起始随机种子。
- `--max_steps`：测试最大步数。
- `--num_balls`、`--num_tasks`：覆盖评测时的机器人/任务规模。
- `--layout_mode`：评测布局。
- `--strict_collision_stop`：发生碰撞立即终止 episode，并统计 collision-stop rate。
- `--summary_path`：聚合统计 JSON 输出路径。
- `--runs_path`：逐次测试 JSONL 输出路径。

该脚本会输出：

- success rate
- average completed tasks / completed fraction
- average deadline satisfaction
- average collisions
- average mean task distance
- average minimum pair distance
- average missed deadlines
- average steps
- collision-stop rate
- average inference time / p95 inference time

### 8. 对比 HRSGA 与 Standard GNN 的轨迹

```powershell
D:/python3.12/python.exe scripts/compare/compare_trained_gnn_hrsga_ball.py --hrsga_model_path models/hrsga_ball_best.pt --gnn_model_path models/standard_gnn_ball_best.pt --num_tests 5 --layout_mode representative --plot_path runs/benchmarks/compare_trained_gnn_hrsga_representative.png --summary_path runs/benchmarks/compare_trained_gnn_hrsga_representative.json
```

如果要隐藏图中的任务编号，同时关闭任务顺序约束：

```powershell
D:/python3.12/python.exe scripts/compare/compare_trained_gnn_hrsga_ball.py --hrsga_model_path models/hrsga_ball_best.pt --gnn_model_path models/standard_gnn_ball_best.pt --num_tests 5 --layout_mode representative --task_label_mode none --no-enforce_visit_order
```

核心参数：

- `--hrsga_model_path`：HRSGA checkpoint。
- `--gnn_model_path`：Standard GNN checkpoint。
- `--num_tests`：共享随机种子下的对比次数。
- `--max_steps`：对比时的最大步数。
- `--plot_path`：轨迹对比图输出路径。
- `--summary_path`：逐 seed 摘要 JSON 输出路径。
- `--show_plot`：显示图窗。
- `--seed`：起始随机种子。
- `--num_balls`、`--num_tasks`：覆盖对比环境规模。
- `--layout_mode`：布局模式。
- `--task_label_mode`：任务标签显示模式，支持 `ranked`、`index`、`none`。
- `--enforce_visit_order` / `--no-enforce_visit_order`：是否在对比环境中强制任务顺序约束。
- `--fixed_layout`、`--random_layout`：兼容旧布局开关。

补充说明：

- `--task_label_mode ranked`：按 `visit_rank` 显示任务编号，适合查看有序访问场景。
- `--task_label_mode index`：按任务索引显示编号，只改图上标签，不改环境规则。
- `--task_label_mode none`：不显示任务编号，只保留任务点标记。
- `--no-enforce_visit_order`：关闭环境里的顺序约束，任务完成不再要求前序任务先完成。

### 9. 对比两个策略的聚合指标

```powershell
D:/python3.12/python.exe scripts/compare/compare_policy_metrics_ball.py --left_model_type hrsga --left_model_path models/hrsga_ball_best.pt --left_label HRSGA --right_model_type standard_gnn --right_model_path models/standard_gnn_ball_best.pt --right_label StandardGNN --num_tests 20 --layout_mode representative --strict_collision_stop --summary_path runs/benchmarks/compare_hrsga_vs_standard_gnn_representative.json --plot_path runs/benchmarks/compare_hrsga_vs_standard_gnn_representative.png
```

核心参数：

- `--left_model_type`、`--right_model_type`：左右两侧模型类型，支持 `hrsga`、`standard_gnn`、`td3`。
- `--left_model_path`、`--right_model_path`：左右模型 checkpoint。
- `--left_label`、`--right_label`：图表展示名称。
- `--num_tests`、`--seed`、`--max_steps`：对比的随机种子数量和步长配置。
- `--num_balls`、`--num_tasks`：统一覆盖两侧评测规模。
- `--layout_mode`：布局类型。
- `--strict_collision_stop`：碰撞即停。
- `--summary_path`、`--plot_path`：聚合摘要和柱状图输出路径。

### 10. 对比多个 HRSGA 消融模型

```powershell
D:/python3.12/python.exe scripts/compare/compare_ablations_ball.py --labels HRSGA NoTemporal Dense UnifiedRelation NoGeometric --model_paths runs/hrsga_formal/checkpoints/hrsga_ball_best.pt runs/hrsga_no_temporal/checkpoints/hrsga_ball_best.pt runs/hrsga_dense/checkpoints/hrsga_ball_best.pt runs/hrsga_unified_relation/checkpoints/hrsga_ball_best.pt runs/hrsga_no_geometric/checkpoints/hrsga_ball_best.pt --num_tests 20 --layout_mode representative --strict_collision_stop --summary_path runs/benchmarks/hrsga_ablations.json --plot_path runs/benchmarks/hrsga_ablations.png
```

核心参数：

- `--labels`：每个消融模型的显示名称，数量必须与 `--model_paths` 一致。
- `--model_paths`：待比较的 checkpoint 路径列表。
- `--num_tests`、`--seed`、`--max_steps`：评测重复次数与步长。
- `--num_balls`、`--num_tasks`：统一覆盖环境规模。
- `--layout_mode`：布局模式。
- `--strict_collision_stop`：碰撞即停。
- `--summary_path`、`--plot_path`：结果输出路径。

### 11. 绘制训练曲线

```powershell
D:/python3.12/python.exe scripts/plot/plot_training_curves_ball.py --metrics_path runs/hrsga_formal/logs/metrics.jsonl --plot_path runs/benchmarks/hrsga_formal_training_curves.png --title "HRSGA Formal Training"
```

核心参数：

- `--metrics_path`：训练日志文件路径，一般为 `logs/metrics.jsonl`。
- `--plot_path`：输出图片路径。
- `--title`：图标题。
- `--smooth_window`：滑动平均窗口大小。
- `--hide_raw_points`：隐藏原始散点，只画平滑曲线。

## 推荐实验流程

### 复现一套完整监督学习对比

1. 训练 HRSGA：运行 `scripts/train/train_hrsga_ball.py`。
2. 训练 Standard GNN：运行 `scripts/train/train_standard_gnn_ball.py`。
3. 跑单模型 benchmark：运行 `scripts/eval/benchmark_policy_ball.py`。
4. 跑轨迹对比图：运行 `scripts/compare/compare_trained_gnn_hrsga_ball.py`。
5. 跑聚合指标对比：运行 `scripts/compare/compare_policy_metrics_ball.py`。
6. 画训练曲线：运行 `scripts/plot/plot_training_curves_ball.py`。

### 做新任务语义下的规模泛化实验

可直接覆盖：

- `--num_balls`
- `--num_tasks`
- `--max_steps`
- `--layout_mode`

例如：

```powershell
D:/python3.12/python.exe scripts/eval/benchmark_policy_ball.py --model_type hrsga --model_path models/hrsga_ball_best.pt --num_balls 2 --num_tasks 5 --max_steps 220 --num_tests 20 --layout_mode representative
```

这条命令表示：使用现有 HRSGA 权重，在“两机器人、五任务、220 步上限”的新规模上做 20 次 benchmark。

## 图表与结果归档规范

- 程序运行产生的原始结果统一写入 `runs/` 和 `runs/benchmarks/`。
- 论文正文默认引用 `docs/paper/` 下的稳定图片快照，避免后续重复运行覆盖论文插图。
- 当 `runs/benchmarks/` 中生成新的正式图表后，如需在论文中固定引用，再将对应 PNG 同步到 `docs/paper/`。
- `smoke_*.json`、`smoke_*.png`、`*_smoke.png` 不进入论文正文，也不覆盖 `docs/paper/` 中的正式快照。
- 新增 benchmark 文件时，优先沿用 `任务名_场景名_指标名` 或 `方法对比_场景名` 的命名方式，避免 `final`、`new` 之类不可追踪命名。
