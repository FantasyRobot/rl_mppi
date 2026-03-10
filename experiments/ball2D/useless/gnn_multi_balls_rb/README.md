# gnn_multi_balls_rb

这个目录是一个 **RoboBallet 风格的 2D 多球复刻实验**。它不是直接复用原 RoboBallet 机械臂环境，而是在当前 2D 多球任务上尽量对齐 RoboBallet 的训练目标和训练方式。

对齐内容：

- 使用 TD3，而不是 SAC
- reward 使用 score difference
- 加入 collision penalty 和 acceleration cost
- 碰撞后采用更接近 RoboBallet 的 safety rollback，而不是只保留惩罚
- 所有 target 完成后，对已接近起点的小球启用 last-mile return takeover
- 不使用 `warmup_easy / warmup_wide / transition_full` 这类显式 curriculum
- 使用 deterministic policy + decaying Gaussian exploration noise
- 使用 episode-level HER rerun，而不是 transition-level relabel
- 保留 return-to-start 任务结构

不能一比一对齐的部分：

- 原 RoboBallet 是多机械臂 3D 任务，这里是 2D 多球环境
- 原 RoboBallet 的碰撞来自机械臂与障碍/机器人几何体，这里映射为小球之间的近距离碰撞惩罚
- 原 RoboBallet 是异步 actor-learner 架构，这里仍然是单进程训练脚本，便于在本仓库直接实验

## 1. 目录功能

- `envball_gnn_rb.py`
  - RoboBallet 风格 2D 环境
  - score-difference reward
  - collision penalty
  - acceleration cost
  - return-to-start score
  - start state 重放支持, 用于 HER rerun
- `train_td3_gnn_ball.py`
  - TD3 + GNN 训练入口
  - twin critics
  - target smoothing
  - random exploration episodes
  - decaying policy sigma
  - episode-level HER rerun
  - 更细的 eval / training stats 聚合
- `test_td3_gnn_ball.py`
  - 加载模型并做 rollout 可视化
- `models/`
  - `td3_gnn_latest.npz`: 最新 checkpoint
  - `td3_gnn_best.npz`: 最优 checkpoint

## 2. 训练目标

环境每一步的 reward 形式为：

```text
reward = new_score - old_score - collision_penalty - acceleration_cost
```

其中：

- `new_score - old_score`
  - 对齐 RoboBallet 的 score-difference 设计
- `collision_penalty`
  - 当前 2D 版本中按“发生近距离碰撞并被安全回滚的小球数量”计算
- `acceleration_cost`
  - 按均方加速度惩罚

当前 score 由两部分组成：

- 目标完成分数
- 所有目标完成后的 return-to-start 分数

当前 return 阶段额外做了两点强化：

- return score 不再只是简单平均距离平滑项，而是混合了阈值内 parked ratio 和更陡的接近度分数
- 首次进入 all-targets-done 阶段时，会额外给一个一次性的 phase-switch bonus，帮助策略从“继续追 target”切到“开始回位”
- 全部目标完成后，若某个 ball 已进入起点附近阈值内，环境会直接接管最后一小段回位，做法上对齐 RoboBallet 原版的 last-mile takeover
- 图特征里的 return progress 现在直接基于原始回位距离计算，并和环境中的 `return_threshold=0.35` 保持一致，避免网络看到的回位语义与成功判定错位

默认目标 shaping 为 `0.0`，即尽量依赖稀疏目标完成信号和 HER，而不是依赖手工 shaping。

## 3. 训练方式

本实验与 `gnn_multi_balls` 的主要差异：

1. 算法从 SAC 改为 TD3
2. 去掉显式 curriculum
3. 使用 episode 数控制随机探索阶段
4. 使用 decaying exploration noise
5. HER 通过“重建新 start state 并重跑整条轨迹”完成

当前 HER 已进一步向 RoboBallet 靠拢：

- 只对最终未完成的目标做 hindsight target replacement
- 优先从真实“目标完成事件”对应的 step 里选取 anchor
- 如果一条轨迹里没有明显完成事件，再退回到带后段偏置的 future step 采样
- 目标位置直接取该时刻某个 ball 的真实位置，优先复用真实完成时最接近目标的 ball
- 用原动作序列整段回放，而不是局部改 reward
- 会记录 hindsight episode 的目标数、改动目标数和 imagined 结果统计
- 默认会同时写入三类 hindsight episode：一个按 `replay_hindsight_num_targets` 做部分完成 curriculum，一个偏向 `NUM_TARGETS-2` 附近的近全完成版本，一个直接对齐到全目标完成，避免 replay 长期停留在中等完成度样本

训练流程：

```text
reset env
-> rollout one episode
-> write transitions to replay
-> TD3 updates
-> generate hindsight episode if needed
-> write hindsight transitions to replay
-> periodic eval and checkpoint
```

## 4. 关键接口

### 4.1 环境

环境类：

```python
BallRoboBalletEnvironment(...)
```

关键方法：

```python
env.reset(start_state=None)
env.step(action)
env.get_start_state()
env.current_score_value()
env.num_targets_done()
env.total_collision_penalty()
env.total_acceleration()
env.terminal()
```

### 4.2 训练

训练入口：

```python
train(
    *,
    total_steps=30000,
    batch_size=128,
    update_after=1500,
    updates_per_step=2,
    buffer_size=250000,
    norm_steps=1500,
  random_exploration_episodes=12,
  policy_sigma=0.25,
    policy_sigma_decay_rate=0.8,
    replay_hindsight_num_targets=4,
  collision_penalty_scaling=1.0,
    resume_path=None,
    save_interval=2500,
    eval_interval=2500,
    log_interval=250,
    eval_episodes=6,
    seed=0,
    reward_loss_weight=0.0,
)
```

### 4.3 测试

测试入口：

```python
test_td3_gnn_ball(
    model_path,
    num_tests=5,
    max_steps=320,
    plot_path='gnn_multi_ball_rb_eval.png',
    show_plot=False,
)
```

## 5. 参数说明

### `--total_steps`

- 总训练步数

### `--random_exploration_episodes`

- 前多少个 episode 完全随机探索
- 对齐 RoboBallet actor 中的 `random_exploration_episodes`
- 当前默认值改为 `12`，避免前期 replay 被过多纯随机轨迹占满

### `--policy_sigma`

- 探索噪声初始标准差
- 当前默认值改为 `0.25`，进一步压低中后期策略噪声，优先打破 eval 平台期

### `--policy_sigma_decay_rate`

- 探索噪声指数衰减率

### `--replay_hindsight_num_targets`

- hindsight 轨迹希望“至少完成多少个目标”
- 当前 8 个目标的默认值为 `4`，更偏向推动中后期从局部完成过渡到更完整任务
- 这个值现在表示部分完成 hindsight 的 curriculum 下限；训练脚本还会额外生成“近全完成”和“全目标完成” hindsight，补足最后几目标和 return-to-start 所需样本
- 如果真实轨迹已经达到该数量，就不会额外生成 hindsight episode

### `--collision_penalty_scaling`

- 碰撞惩罚缩放系数
- 当前默认值改为 `1.0`，继续减轻 critic 被碰撞项主导的问题

### `--reward_loss_weight`

- critic 的 reward auxiliary head 权重
- 默认为 `0.0`，和 RoboBallet 默认一致

## 6. 快捷命令

进入目录：

```powershell
cd d:/VirtualSpace/rl_mppi/experiments/ball2D/gnn_multi_balls_rb
```

从头训练：

```powershell
D:/python3.12/python.exe train_td3_gnn_ball.py --total_steps 30000 --batch_size 128 --update_after 1500 --updates_per_step 2 --buffer_size 250000 --norm_steps 1500 --random_exploration_episodes 12 --policy_sigma 0.25 --policy_sigma_decay_rate 0.8 --replay_hindsight_num_targets 4 --collision_penalty_scaling 1.0 --eval_interval 2500 --save_interval 2500 --log_interval 250 --eval_episodes 6
```

快速冒烟测试：

```powershell
D:/python3.12/python.exe train_td3_gnn_ball.py --total_steps 3000 --batch_size 128 --update_after 256 --updates_per_step 1 --buffer_size 50000 --norm_steps 256 --random_exploration_episodes 6 --policy_sigma 0.25 --replay_hindsight_num_targets 4 --collision_penalty_scaling 1.0 --eval_interval 500 --save_interval 500 --log_interval 100 --eval_episodes 2
```

从 best 继续训练：

```powershell
D:/python3.12/python.exe train_td3_gnn_ball.py --resume_path models/td3_gnn_best.npz --total_steps 10000 --batch_size 128 --update_after 256 --updates_per_step 2 --random_exploration_episodes 0 --policy_sigma 0.2 --replay_hindsight_num_targets 4 --collision_penalty_scaling 1.0 --eval_interval 1000 --save_interval 1000 --log_interval 250 --eval_episodes 6
```

测试 best 模型：

```powershell
D:/python3.12/python.exe test_td3_gnn_ball.py
```

指定模型测试：

```powershell
D:/python3.12/python.exe -c "from test_td3_gnn_ball import test_td3_gnn_ball; test_td3_gnn_ball('models/td3_gnn_best.npz', num_tests=4, max_steps=320, plot_path='gnn_multi_ball_rb_eval.png', show_plot=False)"
```

## 7. 使用建议

- 如果目标是对比 RoboBallet 风格训练方法，用这个目录，不要和 `gnn_multi_balls` 混用
- 如果你只想看“是否比 SAC 更接近稀疏目标 + HER 的训练行为”，优先比较两边的评估曲线和 targets_done
- 如果 TD3 训练早期不稳定，优先调小 `policy_sigma`，再考虑减小 `collision_penalty_scaling`
- 如果训练后期一直 completion 上不去，优先增大 `replay_hindsight_num_targets` 到 `4`

训练日志里现在建议重点看这些指标：

- 日志现在按分组短标签打印：
  - `phase(s/all/ret/park)` = success / all_targets_done / entered_return_phase / return_ready
  - `task(score/ret/tgt)` = avg_final_score / avg_return / avg_targets_done
  - `state(rdist/coll/len)` 或 `state(rdist/coll)` = avg_final_return_distance_mean / avg_collision_penalty / avg_length
  - `her(ep/tr/ret/tgt)` = hindsight episodes / hindsight transitions / her_entered_return_phase_rate / her_avg_targets_done
  - `her_mix(tgt:count)` = 最近一段 hindsight episode 的目标数分布，例如 `4:2,6:5,8:3`
  - `loss(q1/q2/pi)` = q1_loss / q2_loss / policy_loss

- `success_rate`
  - 严格成功率，要求目标完成并回到起点
- `all_targets_done_rate`
  - 目标是否全部完成，不要求已经停回起点
- `entered_return_phase_rate`
  - 是否已经进入回位阶段；在当前实现里基本等同于 all targets done，但单独打印后更方便判断训练是否已经跨过任务阶段边界
- `return_ready_rate`
  - 全目标完成后是否已经进入回位阈值
- `avg_final_score`
  - 最终 score，更接近 RoboBallet 的主评价量
- `avg_targets_done`
  - 平均完成目标数
- `her_eps` / `her_transitions`
  - hindsight episode 数量和写入 replay 的 hindsight transition 数量
- `her_avg_targets_done` / `her_avg_score`
  - imagined trajectory 的完成情况，用来判断 HER 是否真的在提供更容易的成功样本

## 8. 输出文件

- `models/td3_gnn_latest.npz`
- `models/td3_gnn_best.npz`
- `gnn_multi_ball_rb_eval.png` 或自定义 plot 路径