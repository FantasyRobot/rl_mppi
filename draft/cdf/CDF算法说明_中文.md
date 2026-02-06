# CDF (Configuration Space Distance Field) 算法中文说明

## 1. 算法原理

### 1.1 配置空间与任务空间

- **任务空间 (Task Space)**: 机器人末端执行器或工作对象的实际物理空间，通常是笛卡尔空间 (x, y, z)。
- **配置空间 (Configuration Space, C-space)**: 机器人所有关节角度的集合，表示机器人的所有可能姿态，通常用关节角度向量 (q₁, q₂, ..., qₙ) 表示。

在机器人运动规划中，障碍物在任务空间中的表示需要转换为配置空间中的障碍物，这样才能直接在关节角度空间中进行路径规划。

### 1.2 配置空间距离场 (CDF) 的概念

配置空间距离场 (CDF) 是一个定义在配置空间上的标量场，用于描述任意配置 (q) 到最近的配置空间障碍物边界的距离。

- **CDF 值**: 对于配置空间中的任意点 q，CDF(q) 表示 q 到最近的障碍物边界的欧几里得距离。
- **符号约定**: 障碍物内部的配置距离为负，障碍物外部为正，边界上为零。

### 1.3 CDF 与 SDF 的区别

- **SDF (Signed Distance Function)**: 定义在任务空间上，表示点到障碍物表面的有符号距离。
- **CDF**: 定义在配置空间上，表示配置到最近障碍物边界配置的有符号距离。

CDF 考虑了机器人的运动学约束，能够直接在关节空间中指导运动规划，避免了复杂的逆向运动学计算。

## 2. 核心实现

### 2.1 CDF2D 类结构

CDF2D 类是核心实现，主要包含以下功能：

| 模块 | 主要功能 | 关键方法 |
|------|---------|---------|
| 配置空间网格 | 创建配置空间采样网格 | `create_grid_torch()` |
| 数据生成 | 生成配置空间与任务空间的映射关系 | `generate_data()` |
| CDF 计算 | 计算配置到障碍物边界的距离 | `calculate_cdf()` |
| 投影方法 | 将配置投影到障碍物边界 | `projection()` |
| 射击方法 | 在配置空间中生成运动轨迹 | `shooting()` |

### 2.2 核心算法流程

#### 2.2.1 数据生成流程

1. 在任务空间中均匀采样点 p
2. 对于每个点 p，使用优化算法找到所有使机器人表面经过 p 的配置 q
3. 将任务空间点 p 与对应的配置 q 存储起来

```python
# 数据生成核心代码
def generate_data(self, nbDiscretization=50):
    # 在任务空间创建网格点
    x = torch.linspace(self.task_space[0][0], self.task_space[1][0], self.nbDiscretization).to(self.device)
    y = torch.linspace(self.task_space[0][1], self.task_space[1][1], self.nbDiscretization).to(self.device)
    xx, yy = torch.meshgrid(x, y)
    p = torch.cat([xx.reshape(-1,1), yy.reshape(-1,1)], dim=-1).to(self.device)
    
    data = {}
    for i, _p in enumerate(p):
        # 为每个任务空间点创建一个小圆圈
        grids = [Circle(center=_p, radius=0.001, device=device)]
        # 找到使机器人表面经过该点的所有配置
        q = self.find_q(grids)[1]
        data[i] = {
            'p': _p,
            'q': q
        }
    # 保存数据
    torch.save(tensor_data, os.path.join(CUR_PATH, 'data2D.pt'))
```

#### 2.2.2 CDF 计算流程

CDF 支持两种计算方法：

1. **在线计算 (online_computation)**: 实时为给定配置寻找最近的障碍物边界配置
2. **离线网格 (offline_grid)**: 使用预先生成的数据快速查找

```python
# CDF 计算核心代码
def calculate_cdf(self, q, obj_lists, method='online_computation', return_grad=False):
    Np = q.shape[0]
    
    if method == 'online_computation':
        # 在线查找最近的边界配置
        if not hasattr(self, 'q_0_level_set'):
            self.q_0_level_set = self.find_q(obj_lists)[1]
        dist = torch.norm(q.unsqueeze(1) - self.q_0_level_set.unsqueeze(0), dim=-1)
    
    # 计算最短距离
    d = torch.min(dist, dim=-1)[0]
    
    # 根据SDF确定符号（障碍物内部为负）
    d_ts = self.inference_sdf(q, obj_lists)
    mask = (d_ts < 0)
    d[mask] = -d[mask]
    
    if return_grad:
        grad = torch.autograd.grad(d, q, torch.ones_like(d))[0]
        return d, grad
    return d
```

#### 2.2.3 配置空间中的运动规划

CDF 提供了两种在配置空间中生成运动轨迹的方法：

1. **Shooting 方法**: 沿 CDF 梯度的正交方向移动，生成边界附近的轨迹
2. **投影方法**: 将随机配置投影到 CDF 的零水平集上

```python
# Shooting 方法核心代码
def shooting(self, q0, obj_lists, dt=1e-2, timestep=500, method='CDField'):
    q = q0
    q.requires_grad = True
    q_list = []
    
    for t in range(timestep):
        if method == 'CDField':
            # 使用CDF计算梯度
            d, g = self.calculate_cdf(q, obj_lists, return_grad=True)
        
        # 沿梯度的正交方向移动
        g = torch.nn.functional.normalize(g, dim=-1)
        g_orth = torch.stack([g[:,1], -g[:,0]], dim=-1)
        q = q + dt * g_orth
        q_list.append(q.detach().cpu().numpy())
    
    return np.array(q_list).transpose(1, 0, 2)
```

## 3. 源码使用方法

### 3.1 基本使用流程

1. **初始化 CDF2D 实例**

```python
import torch
from cdf import CDF2D

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 创建 CDF2D 实例
cdf = CDF2D(device)
```

2. **定义障碍物**

```python
from primitives2D_torch import Circle

# 创建障碍物列表（可以是多个障碍物的组合）
scene = [
    Circle(center=torch.tensor([2.5, 2.5]), radius=0.5, device=device),
    Circle(center=torch.tensor([-2.0, -1.5]), radius=0.3, device=device)
]
```

3. **计算配置的 CDF 值**

```python
# 定义配置向量（关节角度）
q = torch.tensor([[0.5, 0.8], [-1.2, 0.3]], requires_grad=True).to(device)

# 计算 CDF 值和梯度
d, grad = cdf.calculate_cdf(q, scene, return_grad=True)

print("CDF 值:", d)
print("CDF 梯度:", grad)
```

4. **在配置空间中生成运动轨迹**

```python
# 定义起始配置
q0 = torch.tensor([[-0.2, 0.0], [1.2, 1.0]]).to(device)

# 使用 CDField 方法生成轨迹
trajectories = cdf.shooting(q0, scene, method='CDField', timestep=500)

# 轨迹形状: [batch_size, timestep, num_joints]
print("轨迹形状:", trajectories.shape)
```

### 3.2 可视化功能

源码提供了多种可视化函数：

1. **绘制配置空间 CDF**

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 8))
cdf.plot_cdf(ax, scene)
plt.show()
```

2. **比较 SDF 和 CDF**

```python
fig1, ax1 = plt.subplots(figsize=(10, 8))
fig2, ax2 = plt.subplots(figsize=(10, 8))

# 绘制 SDF
cdf.plot_sdf(ax=ax1, obj_lists=scene)

# 绘制 CDF
cdf.plot_cdf(ax2, scene)

plt.show()
```

3. **绘制梯度投影**

```python
from cdf import plot_projection

# 绘制梯度投影结果
plot_projection(scene)
```

## 4. 应用场景

CDF 算法主要应用于以下场景：

1. **机器人运动规划**: 在复杂环境中为机器人规划无碰撞路径
2. **机械臂避障**: 为机械臂提供实时避障能力
3. **自主导航**: 为移动机器人提供配置空间的距离信息
4. **抓取规划**: 为抓取任务提供配置空间的避障指导

## 5. 代码优化建议

1. **数据生成优化**:
   - 可以使用更高效的采样策略减少计算时间
   - 考虑使用 GPU 加速数据生成过程

2. **CDF 计算优化**:
   - 可以使用 KD-tree 或其他空间索引结构加速最近邻搜索
   - 考虑使用神经网络拟合 CDF 以提高计算速度

3. **内存优化**:
   - 可以使用稀疏表示存储配置空间数据
   - 考虑使用更高效的数据格式减少内存占用

## 6. 总结

CDF (配置空间距离场) 是一种强大的工具，它将任务空间中的障碍物信息转换为配置空间中的距离场，从而能够直接在关节角度空间中进行运动规划。与传统的基于任务空间的方法相比，CDF 避免了复杂的逆向运动学计算，提高了运动规划的效率和可靠性。

本实现提供了完整的 CDF 算法实现，包括数据生成、CDF 计算、运动规划和可视化功能，可以直接用于 2D 机器人系统的运动规划应用。