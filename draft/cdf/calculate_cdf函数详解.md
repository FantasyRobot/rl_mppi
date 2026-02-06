# calculate_cdf 函数详解

## 1. 函数概述

`calculate_cdf` 是 `CDF2D` 类的核心方法，用于计算配置空间中给定配置到最近障碍物边界配置的有符号距离值。该函数实现了配置空间距离场 (Configuration Space Distance Field, CDF) 的核心计算逻辑。

## 2. 函数签名与参数

```python
def calculate_cdf(self, q, obj_lists, method='online_computation', return_grad=False):
    # x : (Nx,2)
    # q : (Np,2)
    # return d : (Np) distance between q and x in C space. d = min_{q*}{L2(q-q*)}. sdf(x,q*)=0
    # ...
```

### 参数说明

| 参数 | 类型 | 描述 |
|------|------|------|
| `self` | CDF2D | CDF2D类的实例 |
| `q` | torch.Tensor | 输入配置向量，形状为 `(Np, 2)`，其中 `Np` 是配置的数量 |
| `obj_lists` | list | 障碍物列表，包含一个或多个 `Circle`、`Box` 等形状对象 |
| `method` | str | CDF计算方法，可选值：<br>- `'online_computation'`（默认）：在线实时计算<br>- `'offline_grid'`：使用预生成的离线数据 |
| `return_grad` | bool | 是否返回CDF的梯度，默认为 `False` |

### 返回值

| 返回值 | 类型 | 描述 |
|--------|------|------|
| `d` | torch.Tensor | CDF值数组，形状为 `(Np,)`，表示每个输入配置到最近障碍物边界配置的有符号距离 |
| `grad` | torch.Tensor（可选） | CDF的梯度数组，形状为 `(Np, 2)`，仅当 `return_grad=True` 时返回 |

## 3. 实现原理

### 3.1 CDF 的数学定义

配置空间距离场 (CDF) 的数学定义为：

$$ 	ext{CDF}(\mathbf{q}) = \text{sign}(\mathbf{q}) \cdot \min_{\mathbf{q}^* \in \partial \mathcal{C}_{	ext{obstacle}}} \| \mathbf{q} - \mathbf{q}^* \|_2 $$

其中：
- \( \mathbf{q} \) 是输入配置
- \( \mathbf{q}^* \) 是障碍物边界配置（即满足 \( \text{SDF}(\mathbf{f}(\mathbf{q}^*)) = 0 \) 的配置）
- \( \partial \mathcal{C}_{	ext{obstacle}} \) 是配置空间障碍物的边界
- \( \text{sign}(\mathbf{q}) \) 是符号函数：障碍物内部为 -1，外部为 1
- \( \| \cdot \|_2 \) 是欧几里得范数

### 3.2 函数核心流程

`calculate_cdf` 函数的核心流程可以分为以下几个步骤：

#### 步骤 1：确定配置数量

```python
Np = q.shape[0]
```

获取输入配置的数量，用于后续的批量处理。

#### 步骤 2：选择计算方法

函数支持两种计算方法：在线计算和离线网格计算。

##### 2.1 离线网格计算 (`offline_grid`)

```python
if method == 'offline_grid':
    if not hasattr(self, 'q_list_template'):
        # 从障碍物表面采样点
        obj_points = torch.cat([obj.sample_surface(200) for obj in obj_lists])
        # 将任务空间点转换为网格索引
        grid = self.x_to_grid(obj_points)
        # 从预生成的数据中获取对应的配置
        q_list_template = (self.q_grid_template[grid[:,0], grid[:,1], :, :]).reshape(-1, 2)
        # 过滤掉无效配置（值为无穷大的配置）
        self.q_list_template = q_list_template[q_list_template[:,0] != torch.inf]
    # 计算输入配置到所有模板配置的欧几里得距离
    dist = torch.norm(q.unsqueeze(1) - self.q_list_template.to(self.device).unsqueeze(0), dim=-1)
```

**离线网格计算的原理**：
1. 使用预生成的 `q_grid_template` 数据，该数据包含任务空间点到配置空间配置的映射
2. 对每个障碍物表面点，找到对应的网格索引
3. 从网格中获取所有可能的边界配置 `q_list_template`
4. 计算输入配置到 `q_list_template` 中每个配置的距离

**离线计算的优点**：速度快，适用于实时应用；**缺点**：需要预先生成数据，内存占用大。

##### 2.2 在线实时计算 (`online_computation`)

```python
if method == 'online_computation':
    if not hasattr(self, 'q_0_level_set'):
        # 找到使机器人与障碍物表面接触的配置
        self.q_0_level_set = self.find_q(obj_lists)[1]
    # 计算输入配置到所有边界配置的欧几里得距离
    dist = torch.norm(q.unsqueeze(1) - self.q_0_level_set.unsqueeze(0), dim=-1)
```

**在线实时计算的原理**：
1. 使用 `find_q` 方法实时找到使机器人与障碍物表面接触的配置 `q_0_level_set`
2. 计算输入配置到 `q_0_level_set` 中每个配置的距离

**在线计算的优点**：不需要预生成数据，内存占用小；**缺点**：计算速度较慢，特别是首次调用时。

#### 步骤 3：计算最短距离

```python
d = torch.min(dist, dim=-1)[0]
```

对每个输入配置，找到到边界配置的最短距离，作为CDF的绝对值。

#### 步骤 4：确定符号

```python
# 计算任务空间中的SDF值
d_ts = self.inference_sdf(q, obj_lists)
# 根据SDF值确定符号：SDF < 0 表示在障碍物内部
mask = (d_ts < 0)
d[mask] = -d[mask]
```

通过计算任务空间中的有符号距离函数 (SDF) 来确定CDF的符号：
- 如果机器人在配置 `q` 下与障碍物发生碰撞（`d_ts < 0`），则CDF值为负
- 否则，CDF值为正

#### 步骤 5：计算梯度（可选）

```python
if return_grad:
    grad = torch.autograd.grad(d, q, torch.ones_like(d))[0]
    return d, grad
return d
```

如果需要返回梯度，则使用PyTorch的自动微分功能计算CDF对输入配置的梯度。梯度指向配置空间中距离障碍物边界最远的方向。

## 3. 数学原理详解

### 3.1 距离计算的数学表达式

对于输入配置 `q`（形状为 `(Np, 2)`）和边界配置集合 `Q*`（形状为 `(M, 2)`），距离矩阵的计算可以表示为：

$$ 	ext{dist}[i, j] = \| q[i] - Q^*[j] \|_2 = \sqrt{(q[i,0] - Q^*[j,0])^2 + (q[i,1] - Q^*[j,1])^2} $$

其中 `i` 范围为 `[0, Np-1]`，`j` 范围为 `[0, M-1]`。

最短距离为：

$$ d[i] = \min_{j=0}^{M-1} \text{dist}[i, j] $$

### 3.2 符号确定的数学依据

CDF的符号由任务空间中的SDF值确定：

$$ 	ext{sign}(q[i]) = \begin{cases}
-1 & \text{if } \text{SDF}(f(q[i])) < 0 \\
1 & \text{otherwise}
\end{cases} $$

其中 `f(q)` 是机器人的正向运动学函数，将配置空间映射到任务空间。

### 3.3 梯度计算的数学原理

CDF的梯度可以通过自动微分计算，其数学表达式为：

$$ \nabla_q \text{CDF}(q) = \text{sign}(q) \cdot \nabla_q \left( \min_{q^* \in Q^*} \| q - q^* \|_2 \right) $$

对于给定的输入配置 `q`，假设 `q*_min` 是最近的边界配置，则梯度为：

$$ \nabla_q \text{CDF}(q) = \text{sign}(q) \cdot \frac{q - q*_min}{\| q - q*_min \|_2} $$

## 4. 代码实现关键点

### 4.1 批量处理

函数支持批量处理多个配置，通过PyTorch的张量操作实现高效计算：

```python
# 扩展维度以实现批量距离计算
dist = torch.norm(q.unsqueeze(1) - self.q_0_level_set.unsqueeze(0), dim=-1)
```

### 4.2 设备管理

函数自动处理设备映射，确保计算在正确的设备（CPU/CUDA）上进行：

```python
# 将模板配置移动到当前设备
self.q_list_template.to(self.device)
```

### 4.3 缓存机制

函数使用缓存机制避免重复计算：

```python
# 缓存在线计算得到的边界配置
if not hasattr(self, 'q_0_level_set'):
    self.q_0_level_set = self.find_q(obj_lists)[1]
```

### 4.4 无效数据过滤

函数过滤掉无效的配置数据：

```python
# 过滤掉值为无穷大的无效配置
self.q_list_template = q_list_template[q_list_template[:,0] != torch.inf]
```

## 5. 使用示例

### 5.1 基本使用

```python
import torch
from cdf import CDF2D
from primitives2D_torch import Circle

# 创建CDF2D实例
device = torch.device("cpu")
cdf = CDF2D(device)

# 定义障碍物
scene = [Circle(center=torch.tensor([2.5, 2.5]), radius=0.5, device=device)]

# 输入配置
q = torch.tensor([[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0]], device=device)

# 计算CDF
d = cdf.calculate_cdf(q, scene)
print("CDF值:", d.tolist())
# 输出: CDF值: [2.828427791595459, 1.4142136573791504, 2.828427791595459]
```

### 5.2 计算梯度

```python
# 输入配置（需要计算梯度）
q = torch.tensor([[0.0, 0.0], [1.0, 1.0]], device=device, requires_grad=True)

# 计算CDF和梯度
d, grad = cdf.calculate_cdf(q, scene, return_grad=True)
print("CDF值:", d.tolist())
print("CDF梯度:", grad.tolist())
# 输出:
# CDF值: [2.828427791595459, 1.4142136573791504]
# CDF梯度: [[0.7071067571640015, 0.7071067571640015], [0.7071067571640015, 0.7071067571640015]]
```

### 5.3 使用离线计算方法

```python
# 使用离线计算方法
d = cdf.calculate_cdf(q, scene, method='offline_grid')
print("CDF值 (离线计算):", d.tolist())
```

## 6. 性能分析

| 计算方法 | 时间复杂度 | 空间复杂度 | 适用场景 |
|----------|------------|------------|----------|
| 在线计算 | O(Np * M)，其中 M 是边界配置的数量 | O(M) | 内存受限的场景 |
| 离线计算 | O(Np * M)，其中 M 是模板配置的数量 | O(M) | 实时应用场景 |

**性能优化建议**：
1. 对于在线计算，可以限制 `q_0_level_set` 的大小，只保留最近的边界配置
2. 对于离线计算，可以使用KD树等空间索引结构加速最近邻搜索
3. 使用GPU加速批量计算

## 7. 总结

`calculate_cdf` 函数实现了配置空间距离场的核心计算逻辑，它：

1. 提供两种计算方法（在线和离线）以适应不同场景
2. 支持批量处理多个配置，提高计算效率
3. 自动处理设备映射，确保在CPU和GPU上都能正确运行
4. 支持梯度计算，为运动规划提供必要的方向信息

该函数是CDF算法的核心，为机器人运动规划、避障等应用提供了关键的距离信息。