# CDF（配置空间距离场）原理、计算与轨迹生成（整合版）

> 目的：给出配置空间距离场（Configuration-space Distance Field, **CDF**）的数学定义、可计算形式，以及如何利用 CDF 直接在关节空间生成“沿等高线运动”的轨迹。
>
> 本文档整合自 `draft/cdf/` 下的多份说明（原理、公式推导、`calculate_cdf` 实现说明、以及论文版轨迹生成描述），并作为后续唯一维护版本。
>
> 记号约定：
> - 关节配置 $\mathbf{q}\in\mathbb{R}^n$（旋转关节常用范围 $[-\pi,\pi]$，实现中通常会 wrap）。
> - 机器人几何在任务空间中的集合记为 $\mathcal{R}(\mathbf{q})$。
> - 障碍物集合为 $\mathcal{O}$（本文实验为圆形障碍物，半径可带安全膨胀）。
> - 碰撞集合（C-space obstacle）为 $\mathcal{C}_{\text{obs}}=\{\mathbf{q}\mid \mathcal{R}(\mathbf{q})\cap\mathcal{O}\neq\emptyset\}$。

---

## 0. 配置空间与运动学基础（补充）

### 0.1 配置空间与正向运动学

对于具有 $n$ 个关节的机器人，其配置空间为

$$
\mathcal{C}=\{\mathbf{q}=(q_1,\dots,q_n)\in\mathbb{R}^n\}.
$$

任务空间位置与配置的映射由正向运动学给出：

$$
\mathbf{x}=\mathbf{f}(\mathbf{q}).
$$

以二维平面 $n$ 连杆为例，末端位置可写为

$$
\begin{aligned}
x(\mathbf{q}) &= \sum_{i=1}^{n} L_i\cos\Big(\sum_{j=1}^{i} q_j\Big),\\
y(\mathbf{q}) &= \sum_{i=1}^{n} L_i\sin\Big(\sum_{j=1}^{i} q_j\Big).
\end{aligned}
$$

在本文 Robot2D（两连杆）场景里，$n=2$。

### 0.2 配置空间障碍物（C-obstacle）

任务空间障碍物集合为 $\mathcal{O}$，机器人在配置 $\mathbf{q}$ 下占据的几何集合为 $\mathcal{R}(\mathbf{q})$。配置空间障碍物集合定义为

$$
\mathcal{C}_{\text{obs}}=\{\mathbf{q}\in\mathcal{C}\mid \mathcal{R}(\mathbf{q})\cap\mathcal{O}\neq\emptyset\}.
$$

其边界 $\partial\mathcal{C}_{\text{obs}}$ 对应“刚好接触”的配置集合。

### 0.3 雅可比矩阵（速度映射，选读）

雅可比矩阵 $\mathbf{J}(\mathbf{q})$ 满足

$$
\dot{\mathbf{x}}=\mathbf{J}(\mathbf{q})\dot{\mathbf{q}}.
$$

对于二维两连杆（连杆长度 $L_1,L_2$），有

$$
\mathbf{J}(\mathbf{q})=
\begin{bmatrix}
-L_1\sin q_1 - L_2\sin(q_1+q_2) & -L_2\sin(q_1+q_2)\\
\ \ L_1\cos q_1 + L_2\cos(q_1+q_2) & \ \ L_2\cos(q_1+q_2)
\end{bmatrix}.
$$

## 1. CDF 的定义与性质

### 1.1 配置空间中的“距离场”

配置空间距离场 CDF（可视作对碰撞集合边界的距离函数）可定义为

$$
\mathrm{CDF}(\mathbf{q}) \triangleq \min_{\mathbf{q}^*\in \partial\mathcal{C}_{\text{obs}}} \|\mathbf{q}-\mathbf{q}^*\|_2.
$$

- $\partial\mathcal{C}_{\text{obs}}$ 是碰撞集合的边界（“刚好接触”）。
- 该定义给出的是**到接触边界的欧式距离**。

> 说明（实现相关）：理论上 CDF 可以扩展为**带符号距离**，常见约定是可行区为正、碰撞内部为负、边界为 0。
>
> 在本仓库的 `CDF2D.calculate_cdf` 实现中，距离的绝对值来自“到边界配置集合的最小欧式距离”，并通过任务空间的 SDF（对机器人表面采样点取最小）来确定符号，因此返回的是**带符号 CDF**。
>
> 但在工程使用时仍建议：碰撞/可行性判定以几何碰撞检测（或环境的 clearance 函数）为准，而不是仅靠 CDF 正负号，原因是离散化、插值、以及最小值/梯度的不可微点会带来数值伪影。

### 1.2 梯度的几何意义

若 $\mathrm{CDF}$ 在 $\mathbf{q}$ 处可微，则其梯度

$$
\nabla_{\mathbf{q}}\mathrm{CDF}(\mathbf{q})
$$

指向“CDF 增长最快”的方向。直观上：
- 沿 $+\nabla\mathrm{CDF}$ 小步移动，会让配置在 C-space 里**远离接触边界**；
- 等高线（level set）$\{\mathbf{q}\mid \mathrm{CDF}(\mathbf{q})=c\}$ 的切向方向与法向方向正交；梯度就是法向。

更形式化地，若轨迹 $\mathbf{q}(t)$ 希望严格沿同一条等高线运动（即 $\mathrm{CDF}(\mathbf{q}(t))$ 保持常数），则由链式法则有

$$
\frac{d}{dt}\mathrm{CDF}(\mathbf{q}(t))=\nabla\mathrm{CDF}(\mathbf{q})^\top\dot{\mathbf{q}}=0.
$$

因此“沿等高线方向”的速度 $\dot{\mathbf{q}}$ 必须与梯度 $\nabla\mathrm{CDF}(\mathbf{q})$ 正交。

在二维情形（$n=2$）下，设梯度 $\mathbf{g}=\nabla\mathrm{CDF}(\mathbf{q})=[g_1,g_2]^\top$，其单位法向为

$$
\hat{\mathbf{n}}=\frac{\mathbf{g}}{\|\mathbf{g}\|+\varepsilon}.
$$

则等高线切向可由 $90^\circ$ 旋转得到（给出一个常用约定）：

$$
\hat{\mathbf{t}}_{\text{cw}}=\begin{bmatrix} \hat{n}_2 \\ -\hat{n}_1 \end{bmatrix},\qquad
\hat{\mathbf{t}}_{\text{ccw}}=\begin{bmatrix} -\hat{n}_2 \\ \hat{n}_1 \end{bmatrix}=-\hat{\mathbf{t}}_{\text{cw}}.
$$

其中 $\varepsilon>0$ 用于避免 $\|\mathbf{g}\|\approx 0$ 时的数值不稳定。注意：若你的关节坐标正方向或绘图坐标系与本文约定相反，顺/逆时针的符号可能需要整体取反。

---

## 2. CDF 的计算（与 `calculate_cdf` 的对应）

### 2.1 计算目标

实践中我们需要的是：给定配置 $\mathbf{q}$，计算
- 标量值：$d = \mathrm{CDF}(\mathbf{q})$
- 梯度：$\mathbf{g}=\nabla_{\mathbf{q}}\mathrm{CDF}(\mathbf{q})$

以支持基于梯度/等高线的轨迹生成。

### 2.2 在线/离线计算方式

常见 CDF 计算有两类思路：

1) **在线优化（online）**：直接求解“最接触”的配置 $\mathbf{q}^*$ 或接触点，使得 $\|\mathbf{q}-\mathbf{q}^*\|$ 最小。这通常需要每次查询解一个优化问题，精度高但代价大。

2) **离线网格/插值（offline grid）**：在配置空间网格上预计算或近似 CDF，再对查询点插值并通过自动微分/数值差分得到梯度。优点是查询快、稳定，适合在线控制/采样。

在本文实验脚本中使用的 `CDF2D.calculate_cdf(..., method='offline_grid', return_grad=True)` 属于第 2 类：
- `d` 为 CDF 值；
- `g` 由 torch 自动微分得到（或者在内部通过可微插值实现）。

### 2.3 `calculate_cdf` 的接口与实现要点（结合仓库代码）

函数签名（二维场景）：

```python
def calculate_cdf(self, q, obj_lists, method='online_computation', return_grad=False):
    ...
```

参数含义：
- `q`: 形状 `(Np, 2)` 的 torch 张量，表示一批配置点。
- `obj_lists`: 障碍物对象列表（如圆形、盒等），需要提供 `sample_surface()` 与 `signed_distance()` 等接口。
- `method`:
  - `online_computation`: 通过 `find_q()` 搜索得到“接触边界配置集合” $\{\mathbf{q}^*\}$，再计算 $\min\|\mathbf{q}-\mathbf{q}^*\|$。
  - `offline_grid`: 基于离线数据 `data2D.pt` 构造模板边界集合 `q_list_template`，查询更快。
- `return_grad`: 是否通过自动微分返回梯度 $\nabla_{\mathbf{q}}\mathrm{CDF}(\mathbf{q})$。

实现流程（概念层面）：
1) **构造/缓存边界配置集合**：
   - `offline_grid`：对障碍物表面采样点 $\mathbf{x}$，映射到任务空间网格索引，再从 `q_grid_template` 中取出对应的候选边界配置并去除 `inf`；结果缓存为 `q_list_template`。
   - `online_computation`：调用 `find_q(obj_lists)` 求解一批“使任务空间 SDF 接近 0”的配置，缓存为 `q_0_level_set`。
2) **距离计算**：

$$
d(\mathbf{q}) = \min_{\mathbf{q}^*}\|\mathbf{q}-\mathbf{q}^*\|_2.
$$

3) **符号确定（关键）**：调用 `inference_sdf(q, obj_lists)` 计算任务空间 signed distance（对机器人表面采样点与障碍物 signed distance 取最小），并用

$$
\mathrm{CDF}(\mathbf{q})=\begin{cases}
-d(\mathbf{q}), & \text{若 } \mathrm{SDF}(\mathbf{q})<0\\
\ \ d(\mathbf{q}), & \text{否则}
\end{cases}
$$

将碰撞内部赋为负号。

4) **梯度**：当 `return_grad=True` 时，使用 `torch.autograd.grad` 计算 $\nabla_{\mathbf{q}}\mathrm{CDF}(\mathbf{q})$。

实现注意：`min` 操作在“最近边界配置切换”的位置不可微，因此梯度可视作分段光滑意义下的（次）梯度；用于轨迹生成与局部控制通常是足够的。

### 2.4 离线数据 `data2D.pt` 与生成（`generate_data`）

`offline_grid` 方法依赖 `data2D.pt`（位于 `algorithms/cdf_rl_mppi/cdf_2d/` 同目录），其含义是：对任务空间网格点 $\mathbf{p}$，存储一组“机器人表面经过该点”的候选配置集合 $\{\mathbf{q}\}$，形成从任务空间到配置集合的离线模板。

仓库代码在初始化 `CDF2D` 时会尝试加载该文件；若缺失，则会触发 `generate_data()` 生成过程。生成过程依赖 `torchmin`（用于 L-BFGS 求解），因此若环境中未安装 `torchmin`，需要：
- 安装 `torchmin` 后再生成；或
- 直接放置已生成的 `data2D.pt`（推荐，便于复现实验）。

---

## 3. 基于 CDF 的轨迹计算：两阶段“匹配等高线 + 沿等高线滑动”

### 3.1 目标

给定
- 起点配置 $\mathbf{q}_0$
- 目标配置 $\mathbf{q}_g$（可由 IK 得到，本文 Robot2D 为两连杆解析 IK）

希望生成一条关节空间轨迹，使其具备以下行为：

1) 先“跨等高线”移动，使当前配置的 CDF 值与目标配置一致：
$$\mathrm{CDF}(\mathbf{q}) \to \mathrm{CDF}(\mathbf{q}_g).$$

2) 再在该目标等高线上，沿指定方向（例如顺时针）滑动，并最终到达目标附近（或至少在同一条等高线分支上接近）。

### 3.2 阶段 1：沿梯度匹配目标等高线（Level Matching）

定义目标等高线值

$$
d_g \triangleq \mathrm{CDF}(\mathbf{q}_g).
$$

考虑能量函数

$$
E(\mathbf{q}) = \tfrac{1}{2}(\mathrm{CDF}(\mathbf{q}) - d_g)^2.
$$

对其做梯度下降（连续时间）得到

$$
\dot{\mathbf{q}} = -k_\ell\,(\mathrm{CDF}(\mathbf{q})-d_g)\,\nabla\mathrm{CDF}(\mathbf{q}),
$$

其中 $k_\ell>0$ 为增益。该式含义：
- 若当前 $d < d_g$，则 $\mathrm{CDF}(\mathbf{q})-d_g<0$，方向变为 $+\nabla\mathrm{CDF}$，推动 CDF 增大；
- 若当前 $d > d_g$，则沿 $-\nabla\mathrm{CDF}$ 使 CDF 降低；
- 直到 $|d-d_g|\le \varepsilon_\ell$（容差）认为匹配成功。

离散化（步长 $h$）可写为

$$
\mathbf{q}_{k+1} = \mathrm{wrap}\Big(\mathbf{q}_k + h\,\dot{\mathbf{q}}_k\Big).
$$

### 3.3 阶段 2：沿目标等高线顺/逆时针滑动（Contour Sliding）

在阶段 2，我们希望：
- “主要”沿切向走（不改变 CDF）；
- 同时用法向小修正把 $\mathrm{CDF}(\mathbf{q})$ 拉回 $d_g$。

纯“沿等高线方向运动”的速度（连续时间）可以写成

$$
\dot{\mathbf{q}} = v_t\,\hat{\mathbf{t}}(\mathbf{q}),\qquad \text{且 }\nabla\mathrm{CDF}(\mathbf{q})^\top\dot{\mathbf{q}}=0,
$$

其中 $v_t$ 是切向速度标量，$\hat{\mathbf{t}}$ 为单位切向（二维时可取顺/逆时针之一）。

离散化（步长 $h$）的单步更新为

$$
\mathbf{q}_{k+1}=\mathrm{wrap}\big(\mathbf{q}_k+h\,v_t\,\hat{\mathbf{t}}(\mathbf{q}_k)\big).
$$

令
- $\mathbf{g}=\nabla\mathrm{CDF}(\mathbf{q})$ 并单位化得到 $\hat{\mathbf{n}}=\mathbf{g}/\|\mathbf{g}\|$（法向）
- 切向（二维）取 $\hat{\mathbf{t}}$ 为 $\hat{\mathbf{n}}$ 旋转 $90^\circ$ 后的单位向量：
  - 逆时针：$\hat{\mathbf{t}}_{\text{ccw}}$
  - 顺时针：$\hat{\mathbf{t}}_{\text{cw}}=-\hat{\mathbf{t}}_{\text{ccw}}$

阶段 2 的速度场定义为

$$
\dot{\mathbf{q}} = k_t\,\hat{\mathbf{t}}_{\text{cw}}\; -\; k_n\,(\mathrm{CDF}(\mathbf{q})-d_g)\,\hat{\mathbf{n}},
$$

其中
- $k_t>0$ 控制沿等高线的滑动速度；
- $k_n>0$ 控制“贴住等高线”的强度（相当于对 level error 的比例反馈）。

停止条件可以是任务空间距离足够小：

$$
\|\mathbf{x}(\mathbf{q})-\mathbf{x}_g\| \le \varepsilon_x
$$

也可以简单设置最大步数（用于可视化/对比实验）。

### 3.4 伪代码

```text
Input: start q0, goal qg
Compute dg = CDF(qg)
q ← q0

# Phase 1: level matching
repeat
    d, g ← CDF(q), ∇CDF(q)
    n ← normalize(g)
    q ← wrap(q - h * k_level * (d - dg) * n)
until |d - dg| ≤ level_tol  or steps exceed budget

# Phase 2: clockwise sliding on level dg
repeat
    d, g ← CDF(q), ∇CDF(q)
    n ← normalize(g)
    t_cw ← normalize( -rotate90(n) )   # or choose sign by convention
    q ← wrap(q + h * (k_tan * t_cw - k_norm * (d - dg) * n))
until ‖x(q)-x_goal‖ ≤ task_tol or steps exceed max

Output: trajectory {q}
```

---

## 4. 实践注意事项（写论文时建议说明）

1) **Unsigned CDF 与碰撞判定分离**：`calculate_cdf` 可能返回 unsigned 距离，因此碰撞/可行性应由几何碰撞检测（clearance）保证。

2) **障碍物膨胀一致性**：若环境碰撞检测使用了安全距离 $r_{safe}$，则 CDF 计算时也应对障碍物半径做一致膨胀（例如圆半径 $r \leftarrow r+r_{safe}+r_{inflate}$）。

3) **梯度退化与步长**：在梯度接近 0 的区域，切向/法向的数值会不稳定；实践中需要对 $\|g\|$ 做下界保护，并限制步长 $h$。

4) **等高线连通性**：即使 $\mathrm{CDF}(\mathbf{q})=d_g$ 匹配成功，阶段 2 仍可能因为等高线的连通分支问题而无法“沿同一条曲线”到达目标邻域。实验上可通过改变初始匹配点/方向、或允许短暂跨越等高线来缓解。

---

## 5. 其他常用轨迹算子：投影与边界附近的“射击”（Projection / Shooting）

除了“匹配等高线 + 沿等高线滑动”的两阶段方法，CDF/SDF 的梯度也常用来构造两类基础算子，便于论文中对比或做可视化。

### 5.1 投影到边界（Projection to level set）

若希望将配置 $\mathbf{q}$ 投影到某个等值面（最常见是边界 $\mathrm{CDF}=0$），一种一阶近似是

$$
\mathbf{q}_{\text{proj}} = \mathbf{q} - \frac{\mathrm{CDF}(\mathbf{q})}{\|\nabla\mathrm{CDF}(\mathbf{q})\|+\varepsilon}\,\nabla\mathrm{CDF}(\mathbf{q}).
$$

直观上，它沿法向把点“推回”到目标等值面附近。实现中常配合多次迭代与步长裁剪。

### 5.2 边界附近的切向射击（Geodesic-like Shooting）

希望沿边界附近运动（近似保持 $\mathrm{CDF}$ 常数）时，可取切向速度

$$
\dot{\mathbf{q}} \propto \hat{\mathbf{t}}(\mathbf{q}) = \mathrm{rot90}\Big(\frac{\nabla\mathrm{CDF}(\mathbf{q})}{\|\nabla\mathrm{CDF}(\mathbf{q})\|+\varepsilon}\Big),
$$

二维离散更新为

$$
\mathbf{q}_{k+1}=\mathrm{wrap}\big(\mathbf{q}_k + h\,\hat{\mathbf{t}}(\mathbf{q}_k)\big).
$$

仓库中的示例实现允许用 `SDF` 或 `CDField`（即 CDF）来提供梯度，然后取其正交方向做更新。

---

## 6. 与本文代码实现的对应

本文仓库中，Robot2D 的两阶段轨迹生成与可视化脚本位于：
- experiments/robot2d/test_robot2d_cdf_contours_and_collision.py

CDF2D 的核心实现位于：
- algorithms/cdf_rl_mppi/cdf_2d/cdf.py

其中关键参数含义（对应上文符号）：
- `--level_gain` 对应 $k_\ell$
- `--level_tol` 对应 $\varepsilon_\ell$
- `--cdf_tangent_gain` 对应 $k_t$
- `--cdf_normal_gain` 对应 $k_n$
- `--contour_step` 对应离散步长 $h$
- `--goal_task_tol` 对应 $\varepsilon_x$

### 6.1 工程分工建议：SAC 学习用 CDF，在线 MPPI/RL-MPPI 用最小间隙

在同一个系统里同时使用 “CDF” 与 “最小间隙（clearance）” 并不矛盾：两者分别更适合承担 **学习阶段的密集塑形信号** 与 **在线规划阶段的快速/可靠安全评估**。

1) **SAC（训练/离线学习）更适合用 CDF 作为 shaping**

- CDF 本质是 C-space 中到碰撞边界的距离：它提供一个连续标量信号，可作为“离障碍物有多安全”的度量，用于奖励塑形（reward shaping）或代价塑形。
- 相比仅用“是否碰撞 / 是否低于阈值”的稀疏惩罚，CDF 产生的密集信号更有利于策略学习在全局上形成对障碍物几何的“认知”，降低探索样本需求。
- 训练阶段通常允许更高的计算预算（例如离线网格 `offline_grid`），并且对少量数值伪影更鲁棒（可通过经验回放平均）。

2) **MPPI/RL-MPPI（在线规划/控制）更适合用最小间隙 clearance**

- 在线采样规划需要在每个控制周期评估大量 rollouts，清晰的几何 clearance 计算通常更快、更稳定，也更容易与环境的碰撞检测保持严格一致。
- `calculate_cdf` 涉及“最近边界配置集合”的离散近似与 `min` 操作，梯度/符号可能在不可微点、离线模板插值处产生数值伪影；因此更适合作为软塑形，而不建议作为在线硬约束的唯一依据。
- 工程上建议：在线安全判定以 `clearance(q) < 0` 为准（配合安全膨胀），CDF 仅作为训练塑形或可视化/分析工具。

3) **一致性要求（两种信号必须对齐的部分）**

- 无论使用 CDF 还是 clearance，障碍物的安全膨胀（例如 $r\leftarrow r+r_{safe}$）必须保持一致，否则会出现“训练认为安全但在线认为碰撞”（或反之）的分布偏移。
- 推荐做法是：碰撞检测与在线规划统一走 clearance；训练的 CDF shaping 也使用同样的膨胀半径配置。

在本文仓库中：
- CDF 的实现见 `algorithms/cdf_rl_mppi/cdf_2d/cdf.py`，示例可视化见 `experiments/robot2d/test_robot2d_cdf_contours_and_collision.py`。
- Robot2D 障碍物环境的在线碰撞判定与最小间隙计算由环境几何函数提供；并可选启用“沿等高线偏移”的速度整形（默认基于 clearance）。

---

## 7. 障碍物处理：基于最小间隙的“沿等高线偏移”速度整形（论文可直接引用）

本节描述一种工程上稳定且易实现的避障处理：当机器人接近障碍物时，不直接让速度“朝目标硬冲”，而是将关节速度在局部进行整形，使其**沿最小间隙（clearance）的等高线方向滑动**，从而在一阶近似下避免继续降低间隙，表现为“类似斥力导致轨迹偏离碰撞方向”。

### 7.1 任务空间障碍物与安全膨胀

设障碍物为圆形集合 $\{\mathcal{O}_j\}$，每个障碍物由中心 $\mathbf{o}_j\in\mathbb{R}^2$ 与半径 $r_j$ 给出。为引入安全裕度，使用膨胀半径

$$
	ilde r_j = r_j + r_{\text{safe}}.
$$

其中 $r_{\text{safe}}>0$ 为安全距离。

### 7.2 机器人与障碍物的最小间隙（Minimum Clearance）

对平面 $n$ 连杆机器人，令连杆端点序列为

$$
\mathbf{p}_0(\mathbf{q}),\mathbf{p}_1(\mathbf{q}),\dots,\mathbf{p}_n(\mathbf{q})\in\mathbb{R}^2,
$$

其中 $\mathbf{p}_0$ 为基座点，$\mathbf{p}_n$ 为末端执行器。

对第 $i$ 根连杆，其几何可用线段 $[\mathbf{p}_i,\mathbf{p}_{i+1}]$ 近似。定义点到线段的欧氏距离为

$$
\mathrm{dist}(\mathbf{o}, [\mathbf{a},\mathbf{b}])=\big\|\mathbf{o}-(\mathbf{a}+t^*(\mathbf{b}-\mathbf{a}))\big\|_2,\quad
t^* = \mathrm{clip}_{[0,1]}\Big(\frac{(\mathbf{o}-\mathbf{a})^\top(\mathbf{b}-\mathbf{a})}{\|\mathbf{b}-\mathbf{a}\|_2^2}\Big).
$$

则配置 $\mathbf{q}$ 下的最小间隙函数定义为

$$
c(\mathbf{q}) \triangleq \min_{j}\min_{i\in\{0,\dots,n-1\}}
\Big(\mathrm{dist}(\mathbf{o}_j,[\mathbf{p}_i(\mathbf{q}),\mathbf{p}_{i+1}(\mathbf{q})]) - \tilde r_j\Big).
$$

若 $c(\mathbf{q})<0$ 则表示碰撞（穿透深度为 $-c(\mathbf{q})$）。

> 备注：某些任务仅检查末端（EEF）与障碍物的间隙，此时可用
> $$c_{\text{eef}}(\mathbf{q})=\min_j (\|\mathbf{x}(\mathbf{q})-\mathbf{o}_j\|_2-\tilde r_j).$$

### 7.3 间隙梯度与一阶“避免继续靠近”条件

设关节速度为 $\dot{\mathbf{q}}$，则间隙的一阶变化率为

$$
\dot c(\mathbf{q}) = \nabla c(\mathbf{q})^\top \dot{\mathbf{q}}.
$$

当 $\dot c(\mathbf{q})<0$ 时，表示当前速度在一阶近似下**正在降低间隙**（更接近障碍物）。因此，一个自然的避障处理是：在接近障碍物区域内，约束/修正速度使得

$$
\nabla c(\mathbf{q})^\top \dot{\mathbf{q}} \ge 0,
$$

或者更强一些，令其近似为 0（沿等高线滑动）。

### 7.4 沿等高线偏移：速度在切空间的投影（核心公式）

设策略/控制器给出的“期望关节速度”为 $\dot{\mathbf{q}}_{\text{des}}$（可由动作积分得到）。当接近障碍物且 $\dot c < 0$ 时，将其投影到 $c(\mathbf{q})$ 的等高线切空间：

1) 计算梯度 $\mathbf{g}=\nabla c(\mathbf{q})$。

2) 计算接近率 $\dot c_{\text{des}} = \mathbf{g}^\top \dot{\mathbf{q}}_{\text{des}}$。

3) 若 $\dot c_{\text{des}}<0$，则定义切向速度

$$
\dot{\mathbf{q}}_{\text{tan}} = \dot{\mathbf{q}}_{\text{des}} - \frac{\mathbf{g}^\top\dot{\mathbf{q}}_{\text{des}}}{\|\mathbf{g}\|_2^2+\delta}\,\mathbf{g},
$$

其中 $\delta>0$ 为数值稳定项。

该投影满足（忽略 $\delta$ 时严格成立）

$$
\mathbf{g}^\top\dot{\mathbf{q}}_{\text{tan}} \approx 0,
$$

即速度在一阶近似下沿 $c(\mathbf{q})$ 的等高线方向滑动，从而“不会继续逼近障碍物”。

### 7.5 可选：法向斥力项（增加间隙）

仅做切向投影可以避免继续靠近，但不保证快速远离。可加入一个弱斥力法向项

$$
\dot{\mathbf{q}}_{\text{rep}} = k_{\text{rep}}\,w(c)\,\frac{\mathbf{g}}{\|\mathbf{g}\|_2+\varepsilon},
$$

其中 $k_{\text{rep}}\ge 0$，$\varepsilon>0$。

### 7.6 距离相关的平滑融合权重

为避免远离障碍物时也被过度影响，引入基于间隙的融合权重 $w(c)\in[0,1]$。给定两个阈值 $c_{\text{start}}>c_{\text{full}}$：

$$
w(c)=
\begin{cases}
0,& c\ge c_{\text{start}}\\
1,& c\le c_{\text{full}}\\
\frac{c_{\text{start}}-c}{c_{\text{start}}-c_{\text{full}}},& \text{otherwise.}
\end{cases}
$$

最终整形速度（连续时间形式）为

$$
\dot{\mathbf{q}}_{\text{safe}} = (1-w)\,\dot{\mathbf{q}}_{\text{des}} + w\,(\dot{\mathbf{q}}_{\text{tan}} + \dot{\mathbf{q}}_{\text{rep}}).
$$

### 7.7 梯度的数值计算（有限差分）

由于 $c(\mathbf{q})$ 由“min over obstacles & links”构成，解析梯度较难且在切换点不可微。实践中可用中心差分近似：

$$
\frac{\partial c}{\partial q_k}(\mathbf{q}) \approx \frac{c(\mathbf{q}+\epsilon\mathbf{e}_k)-c(\mathbf{q}-\epsilon\mathbf{e}_k)}{2\epsilon},
$$

其中 $\mathbf{e}_k$ 是第 $k$ 个基向量，$\epsilon>0$ 为小扰动。

### 7.8 伪代码（速度层避障整形）

```text
Input: q, qdot_des
c ← clearance(q)
w ← weight(c)
if w == 0: return qdot_des

g ← grad_clearance_fd(q)
if ||g|| small: return qdot_des

cdot_des ← g^T qdot_des
qdot_tan ← qdot_des
if cdot_des < 0:
  qdot_tan ← qdot_des - (cdot_des/(||g||^2+δ)) g

qdot_rep ← k_rep * w * g/(||g||+ε)
qdot_safe ← (1-w) qdot_des + w (qdot_tan + qdot_rep)
return clip(qdot_safe)
```

### 7.9 解释与性质（论文表述建议）

- 该方法本质上是对“最小间隙场 $c(\mathbf{q})$”做局部几何利用：当检测到速度使得间隙下降（$\dot c<0$）时，移除其在法向 $\nabla c$ 上的分量，从而在一阶近似下满足 $\dot c\approx 0$，实现沿等高线滑动。
- 与传统人工势场（直接沿 $+\nabla c$ 施加斥力）相比，切向投影更像“绕着障碍走”，不容易出现把系统强行顶开导致的大幅振荡；同时可以叠加一个弱法向项作为保守安全裕度。

### 7.10 CBF/安全滤波（Safety Filter）等价表述（可选，偏控制理论写法）

将最小间隙 $c(\mathbf{q})$ 视作安全函数（安全集合为 $\mathcal{S}=\{\mathbf{q}: c(\mathbf{q})\ge 0\}$）。在速度层可采用一阶控制屏障函数（Control Barrier Function, CBF）形式约束：

$$
\dot c(\mathbf{q}) = \nabla c(\mathbf{q})^\top \dot{\mathbf{q}} \ge -\alpha\big(c(\mathbf{q})\big),
$$

其中 $\alpha(\cdot)$ 为类-$\mathcal{K}$ 函数（常用线性 $\alpha(c)=\gamma c$，$\gamma>0$）。

在工程实现中，可将其作为一个“安全滤波器”：对任意期望速度 $\dot{\mathbf{q}}_{\text{des}}$，求最近的可行速度

$$
\dot{\mathbf{q}}_{\text{safe}} = \arg\min_{\dot{\mathbf{q}}}\; \|\dot{\mathbf{q}}-\dot{\mathbf{q}}_{\text{des}}\|_2^2
\quad\text{s.t.}\quad \nabla c(\mathbf{q})^\top \dot{\mathbf{q}} \ge -\alpha\big(c(\mathbf{q})\big).
$$

这是一个带单个线性不等式约束的二次规划（QP）。当约束不激活（即 $\nabla c^\top\dot{\mathbf{q}}_{\text{des}}\ge -\alpha(c)$）时，解为 $\dot{\mathbf{q}}_{\text{safe}}=\dot{\mathbf{q}}_{\text{des}}$。

当约束激活时，QP 的闭式解等价于将 $\dot{\mathbf{q}}_{\text{des}}$ 投影到半空间边界：

$$
\dot{\mathbf{q}}_{\text{safe}} = \dot{\mathbf{q}}_{\text{des}} + \lambda\,\nabla c(\mathbf{q}),
\qquad
\lambda = \frac{-\alpha(c)-\nabla c(\mathbf{q})^\top\dot{\mathbf{q}}_{\text{des}}}{\|\nabla c(\mathbf{q})\|_2^2+\delta},
$$

其中 $\delta>0$ 为数值稳定项。

特别地，若取“硬等高线”形式 $\alpha(c)\equiv 0$，则当 $\nabla c^\top\dot{\mathbf{q}}_{\text{des}}<0$ 时，上式退化为

$$
\dot{\mathbf{q}}_{\text{safe}} = \dot{\mathbf{q}}_{\text{des}} - \frac{\nabla c(\mathbf{q})^\top\dot{\mathbf{q}}_{\text{des}}}{\|\nabla c(\mathbf{q})\|_2^2+\delta}\,\nabla c(\mathbf{q}),
$$

与本文第 7.4 节的“沿等高线偏移（速度投影）”完全一致。

---

## 8. SAC 训练奖励设计：CDF 作为主要安全项（论文可直接引用）

本节给出 Robot2D 场景中用于 SAC 的奖励设计：以“到达目标”为主任务，同时以 **CDF 作为主要安全项（dense safety shaping）** 引导策略在训练中学会避障。

重要区别：
- **训练时不使用基于 CDF/clearance 的速度整形**（不改变动力学/控制律），即不采用第 7 节的“沿等高线偏移”作为安全滤波；
- 避障能力来自奖励中的 CDF 安全项（以及碰撞终止/惩罚），因此策略学到的是“自身的避障行为”。

### 8.1 主任务项：到达目标（progress shaping）

令末端与目标的距离为

$$
d_t \triangleq \|\mathbf{x}(\mathbf{q}_t)-\mathbf{x}_g\|_2.
$$

使用进度型奖励（距离变小则奖励为正）：

$$
r_{\text{goal}}(t)=k_g\,(d_{t-1}-d_t),\qquad k_g>0.
$$

该项提供稳定的“朝目标收敛”梯度信号，避免纯终点奖励过稀疏。

### 8.2 安全项：基于 CDF 的铰链（hinge）惩罚（主要安全项）

令 $\mathrm{CDF}(\mathbf{q}_t)$ 表示配置 $\mathbf{q}_t$ 在配置空间中到碰撞边界（接触集合）的距离度量（可按第 1–2 节计算）。给定安全裕度阈值 $d_{\text{m}}>0$（CDF margin），定义安全惩罚为

$$
r_{\text{safe}}(t)=-\lambda\,\big[\max(0,\; d_{\text{m}}-\mathrm{CDF}(\mathbf{q}_t))\big]^p,
$$

其中 $\lambda>0$ 为安全权重，$p\ge 1$ 为惩罚幂次（常用 $p=2$ 以在靠近边界时更陡峭，呈现“barrier-like”效果）。

直观解释：
- 当 $\mathrm{CDF}(\mathbf{q}_t)\ge d_{\text{m}}$ 时，安全项为 0，不干扰主任务；
- 当进入低 CDF 区域（靠近接触/碰撞边界）时，惩罚随“缺口” $(d_{\text{m}}-\mathrm{CDF})$ 非线性增长，从而把策略推回到更安全的 C-space 区域。

> 工程注意：碰撞/可行性判定仍建议由几何 clearance（第 7.2 节）完成；CDF 用作密集塑形信号，而非唯一硬判据。

（可选）为鼓励更大的安全裕度，可加入一个有界的正向奖励项：

$$
r_{\text{bonus}}(t)=\eta\,\min\big(\mathrm{CDF}(\mathbf{q}_t),\; d_{\text{cap}}\big),\qquad \eta\ge 0.
$$

一般建议 $\eta$ 取较小或置 0，避免安全奖励过强导致策略“原地躲避”而不去追目标。

### 8.3 碰撞终止与碰撞惩罚

碰撞事件由环境的 clearance 判定（例如 $c(\mathbf{q}_t)<0$）。训练中通常采用：
- **发生碰撞立即终止 episode**；
- 并可叠加一次性惩罚
$$
r_{\text{col}}(t)=-\kappa\,\mathbb{I}[c(\mathbf{q}_t)<0],\qquad \kappa\ge 0.
$$

在仅使用“碰撞终止”而不额外惩罚时，可取 $\kappa=0$。

### 8.4 总奖励

综合上述项，一个论文中可直接呈现的奖励形式为

$$
r_t = r_{\text{goal}}(t) + r_{\text{safe}}(t) + r_{\text{bonus}}(t) + r_{\text{col}}(t).
$$

其中 $r_{\text{safe}}$（CDF hinge 惩罚）被设计为主要安全项。

### 8.5 与本文代码实现的对应（参数映射）

Robot2D 环境实现（`env/envrobot2d_obstacles.py`）中对应参数为：
- `obstacle_shaping='cdf'`：启用 CDF 作为软安全项（训练默认）。
- `cdf_margin`：$d_{\text{m}}$。
- `cdf_penalty`：$\lambda$。
- `cdf_penalty_power`：$p$。
- `cdf_bonus_gain`：$\eta$，`cdf_bonus_cap`：$d_{\text{cap}}$（可选）。
- `collision_penalty`：$\kappa$（可选）。
- `cdf_method` 与 `cdf_obstacle_inflate`：控制 CDF 的计算方式及障碍物膨胀一致性。

SAC 训练脚本（`experiments/robot2d/train_sac_robot2d_online.py`）中显式设置 `contour_avoidance=False`，确保训练过程不进行“沿等高线偏移”的速度整形，避免将安全性“写死在动力学里”，从而让策略真正学习避障。

### 8.6 SAC 训练超参数表（可直接贴论文/README）

下表汇总 Robot2D 在线交互训练（SAC）的核心超参数。为避免歧义，标注每个参数是 **CLI 可配置**（脚本参数）还是 **代码固定**（实现中写死/默认）。

| 分组 | 参数 | 默认值 | 来源 | 说明 |
|---|---|---:|---|---|
| 训练流程 | `total_steps` | 250000 | CLI | 总交互步数 |
| 训练流程 | `start_steps` | 10000 | CLI | 初期随机探索步数 |
| 训练流程 | `update_after` | 2000 | CLI | 多少步后开始梯度更新 |
| 训练流程 | `update_every` | 1 | 代码固定 | 每隔多少环境步触发更新 |
| 训练流程 | `updates_per_step` | 1 | 代码固定 | 每次触发更新时做几次梯度更新 |
| 训练流程 | `batch_size` | 256 | CLI | 每次更新的 batch 大小 |
| 训练流程 | `replay_size` | 400000 | CLI | 回放池容量 |
| 训练流程 | `max_ep_steps` | 450 | CLI | 单回合最大步数（env.max_steps） |
| 训练流程 | `eval_every` | 25000 | CLI | 周期性评估 + 保存（0 则不评估） |
| 训练流程 | `seed` | 42 | CLI | NumPy/PyTorch 随机种子 |
| SAC 算法 | `hidden_dim` | 256 | 代码固定 | Actor/Critic 两层 MLP 宽度 |
| SAC 算法 | `learning_rate` | 3e-4 | 代码固定 | Actor/Critic（以及自动熵温度）学习率 |
| SAC 算法 | `gamma` | 0.99 | 代码固定 | 折扣因子 |
| SAC 算法 | `tau` | 0.005 | 代码固定 | Target 网络软更新系数 |
| SAC 熵项 | `auto_entropy_tuning` | True | CLI（默认开） | 开：学习 $\alpha$，目标熵为 $-\mathrm{dim}(\mathbf{u})$ |
| SAC 熵项 | `alpha` | 0.2 | CLI | 仅在关闭自动熵时生效的固定 $\alpha$ |
| 优化细节 | `weight_decay` | 1e-5 | 代码固定 | Adam 权重衰减 |
| 优化细节 | `grad_clip_norm` | 1.0 | 代码固定 | 梯度裁剪阈值 |
| 策略分布 | `log_std_min/max` | -20 / 2 | 代码固定 | 高斯策略 log-std 截断 |
| 观测编码 | `normalize_state` | 1 | CLI | 归一化：$q/\pi$, $\dot q/\dot q_{\max}$, $x_g/\mathrm{reach}$ |
| 观测编码 | `include_obstacles_in_obs` | 1 | CLI | 是否在 obs 中拼接障碍物编码 |
| 观测编码 | `max_obstacles_in_obs` | 4 | CLI | 最大编码障碍物数量（不足补 0，超出截断） |
| 环境 | `dt` | 0.02 | 环境默认 | 仿真步长（未在该训练脚本暴露为 CLI） |
| 环境 | `qd_max` | 4.0 | CLI | 关节速度上限 |
| 环境 | `qdd_max` | 8.0 | CLI | 关节加速度上限 |
| 环境 | `reset_noise` | 0.20 | CLI | Reset 扰动 |
| 环境 | `reach_threshold` | 0.15 | CLI | 末端到目标距离阈值 |
| 安全/终止 | `terminate_on_collision` | True | 代码固定 | 碰撞立即终止（硬约束由 clearance 判定） |
| 动力学整形 | `contour_avoidance` | False | 代码固定 | SAC 训练强制关闭（不改变动力学） |
| 奖励 shaping | `obstacle_shaping` | cdf | CLI | 软安全项来源：cdf/clearance/both/none |
| 奖励 shaping | `cdf_method` | offline_grid | CLI | CDF 计算方式 |
| 奖励 shaping | `cdf_obstacle_inflate` | 0.0 | CLI | CDF 额外膨胀 |
| 奖励 shaping | `cdf_margin` | 0.25 | CLI | hinge margin：$d_m$ |
| 奖励 shaping | `cdf_penalty` | 800.0 | CLI | hinge 权重：$\lambda$ |
| 奖励 shaping | `cdf_penalty_power` | 2.0 | CLI | hinge 幂次：$p$ |
| 奖励 shaping | `cdf_bonus_gain/cap` | 0.0 / 2.0 | CLI | 可选安全裕度 bonus：$\eta, d_{cap}$ |
| 奖励 shaping | `collision_penalty` | 0.0 | CLI | 可选碰撞额外惩罚：$\kappa$ |

注：上述“代码固定”项来自 `algorithms/sac/sac_utils.py` 与训练脚本中 `SACAgent(...)` 的构造参数；若需对其进行论文消融（如改变网络宽度、学习率、$\gamma$、$\tau$ 等），建议将其也显式暴露为 CLI 并记录到 checkpoint 的元数据中。
