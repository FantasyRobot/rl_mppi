下面给出可直接放入论文“方法（Method）/算法（Algorithm）”部分的文字，包含：MPPI 基本原理、RL‑MPPI（SAC 引导 MPPI）的改进、以及改进带来的优势。符号和叙述尽量与控制/优化论文常见写法一致，可按你的章节结构微调。

---

## 1. 记号与问题定义（Notation）

考虑离散时间系统

$$x_{t+1}=f(x_t,u_t),\quad x_t\in\mathbb{R}^{n},\ u_t\in\mathbb{R}^{m},$$

其中 $x_t$ 为状态，$u_t$ 为控制输入。给定目标（例如目标点位置）与约束（例如动作/状态边界、障碍物等），在滚动时域 $H$ 上最小化累计代价

$$S(\mathbf U; x_t)=\sum_{i=0}^{H-1}\ell(x_{t+i},u_{t+i})+\ell_f(x_{t+H}),\quad \mathbf U=\{u_t,\dots,u_{t+H-1}\}.$$

在本文的 2D 小球任务中，状态可写为 $x=[p_x,p_y,v_x,v_y]$，动作为归一化加速度 $a\in[-1,1]^2$（环境内部再缩放为物理加速度）。代价函数 $\ell(\cdot)$ 同时编码“到达目标”和“避障安全”等偏好。

---

## 2. MPPI 基本原理（Model Predictive Path Integral）

MPPI 属于采样型模型预测控制方法。其关键思想是：

1) 维护一个名义控制序列（mean / nominal）$\bar{\mathbf U}=\{\bar u_0,\dots,\bar u_{H-1}\}$；
2) 在名义序列附近采样多条随机扰动控制序列；
3) 对每条采样序列进行前向 rollout 得到轨迹并计算代价；
4) 用路径积分（Gibbs）权重对采样进行重要性加权，从而更新名义控制序列；
5) 输出更新后的第一步控制作为当前动作，并将控制序列 shift 进入下一时刻重复上述过程。

### 2.1 采样与代价评估

在时刻 $t$，对 $k=1,\dots,K$ 条样本采样噪声

$$\epsilon^k_i\sim\mathcal N(0,\Sigma),\quad i=0,\dots,H-1,$$

并构造候选控制

$$u^k_i=\bar u_i+\epsilon^k_i.$$

对每条候选序列执行 rollout 得到轨迹 $\{x^k_{t+i}\}_{i=0}^{H}$，累计代价为 $S_k=S(\mathbf U^k;x_t)$。

### 2.2 路径积分权重与控制更新

MPPI 采用指数型权重强调低代价轨迹（温度系数 $\lambda>0$ 控制“择优强度”）：

$$w_k=\exp\left(-\frac{1}{\lambda}(S_k-S_{\min})\right),\quad \tilde w_k=\frac{w_k}{\sum_{j=1}^K w_j},\quad S_{\min}=\min_j S_j.$$

用归一化权重对噪声做加权平均，更新名义控制序列（常见形式）：

$$\bar u_i\leftarrow \bar u_i+\sum_{k=1}^{K}\tilde w_k\,\epsilon^k_i,\quad i=0,\dots,H-1.$$

最终控制输入为 $u_t=\bar u_0$，并在下一时刻将序列滚动（shift）为 $\{\bar u_1,\dots,\bar u_{H-1},\bar u_{H-1}\}$（或附加一个末端启发式/零输入）。

> 直观解释：MPPI 在名义控制附近做“随机搜索”，但不是选最小代价轨迹，而是用指数权重把更新方向集中到一组低代价样本上，从而得到更平滑的在线优化控制。

---

## 3. RL‑MPPI：用 SAC 策略引导的 MPPI（RL‑Driven MPPI）

标准 MPPI 的采样中心通常来自上一时刻的名义控制或简单启发式（如 PD 控制）。当任务具有复杂约束（障碍物、窄通道）或需要更长时域的绕行时，纯随机采样容易出现方差大、样本利用率低的问题。

为提高采样效率，RL‑MPPI 引入离线训练得到的策略（本文使用 Soft Actor‑Critic, SAC）作为先验控制律 $\pi_\theta$，并将其输出作为名义序列（proposal / nominal）：

$$u^{\mathrm{RL}}_i = \pi_\theta(\hat x_{t+i}),\quad i=0,\dots,H-1,$$

其中 $\hat x_{t+i}$ 可以来自名义 rollout（用 $u^{\mathrm{RL}}$ 递推得到）或用当前状态近似构造。随后仍按 MPPI 的方式在该先验附近采样：

$$u^k_i = u^{\mathrm{RL}}_i + \epsilon^k_i.$$

最后仍使用路径积分权重 $\tilde w_k$ 对噪声进行加权平均更新控制。等价地，可将 RL‑MPPI 理解为：**SAC 提供“有经验的初解/搜索中心”，MPPI 用显式代价函数做在线修正与优化。**

### 3.1 与实现相关的关键工程一致性

为避免训练/测试分布不匹配，RL‑MPPI 在使用 SAC 策略时保持与训练一致的状态预处理（例如相对目标的归一化/固定边界归一化），并可从 checkpoint 中读取训练时使用的目标参数（如目标位置）以确保评测一致。

---

## 4. 代价函数（到达 + 避障）示例写法

本文在 2D 小球任务中采用如下结构的代价：

$$\ell(x,u)=\ell_{\text{goal}}(x)+\ell_{\text{ctrl}}(u)+\ell_{\text{obs}}(x),\qquad \ell_f(x)=\beta\,\ell_{\text{goal}}(x).$$

- 到达目标项（示例）：
  $$\ell_{\text{goal}}(x)=\|p-p_g\|_2^2.$$
- 控制正则（示例）：
  $$\ell_{\text{ctrl}}(u)=\rho\|u\|_2^2.$$
- 避障项可由以下组件构成（按需要取舍）：
  1) **影响区惩罚（margin penalty）**：当与障碍物距离小于影响距离时增加二次惩罚，鼓励提前绕行；
  2) **碰撞硬惩罚（collision cost）**：若预测轨迹进入障碍物（clearance < 0），施加极大代价；
  3) **安全膨胀（safety inflation）**：将障碍物半径按 $r+r_{\mathrm{safety}}$ 膨胀，减少“擦边”导致的碰撞；
  4) **视线惩罚（LOS penalty）**：当障碍物靠近“当前位置到目标点”的连线时惩罚，促使更早绕行。

上述避障代价对 MPPI 与 RL‑MPPI 一致使用，可确保对比公平：差异主要来自采样分布（是否由 RL 先验引导）。

---

## 5. 优势分析（Why RL‑MPPI helps）

相较标准 MPPI，RL‑MPPI 的主要优势可概括为：

1) **采样效率更高、实时性更强**：RL 先验将采样集中在更可能低代价/可行的区域，重要性采样方差更小；在相同样本数 $K$ 下更容易得到有效轨迹，或在达到同等性能时可减少 $K$。

2) **对分布外场景更鲁棒**：SAC 策略在目标变化、障碍物加入等情况下可能出现策略偏差；MPPI 的在线代价优化可对 RL 输出进行纠偏，避免直接冲入不可行区域。

3) **更容易融合显式安全偏好/约束**：避障、安全裕度等可以主要通过代价函数表达，不必完全依赖重新训练策略；这使系统更可解释、也更利于迁移与调参。

4) **学习先验 + 在线优化互补**：RL 提供经验知识（快速给出合理控制方向），MPPI 提供在线最优性提升（针对当前状态与环境即时重算），两者结合通常能在复杂场景下获得更稳定的到达与避障行为。

---

## 6. Algorithm（可直接放论文）

**Algorithm 1: RL‑MPPI (SAC‑guided MPPI)**

**Input:** 当前状态 $x_t$，预测时域 $H$，样本数 $K$，温度 $\lambda$，噪声协方差 $\Sigma$，动力学 $f$，代价函数 $\ell,\ell_f$，先验策略 $\pi_\theta$。

1. （先验名义）构造名义控制序列 $\bar{\mathbf U}$：对 $i=0,\dots,H-1$，令 $\bar u_i\leftarrow\pi_\theta(\hat x_{t+i})$，并用 $f$ 递推得到名义预测状态 $\hat x_{t+i+1}$。
2. （采样）对 $k=1,\dots,K$：采样 $\epsilon^k_i\sim\mathcal N(0,\Sigma)$，令 $u^k_i\leftarrow\bar u_i+\epsilon^k_i$。
3. （rollout）对每条样本序列 $\mathbf U^k$：用 $f$ 从 $x_t$ rollout 得到 $\{x^k_{t+i}\}$ 并计算代价 $S_k=\sum_{i=0}^{H-1}\ell(x^k_{t+i},u^k_i)+\ell_f(x^k_{t+H})$。
4. （权重）令 $S_{\min}=\min_k S_k$，计算 $w_k=\exp(-(S_k-S_{\min})/\lambda)$ 并归一化得到 $\tilde w_k$。
5. （更新）对 $i=0,\dots,H-1$：$\bar u_i\leftarrow \bar u_i+\sum_k \tilde w_k\,\epsilon^k_i$。
6. （执行与滚动）执行 $u_t\leftarrow\bar u_0$，并将 $\bar{\mathbf U}$ shift 一步进入下一时刻。

---

如果你告诉我论文是中文还是英文，以及你希望“代价函数”写成更具体的公式（例如明确 obstacle margin/collision/LOS 的数学表达），我可以把第 4 节进一步公式化成完全可发表版本，并对齐你的符号体系。